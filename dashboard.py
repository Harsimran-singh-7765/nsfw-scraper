import os
import subprocess
import json
import random
import threading
import queue
import time
import shutil
import requests
from bs4 import BeautifulSoup
import concurrent.futures
from urllib.parse import urljoin
from flask import Flask, render_template, Response, jsonify, request, send_from_directory
from model_utils import get_classifier

app = Flask(__name__)
# Project root
BASE_DIR = os.path.dirname(os.path.abspath(__name__))
RAW_DATA_DIR = os.path.join(BASE_DIR, 'raw_data')
UPLOAD_FOLDER = os.path.join(BASE_DIR, 'uploads')
FEEDBACK_DIR = os.path.join(BASE_DIR, 'feedback_data')
FEEDBACK_JSON = os.path.join(FEEDBACK_DIR, 'feedback.json')
INGEST_DIR = os.path.join(BASE_DIR, 'temp_ingest')
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(FEEDBACK_DIR, exist_ok=True)
os.makedirs(INGEST_DIR, exist_ok=True)

CATEGORIES = ["neutral", "drawings", "sexy", "porn", "hentai"]

# Create subdirs for each category in feedback
for cat in CATEGORIES:
    os.makedirs(os.path.join(FEEDBACK_DIR, cat), exist_ok=True)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# 2GB in kilobytes matching the bash scripts
MAX_SIZE_KB = 2097152 

# Global sub-process tracking
scraper_process = None

# Smart Ingest Global State
ingest_queue = queue.Queue()
ingest_logs = [] # List of strings to stream
ingest_status = {"active": False, "total": 0, "processed": 0}
stop_ingest = False

def resolve_redgifs(url):
    try:
        r = requests.get("https://api.redgifs.com/v2/auth/temporary", timeout=5)
        token = r.json().get("token")
        
        gif_id = url.rstrip('/').split('/')[-1]
        headers = {"Authorization": f"Bearer {token}"}
        r = requests.get(f"https://api.redgifs.com/v2/gifs/{gif_id}", headers=headers, timeout=5)
        if r.status_code == 200:
            urls = r.json().get("gif", {}).get("urls", {})
            return urls.get("hd") or urls.get("sd")
    except:
        pass
    return None

def download_file(url, dest):
    ua = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
    try:
        r = requests.get(url, headers={"User-Agent": ua}, timeout=15, stream=True)
        if r.status_code == 200:
            with open(dest, 'wb') as f:
                for chunk in r.iter_content(chunk_size=8192):
                    f.write(chunk)
            return True
    except: pass
    return False

def get_reddit_urls(reddit_url):
    ua = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
    json_url = reddit_url.rstrip('/') + ".json?limit=100"
    urls = []
    try:
        r = requests.get(json_url, headers={"User-Agent": ua}, timeout=10)
        data_list = r.json() if isinstance(r.json(), list) else [r.json()]
        for data in data_list:
            posts = data.get('data', {}).get('children', [])
            for post in posts:
                pdata = post['data']
                if pdata.get('url'):
                    u = pdata['url']
                    if 'redgifs.com' in u:
                        resolved = resolve_redgifs(u)
                        if resolved: urls.append(resolved)
                    elif any(u.lower().endswith(ext) for ext in ('.jpg', '.jpeg', '.png', '.gif', '.mp4')):
                        urls.append(u)
                
                if pdata.get('is_video') and pdata.get('media', {}).get('reddit_video', {}).get('fallback_url'):
                    urls.append(pdata['media']['reddit_video']['fallback_url'].replace('&amp;', '&'))
                    
                if pdata.get('is_gallery') and pdata.get('media_metadata'):
                    for item in pdata['media_metadata'].values():
                       if item['status'] == 'valid':
                           if item['e'] == 'Image':
                               urls.append(item['s']['u'].replace('&amp;', '&'))
                           elif item['e'] == 'AnimatedImage':
                               urls.append(item['s']['mp4'].replace('&amp;', '&'))
    except: pass
    return list(set(urls))

def _fetch_generic_urls(url):
    ua = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
    urls = []
    try:
        r = requests.get(url, headers={"User-Agent": ua}, timeout=15)
        html_content = r.text
        
        # 1. Regex approach: Find raw URLs hidden in JSON/JS vars
        import re
        raw_matches = re.findall(r'(https?://[^\s"\'<>*?\\!]*?\.(?:jpg|jpeg|png|webp|gif))', html_content, re.IGNORECASE)
        for match in raw_matches:
            urls.append(match)
            
        # 2. BeautifulSoup approach: Find standard DOM elements
        soup = BeautifulSoup(html_content, 'html.parser')
        
        for img in soup.find_all('img'):
            src = img.get('data-src') or img.get('src') or img.get('data-original')
            if src:
                src = urljoin(url, src)
                if not src.startswith('data:'):
                    urls.append(src)
        
        for source in soup.find_all('source'):
            src = source.get('srcset') or source.get('data-srcset')
            if src:
                first_url = src.split(',')[0].split(' ')[0]
                first_url = urljoin(url, first_url)
                if not first_url.startswith('data:'):
                    urls.append(first_url)
                    
    except Exception as e:
        print(f"Generic scrape error for {url}: {e}")
    return urls

def get_generic_image_urls(base_url):
    all_raw_urls = set()
    urls_to_fetch = [base_url]
    
    # Special pagination handling for infinite-scroll sites
    if 'pornpics' in base_url.lower():
        # Scrape up to 15 pages worth of images (~300 images)
        for offset in range(20, 300, 20):
            sep = "&" if "?" in base_url else "?"
            urls_to_fetch.append(f"{base_url}{sep}offset={offset}")
            
    for target_url in urls_to_fetch:
        page_urls = set(_fetch_generic_urls(target_url))
        if not page_urls:
            break
        all_raw_urls.update(page_urls)

    valid_urls = []
    # Deduplicate and filter out obvious garbage or tiny icons
    for u in all_raw_urls:
        if len(u) > 15 and not any(bad in u.lower() for bad in ['favicon', 'logo', 'icon', 'tracker', 'pixel']):
            # Special case for PornPics to upgrade thumbnails to high resolution
            if 'pornpics.de' in u or 'pornpics.com' in u:
                u = u.replace('/460/', '/1280/')
                u = u.replace('/300/', '/1280/')
            valid_urls.append(u)
            
    return valid_urls

def classify_worker(classify_q, ingest_id, force_category=None):
    classifier = get_classifier()
    while True:
        filepath = classify_q.get()
        if filepath is None:
            break
            
        try:
            vid_exts = ('.mp4', '.gif', '.webm', '.mov')
            img_exts = ('.jpg', '.jpeg', '.png', '.webp')
            file_lower = filepath.lower()
            
            files_to_classify = []
            if file_lower.endswith(vid_exts):
                ingest_logs.append(f"LOG:🎬 Extracting {os.path.basename(filepath)[:15]}...")
                base_name = os.path.splitext(os.path.basename(filepath))[0]
                frame_pattern = os.path.join(os.path.dirname(filepath), f"frame_{base_name}_%03d.jpg")
                try:
                    subprocess.run(["ffmpeg", "-v", "quiet", "-hide_banner", "-i", filepath, "-r", "1", "-vframes", "3", frame_pattern], check=True, timeout=60)
                    os.remove(filepath)
                    for f in os.listdir(os.path.dirname(filepath)):
                        if f.startswith(f"frame_{base_name}_") and f.endswith('.jpg'):
                            files_to_classify.append(os.path.join(os.path.dirname(filepath), f))
                except Exception as e:
                    ingest_logs.append(f"LOG:❌ FFMPEG Error: {e}")
            elif file_lower.endswith(img_exts):
                files_to_classify.append(filepath)
                
            for src_path in files_to_classify:
                cat = None
                if force_category and force_category in CATEGORIES:
                    cat = force_category
                else:
                    prediction = classifier.predict(src_path)
                    if "error" not in prediction:
                        cat = prediction["class"]
                
                if cat:
                    ingest_status["processed"] += 1
                    
                    fname = os.path.basename(src_path)
                    dest_cat_dir = os.path.join(RAW_DATA_DIR, cat, "IMAGES")
                    os.makedirs(dest_cat_dir, exist_ok=True)
                    prefix = f"force_{cat}" if force_category else "auto"
                    dest_filename = f"{prefix}_{ingest_id}_{fname}"
                    dest_path = os.path.join(dest_cat_dir, dest_filename)
                    shutil.move(src_path, dest_path)
                    
                    ingest_logs.append(f"PRED:{cat}|{dest_filename}")
        except Exception as e:
            ingest_logs.append(f"LOG:❌ Classify Error: {e}")
            
        classify_q.task_done()

def ingest_worker():
    global ingest_status, stop_ingest
    while True:
        task = ingest_queue.get()
        if task is None: break
        
        if isinstance(task, str):
            url = task
            force_category = None
        else:
            url = task.get("url")
            force_category = task.get("force_category")
        
        stop_ingest = False
        ingest_status["active"] = True
        ingest_status["total"] = 0
        ingest_status["processed"] = 0
        
        if force_category:
            ingest_logs.append(f"LOG:🚀 Starting targeted ingest for {url} --> [{force_category.upper()}]")
        else:
            ingest_logs.append(f"LOG:🚀 Starting AI smart ingest for {url}")
        
        try:
            classifier = get_classifier()
            ripme_jar = os.path.join(BASE_DIR, 'scripts', 'ripme.jar')
            ingest_id = int(time.time())
            current_ingest_path = os.path.join(INGEST_DIR, str(ingest_id))
            os.makedirs(current_ingest_path, exist_ok=True)
            
            classify_q = queue.Queue()
            classifier_thread = threading.Thread(target=classify_worker, args=(classify_q, ingest_id, force_category))
            classifier_thread.start()
            
            if "reddit.com" in url or "redgifs.com" in url:
                ingest_logs.append("LOG:👽 Native API Fallback mode...")
                if "redgifs.com" in url:
                    resolved = resolve_redgifs(url)
                    urls = [resolved] if resolved else []
                else:
                    urls = get_reddit_urls(url)
                    
                ingest_logs.append(f"LOG:🔍 Found {len(urls)} media links.")
                ingest_status["total"] = len(urls)
                
                def download_and_queue(u, idx):
                    if stop_ingest: return
                    ext = u.split('?')[0].split('.')[-1]
                    if len(ext) > 4: ext = 'mp4' if 'mp4' in u else 'jpg'
                    dest = os.path.join(current_ingest_path, f"dl_{idx}.{ext}")
                    if download_file(u, dest):
                        if stop_ingest: return
                        ingest_logs.append(f"LOG:⬇️ Got chunk! ({idx+1}/{len(urls)})")
                        classify_q.put(dest)
                
                with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
                    futures = [executor.submit(download_and_queue, u, i) for i, u in enumerate(urls)]
                    while any(f.running() or not f.done() for f in futures):
                        if stop_ingest:
                            for f in futures: f.cancel()
                            break
                        time.sleep(0.5)
                    concurrent.futures.wait(futures)
                    
            else:
                # 1. Try RipMe first
                ua = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
                ripme_jar = os.path.join(BASE_DIR, 'scripts', 'ripme.jar')
                cmd = ["java", f"-Dhttp.agent={ua}", "-jar", ripme_jar, "--skip404", "--no-prop-file", "--ripsdirectory", current_ingest_path, "--url", url]
                ingest_logs.append("LOG:📦 Running Universal Scraper (RipMe)...")
                
                process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
                polled_files = set()
                
                # Check for output, but don't hang forever if RipMe is failing silently
                start_wait = time.time()
                while True:
                    if stop_ingest:
                        process.terminate()
                        ingest_logs.append("LOG:🛑 Ingest stopping...")
                        break
                        
                    retcode = process.poll()
                    found_new = False
                    for root, dirs, files in os.walk(current_ingest_path):
                        for f in files:
                            filepath = os.path.join(root, f)
                            if filepath not in polled_files and not f.endswith('.tmp'):
                                size_a = os.path.getsize(filepath)
                                time.sleep(0.1)
                                size_b = os.path.getsize(filepath)
                                if size_a == size_b and size_a > 0:
                                    polled_files.add(filepath)
                                    ingest_logs.append(f"LOG:⬇️ Got chunk! {f[:20]}")
                                    classify_q.put(filepath)
                                    found_new = True
                    
                    if found_new:
                        start_wait = time.time() # Reset timeout if we are successfully downloading
                        
                    if retcode is not None:
                        break
                        
                    # If 20 seconds pass without RipMe finding any files, it likely failed/blocked. 
                    if len(polled_files) == 0 and time.time() - start_wait > 20.0:
                        process.terminate()
                        ingest_logs.append("LOG:⚠️ RipMe timeout. Attempting HTML parser fallback...")
                        break
                        
                    time.sleep(0.5)
                
                # If RipMe failed to find anything, run the Generic BeautifulSoup scrape
                if not stop_ingest and len(polled_files) == 0:
                    ingest_logs.append("LOG:🕷️ Running HTML fallback parser...")
                    urls = get_generic_image_urls(url)
                    ingest_logs.append(f"LOG:🔍 Found {len(urls)} media links via HTML.")
                    ingest_status["total"] = len(urls)
                    
                    def download_and_queue_generic(u, idx):
                        if stop_ingest: return
                        ext = u.split('?')[0].split('.')[-1]
                        if len(ext) > 4 or not ext.isalnum(): ext = 'jpg'
                        dest = os.path.join(current_ingest_path, f"html_dl_{idx}.{ext}")
                        if download_file(u, dest):
                            if stop_ingest: return
                            ingest_logs.append(f"LOG:⬇️ Got generic chunk! ({idx+1}/{len(urls)})")
                            classify_q.put(dest)
                    
                    if urls:
                        with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
                            futures = [executor.submit(download_and_queue_generic, u, i) for i, u in enumerate(urls)]
                            while any(f.running() or not f.done() for f in futures):
                                if stop_ingest:
                                    for f in futures: f.cancel()
                                    break
                                time.sleep(0.5)
                            concurrent.futures.wait(futures)

            # Shutdown classifier
            classify_q.put(None)
            classifier_thread.join()
            
            shutil.rmtree(current_ingest_path, ignore_errors=True)
            ingest_logs.append(f"LOG:✅ Finished! Processed {ingest_status['processed']} items.")
            
        except Exception as e:
            ingest_logs.append(f"LOG:❌ Fatal Error: {str(e)}")
        
        if ingest_queue.empty():
            ingest_status["active"] = False
        ingest_queue.task_done()

# Start the worker thread
thread = threading.Thread(target=ingest_worker, daemon=True)
thread.start()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/test')
def test_page():
    return render_template('test_model.html')

@app.route('/raw/<path:filename>')
def serve_raw(filename):
    return send_from_directory(RAW_DATA_DIR, filename)

@app.route('/api/models')
def get_models():
    """Returns the list of available models and the currently active one"""
    classifier = get_classifier()
    return jsonify(classifier.get_available_models())

@app.route('/api/set_model', methods=['POST'])
def set_model():
    """Switches the active model"""
    data = request.get_json()
    model_id = data.get("model_id")
    if not model_id:
        return jsonify({"error": "No model_id provided"}), 400
        
    classifier = get_classifier()
    success = classifier.load_model(model_id)
    if success:
        return jsonify({"status": "success", "active_model": model_id})
    else:
        return jsonify({"error": f"Failed to load model {model_id}"}), 500

@app.route('/api/data_insights')
def data_insights():
    """Provides an overview of the entire dataset composition"""
    total_files = 0
    total_kb = 0
    auto_count = 0
    manual_count = 0
    
    cat_breakdown = {}
    
    for cat in CATEGORIES:
        img_dir = os.path.join(RAW_DATA_DIR, cat, "IMAGES")
        cat_auto = 0
        cat_manual = 0
        cat_size = 0
        
        if os.path.exists(img_dir):
            try:
                files = os.listdir(img_dir)
                for f in files:
                    f_path = os.path.join(img_dir, f)
                    if os.path.isfile(f_path):
                        total_files += 1
                        cat_size += os.path.getsize(f_path)
                        if f.startswith("auto_"):
                            auto_count += 1
                            cat_auto += 1
                        else:
                            manual_count += 1
                            cat_manual += 1
            except: pass
            
        total_kb += (cat_size // 1024)
        cat_breakdown[cat] = {
            "auto": cat_auto,
            "manual": cat_manual,
            "total": cat_auto + cat_manual
        }
        
    return jsonify({
        "total_files": total_files,
        "total_mb": round(total_kb / 1024, 1),
        "auto_count": auto_count,
        "manual_count": manual_count,
        "cat_breakdown": cat_breakdown,
        "auto_percentage": round((auto_count / total_files * 100), 1) if total_files > 0 else 0
    })

@app.route('/api/metrics')
def metrics():
    """Returns the current size of each category in KB"""
    sizes = {}
    for cat in CATEGORIES:
        cat_path = os.path.join(RAW_DATA_DIR, cat)
        images_dir = os.path.join(cat_path, "IMAGES")
        
        size_kb = 0
        file_count = 0
        
        if os.path.exists(images_dir):
            try:
                # `du -sk` returns size in KB
                result = subprocess.run(['du', '-sk', images_dir], capture_output=True, text=True)
                size_kb = int(result.stdout.split()[0])
            except Exception:
                size_kb = 0
                
            try:
                # `ls -f | wc -l` for blazing fast file counting
                count_result = subprocess.run(f'ls -f "{images_dir}" | wc -l', shell=True, capture_output=True, text=True)
                # Subtract 2 for . and .. 
                file_count = max(0, int(count_result.stdout.strip()) - 2)
            except Exception:
                file_count = 0
        
        sizes[cat] = {
            "current_kb": size_kb,
            "current_files": file_count,
            "max_kb": MAX_SIZE_KB,
            "percentage": min(100, (size_kb / MAX_SIZE_KB) * 100)
        }
    
    status = "Running" if scraper_process and scraper_process.poll() is None else "Stopped"
    return jsonify({"sizes": sizes, "status": status})

@app.route('/api/start', methods=['POST'])
def start_scraper():
    """Starts the Docker container scraper"""
    global scraper_process
    if scraper_process and scraper_process.poll() is None:
        return jsonify({"status": "Already running"}), 400
    
    data = request.get_json(silent=True) or {}
    category = data.get("category", "all")
    
    # Run the docker command inside scripts/
    cmd = ["docker", "run", "-v", f"{BASE_DIR}:/root/nsfw_data_scraper"]
    if category != "all":
        cmd.extend(["--env", f"TARGET_CATEGORY={category}"])
    
    cmd.extend(["docker_nsfw_data_scraper", "scripts/runall.sh"])
    
    scraper_process = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
        universal_newlines=True
    )
    return jsonify({"status": "Started"})

@app.route('/api/stop', methods=['POST'])
def stop_scraper():
    """Stops the scraper"""
    global scraper_process
    if scraper_process and scraper_process.poll() is None:
        scraper_process.terminate()
        return jsonify({"status": "Stopped"})
    return jsonify({"status": "Not running"}), 400

@app.route('/stream')
def stream():
    """SSE stream for docker logs"""
    def generate():
        global scraper_process
        if not scraper_process or scraper_process.poll() is not None:
            yield f"data: Engine is currently stopped.\n\n"
            return
            
        try:
            for line in iter(scraper_process.stdout.readline, ''):
                if line:
                    # properly format for SSE
                    data = line.strip().replace('\n', '')
                    yield f"data: {data}\n\n"
        except GeneratorExit:
            pass
            
    return Response(generate(), mimetype='text/event-stream')

@app.route('/api/random_image')
def random_image():
    """Returns a random image path from the RAW_DATA_DIR"""
    all_images = []
    for cat in CATEGORIES:
        img_dir = os.path.join(RAW_DATA_DIR, cat, "IMAGES")
        if os.path.exists(img_dir):
            files = [f for f in os.listdir(img_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png', '.webp'))]
            for f in files:
                all_images.append(f"{cat}/IMAGES/{f}")
    
    if not all_images:
        return jsonify({"error": "No images found"}), 404
        
    random_path = random.choice(all_images)
    return jsonify({"path": random_path, "url": f"/raw/{random_path}"})

@app.route('/api/sequential_image')
def sequential_image():
    """Returns an image from a specific category at a given offset"""
    category = request.args.get('category', 'sexy')
    offset = int(request.args.get('offset', 0))
    
    img_dir = os.path.join(RAW_DATA_DIR, category, "IMAGES")
    if not os.path.exists(img_dir):
        return jsonify({"error": f"Category folder {category} not found"}), 404
        
    # Get all images and sort them for consistent sequential access
    try:
        files = sorted([f for f in os.listdir(img_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png', '.webp'))])
    except Exception as e:
        return jsonify({"error": str(e)}), 500
    
    if not files:
        return jsonify({"error": "No images found in this category"}), 404
        
    if offset < 0:
        offset = 0
    elif offset >= len(files):
        return jsonify({"error": "End of list reached", "total": len(files)}), 404
        
    filename = files[offset]
    rel_path = f"{category}/IMAGES/{filename}"
    
    return jsonify({
        "path": rel_path,
        "url": f"/raw/{rel_path}",
        "index": offset,
        "total": len(files),
        "filename": filename
    })

@app.route('/api/classify', methods=['POST'])
def classify():
    """Runs inference on a specific image folder path"""
    data = request.get_json()
    img_rel_path = data.get("path")
    if not img_rel_path:
        return jsonify({"error": "No path provided"}), 400
        
    full_path = os.path.join(RAW_DATA_DIR, img_rel_path)
    if not os.path.exists(full_path):
        return jsonify({"error": f"File not found: {img_rel_path}"}), 404
        
    classifier = get_classifier()
    result = classifier.predict(full_path)
    return jsonify(result)

@app.route('/api/upload', methods=['POST'])
def upload_file():
    if 'image' not in request.files:
        return jsonify({"error": "No file part"}), 400
    file = request.files['image']
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400
    
    filename = "temp_upload.png" # Standardize for testing
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(filepath)
    
    classifier = get_classifier()
    result = classifier.predict(filepath)
    return jsonify(result)

@app.route('/api/classify_url', methods=['POST'])
def classify_url():
    data = request.get_json()
    url = data.get("url")
    if not url:
        return jsonify({"error": "No URL provided"}), 400
    
    try:
        import requests
        response = requests.get(url, timeout=10)
        if response.status_code == 200:
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], "temp_url.png")
            with open(filepath, 'wb') as f:
                f.write(response.content)
            
            classifier = get_classifier()
            result = classifier.predict(filepath)
            return jsonify(result)
        else:
            return jsonify({"error": f"Failed to fetch image: {response.status_code}"}), 400
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/feedback', methods=['POST'])
def save_feedback():
    data = request.get_json()
    img_rel_path = data.get("path")
    correct_label = data.get("correct_label")
    orig_pred = data.get("orig_pred")
    source_type = data.get("source_type") # 'local', 'upload', 'url'
    
    if not correct_label or not img_rel_path:
        return jsonify({"error": "Missing data"}), 400
        
    try:
        import shutil
        import time
        import json
        target_dir = os.path.join(FEEDBACK_DIR, correct_label)
        # Use current timestamp for uniqueness
        basename = os.path.basename(img_rel_path)
        filename = f"{int(time.time())}_{basename}"
        target_path = os.path.join(target_dir, filename)
        
        # Determine source
        if source_type == 'local':
            src_path = os.path.join(RAW_DATA_DIR, img_rel_path)
        elif source_type in ['upload', 'url']:
            src_filename = "temp_upload.png" if source_type == 'upload' else "temp_url.png"
            src_path = os.path.join(app.config['UPLOAD_FOLDER'], src_filename)
        else:
            return jsonify({"error": "Invalid source"}), 400

        if os.path.exists(src_path):
            shutil.copy(src_path, target_path)
            
            # Update feedback.json
            entry = {
                "timestamp": int(time.time()),
                "filename": filename,
                "orig_path": img_rel_path,
                "orig_pred": orig_pred,
                "ground_truth": correct_label,
                "source": source_type
            }
            
            feedback_data = []
            if os.path.exists(FEEDBACK_JSON):
                with open(FEEDBACK_JSON, 'r') as f:
                    try:
                        feedback_data = json.load(f)
                    except: pass
            
            feedback_data.append(entry)
            with open(FEEDBACK_JSON, 'w') as f:
                json.dump(feedback_data, f, indent=2)
        else:
            return jsonify({"error": "Original file lost"}), 404
                
        return jsonify({"status": "success", "saved_to": correct_label})
    except Exception as e:
        return jsonify({"error": str(e)}), 500
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/feedback_stats')
def get_feedback_stats():
    if not os.path.exists(FEEDBACK_JSON):
        return jsonify({"total": 0, "correct": 0, "accuracy": 0, "per_category": {}, "confusion": []})

    with open(FEEDBACK_JSON, 'r') as f:
        data = json.load(f)

    results = {
        "total": len(data),
        "correct": 0,
        "per_category": {cat: {"total": 0, "correct": 0} for cat in CATEGORIES},
        "confusion": []
    }
    
    confusion_map = {}

    for entry in data:
        gt = entry["ground_truth"]
        pred = entry["orig_pred"]
        
        if gt not in results["per_category"]:
            results["per_category"][gt] = {"total": 0, "correct": 0}

        results["per_category"][gt]["total"] += 1
        
        if gt == pred:
            results["correct"] += 1
            results["per_category"][gt]["correct"] += 1
        else:
            key = (gt, pred)
            confusion_map[key] = confusion_map.get(key, 0) + 1

    for (gt, pred), count in confusion_map.items():
        results["confusion"].append({"gt": gt, "pred": pred, "count": count})
    
    results["confusion"].sort(key=lambda x: x["count"], reverse=True)
    results["accuracy"] = (results["correct"] / results["total"] * 100) if results["total"] > 0 else 0
    
    return jsonify(results)

@app.route('/api/auto_ingest', methods=['POST'])
def auto_ingest():
    data = request.get_json()
    url = data.get("url")
    force_category = data.get("force_category")
    queue_mode = data.get("queue_mode", False)
    
    if not url:
        return jsonify({"error": "No URL provided"}), 400
    
    if ingest_status["active"] and not queue_mode:
        return jsonify({"error": "Another ingest is already in progress. Enable queueing."}), 400

    # Save URLs as seed data if a category is explicitly specified
    if force_category and force_category in CATEGORIES:
        try:
            seed_file = os.path.join(DATA_DIR, f"seed_{force_category}_urls.txt")
            with open(seed_file, "a") as f:
                f.write(url + "\n")
        except Exception as e:
            print(f"Failed to save seed data: {e}")

    # Clear logs only if we are starting fresh (not just appending to an active queue)
    if not ingest_status["active"]:
        ingest_logs.clear() 
        
    ingest_queue.put({"url": url, "force_category": force_category})
    
    return jsonify({
        "status": "queued", 
        "position": ingest_queue.qsize(), 
        "active": ingest_status["active"]
    })

@app.route('/api/ingest_stop', methods=['POST'])
def ingest_stop():
    global stop_ingest
    stop_ingest = True
    return jsonify({"status": "Stopping"})

@app.route('/ingest_stream')
def ingest_stream():
    def generate():
        last_idx = 0
        while True:
            if last_idx < len(ingest_logs):
                for i in range(last_idx, len(ingest_logs)):
                    yield f"data: {ingest_logs[i]}\n\n"
                last_idx = len(ingest_logs)
            
            if not ingest_status["active"] and last_idx >= len(ingest_logs):
                # Send a final 'DONE' signal if we just finished
                if last_idx > 0:
                   yield f"data: DONE\n\n"
                break
            time.sleep(0.5)
            
    return Response(generate(), mimetype='text/event-stream')

if __name__ == '__main__':
    app.run(debug=True, port=5000)
