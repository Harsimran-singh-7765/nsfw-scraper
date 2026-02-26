import os
import json
import shutil
from tqdm import tqdm

# Paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
RAW_DATA_DIR = os.path.join(BASE_DIR, 'raw_data')
FEEDBACK_DIR = os.path.join(BASE_DIR, 'feedback_data')
FEEDBACK_JSON = os.path.join(FEEDBACK_DIR, 'feedback.json')
CATEGORIES = ["neutral", "drawings", "sexy", "porn", "hentai"]

def sync_feedback():
    if not os.path.exists(FEEDBACK_JSON):
        print("❌ Error: feedback.json not found!")
        return

    with open(FEEDBACK_JSON, 'r') as f:
        try:
            feedback_data = json.load(f)
        except Exception as e:
            print(f"❌ Error parsing JSON: {e}")
            return

    print(f"🔄 Starting synchronization for {len(feedback_data)} entries...")
    
    stats = {
        "moved": 0,
        "new": 0,
        "already_correct": 0,
        "skipped": 0,
        "errors": 0
    }

    for entry in tqdm(feedback_data, desc="Syncing"):
        filename = entry.get('filename')
        ground_truth = entry.get('ground_truth')
        source = entry.get('source')
        orig_path = entry.get('orig_path', 'unknown')

        if not filename or not ground_truth:
            stats["skipped"] += 1
            continue

        # 1. Locate the file in feedback_data
        src_path = os.path.join(FEEDBACK_DIR, ground_truth, filename)
        if not os.path.exists(src_path):
            # Try other category folders in feedback_data just in case
            found_alt = False
            for cat in CATEGORIES:
                alt_path = os.path.join(FEEDBACK_DIR, cat, filename)
                if os.path.exists(alt_path):
                    src_path = alt_path
                    found_alt = True
                    break
            if not found_alt:
                stats["errors"] += 1
                continue

        # 2. Determine target filename (strip timestamp prefix for 'local' source)
        # Many local files look like: 1772096691_7pyinmioh4b31.jpg
        # The original in raw_data was: 7pyinmioh4b31.jpg
        target_filename = filename
        if source == 'local' and orig_path != 'migrated' and '/' in orig_path:
            target_filename = os.path.basename(orig_path)
        
        # 3. Final target in raw_data
        dest_dir = os.path.join(RAW_DATA_DIR, ground_truth, "IMAGES")
        os.makedirs(dest_dir, exist_ok=True)
        dest_path = os.path.join(dest_dir, target_filename)

        # 4. If it was a 'local' correction, remove the OLD file in raw_data
        if source == 'local' and orig_path != 'migrated' and '/' in orig_path:
            old_raw_path = os.path.join(RAW_DATA_DIR, orig_path)
            if os.path.exists(old_raw_path) and old_raw_path != dest_path:
                try:
                    os.remove(old_raw_path)
                    stats["moved"] += 1
                except Exception as e:
                    print(f"Error removing old file {old_raw_path}: {e}")
            elif old_raw_path == dest_path:
                stats["already_correct"] += 1
            else:
                stats["new"] += 1
        else:
            stats["new"] += 1

        # 5. Copy the verified file to raw_data
        try:
            shutil.copy2(src_path, dest_path)
        except Exception as e:
            print(f"Error copying to {dest_path}: {e}")
            stats["errors"] += 1

    print("\n✨ Synchronization Summary:")
    print(f"✅ Total Processed: {len(feedback_data)}")
    print(f"🚚 Files Moved (Corrected): {stats['moved']}")
    print(f"➕ New Files Added: {stats['new']}")
    print(f"🎯 Already in Correct Folder: {stats['already_correct']}")
    print(f"❌ Errors: {stats['errors']}")
    print(f"⏭️  Skipped: {stats['skipped']}")

if __name__ == "__main__":
    sync_feedback()
