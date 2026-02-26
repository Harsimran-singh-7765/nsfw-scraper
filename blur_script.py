import cv2
import glob
import os

images = glob.glob('assets/*.png') + glob.glob('assets/*.jpg')
for img_path in images:
    if "blurred" in img_path: continue
    img = cv2.imread(img_path)
    if img is None: continue
    
    h, w = img.shape[:2]
    
    # Validation Screenshots (center panel)
    if "18-45-50" in img_path or "18-46-21" in img_path:
        y1, y2 = int(h*0.35), int(h*0.65)
        x1, x2 = int(w*0.35), int(w*0.5)
        roi = img[y1:y2, x1:x2]
        if roi.size > 0:
            roi = cv2.GaussianBlur(roi, (151, 151), 50)
            img[y1:y2, x1:x2] = roi

    # Scraper/Ingestor Screenshots (bottom left thumb)
    if "18-44-19" in img_path or "18-45-35" in img_path:
        # Based on layout, thumb is center-left
        y3, y4 = int(h*0.45), int(h*0.58)
        x3, x4 = int(w*0.27), int(w*0.34)
        roi2 = img[y3:y4, x3:x4]
        if roi2.size > 0:
            roi2 = cv2.GaussianBlur(roi2, (99, 99), 30)
            img[y3:y4, x3:x4] = roi2

    out_name = os.path.basename(img_path).replace('.png', '_blurred.png').replace('.jpg', '_blurred.jpg')
    cv2.imwrite(os.path.join('assets', out_name), img)
    print(f"Blurred {out_name}")
