# NSFW Data Scraper and Refiner

A modern tool to collect, refine, and manually supervise datasets of NSFW classifications. 

## ⚠️ Disclaimer
**Use with caution - the dataset is natively noisy and requires manual review.** This tool is intended for machine learning research purposes to help train robust moderation classifiers.

## 🚀 Features

### 1. The Scraping Pipeline
* Scripts designed to download tens of thousands of images across 5 categories (`porn`, `hentai`, `sexy`, `neutral`, `drawings`).
* **Source Links:** Utilizes extensive URL lists (e.g. from Reddit) tracked locally in `raw_data/*/*.txt`.

### 2. The Smart Feedback Dashboard
Because the scraped datasets are inevitably noisy, we built a modern UI to allow humans to rapidly review and correct classifications.
* **Streamlined UI:** A clean, high-contrast, edge-to-edge gallery built with pure HTML/Tailwind CSS.
* **Keyboard Hotkeys:** Rapid-fire categorization using keys (`1` for Drawings, `2` for Hentai, `3` for Neutral, etc).
* **Smart Ingestor:** Live-feed ingestion with Server-Sent Events (SSE) tracking and Reddit gallery fallbacks.
* **Correction Pipeline:** `sync_feedback.py` safely merges human corrections back into the `raw_data` ground truth, cleaning up the dataset for the next training iteration.

## 🛠️ Tech Stack & Architecture

- **Backend:** Python + Flask (`dashboard.py`) 
- **Frontend:** Vanilla JS + Tailwind CSS (`templates/index.html`)
- **Data Sync:** `sync_feedback.py` processes JSON feedback files locally and mirrors changes into the master dataset directory.

## 📦 Training

See `train_v2s.py` for a modern, Colab-ready training script implementing `tf_efficientnetv2_s` with data augmentation, label smoothing, and INT8 quantization capabilities.

## 🤝 Credits

The original shell scraping scripts and the genesis of this taxonomy (the `urls.txt` structure and RipMe logic) were originally created by [alex000kim](https://github.com/alex000kim/nsfw_data_scraper). This repository expands upon that foundation with the Flask feedback UI, modernized Python 3 training logic, and the manual-correction syncing pipeline.
