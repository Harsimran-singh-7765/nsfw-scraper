# NSFW Data Scraper and Refiner

[![Build Status](https://img.shields.io/badge/build-passing-brightgreen)](#)
[![Python Version](https://img.shields.io/badge/python-3.8%2B-blue)](#)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A modern tool to collect, refine, and manually supervise datasets of NSFW classifications. 

**Disclaimer:** Use with caution. The dataset is natively noisy and requires manual review. This tool is strictly intended for machine learning research purposes to help train robust moderation classifiers.

---

## Overview and Features

This suite of tools is designed to bridge the gap between large-scale raw data ingestion and highly curated, supervised datasets ready for neural network training. It operates in two major phases: Data Collection and Manual Refinement.

### 1. The Scraping Pipeline
* Scripts designed to download tens of thousands of images across 5 categories (`porn`, `hentai`, `sexy`, `neutral`, `drawings`).
* **Source Links:** Utilizes extensive URL lists tracked locally in `raw_data/*/*.txt`.

### 2. The Smart Feedback Dashboard
Because scraped datasets are inevitably noisy, we built a modern Flask-based UI to allow humans to rapidly review and correct classifications.
* **Streamlined UI:** A clean, high-contrast, edge-to-edge gallery built with pure HTML and Tailwind CSS.
* **Keyboard Hotkeys:** Rapid-fire categorization using keys (`1` for Drawings, `2` for Hentai, `3` for Neutral, etc).
* **Smart Ingestor:** Live-feed ingestion with Server-Sent Events (SSE) tracking and native fallback parsers.
* **Correction Pipeline:** `sync_feedback.py` safely merges human corrections back into the `raw_data` ground truth, cleaning up the dataset for the next training iteration.

---

## Screenshots

### Scraper Engine and Live Ingestor
The dashboard provides a real-time view into the scraping tasks, displaying active ingestion logs and a live preview of the incoming images as they are classified.

![Scraper Engine Logs](assets/Screenshot%20from%202026-02-26%2018-44-19_blurred.png)
*Live ingestion feed demonstrating the Reddit scraping fallback mode.*

![Scraper Category Targets](assets/Screenshot%20from%202026-02-26%2018-45-35_blurred.png)
*Engine dashboard showcasing category balance targets and local data insights.*

### Model Validation and Refinement
The validation view allows for rapid human-in-the-loop review of the classification model's predictions.

![Validation View](assets/Screenshot%20from%202026-02-26%2018-45-50_blurred.png)
*Detailed confidence breakdown panel and manual correction actions.*

![Prediction Results](assets/Screenshot%20from%202026-02-26%2018-46-21_blurred.png)
*Reviewing random local images and submitting ground-truth feedback.*

---

## Quick Start Guide

### Prerequisites
- Python 3.8+
- Docker (Optional, only required if using the legacy terminal scrapers)

#### Environment Setup
It is highly recommended to use a Python virtual environment to manage dependencies:
```bash
# Create the virtual environment
python3 -m venv venv

# Activate the virtual environment
source venv/bin/activate  # On Linux/macOS
# venv\Scripts\activate   # On Windows

# Install all dependencies
pip install -r requirements.txt
```

### 1. Running the Dashboard
To launch the Smart Feedback Dashboard and view the UI:
```bash
python dashboard.py
```
This will start a local Flask server. Open your web browser and navigate to `http://127.0.0.1:5000` to access the Scraper Engine, and `http://127.0.0.1:5000/test` to access the Model Validation suite.

### 2. (Alternative) Legacy Terminal Scraping
If you do not want to use the UI and simply want to scrape the raw data using the original terminal scripts, you can use the provided Docker container. This will run the `ripme` Java application to download tens of thousands of images based on the `urls.txt` sources.

```bash
# Build the docker container
docker build . -t docker_nsfw_data_scraper

# Run the scraping pipeline (This might take several hours, recommended to run overnight)
docker run -v $(pwd):/root/nsfw_data_scraper docker_nsfw_data_scraper scripts/runall.sh
```

### 2. Manual Refinement Workflow
1. Use the **Model Validation** tab to review images.
2. If an image is incorrectly categorized, select the correct category. 
3. This creates an entry in `feedback_data/feedback.json`.
4. Run the synchronization script to permanently fix the dataset:
```bash
python sync_feedback.py
```
This will physically move the images into the correct `raw_data` category folders based on your manual review.

### 3. Model Training
Once the dataset is cleaned to your specifications, a Colab-ready script is provided to train an EfficientNet-V2-S model.
* Extract `train_v2s.py` to your training environment.
* The script features advanced augmentations (RandomErasing, Cutout), Stratified splitting, and mixed-precision (AMP) training.
* It will automatically output a lightweight INT8-quantized `.pth` model ready for mobile deployment at the end of the epochs.

---

## Architecture

- **Backend:** Python + Flask (`dashboard.py`) 
- **Frontend:** Vanilla JS + Tailwind CSS (`templates/index.html`)
- **Data Sync:** `sync_feedback.py` processes JSON feedback files locally and mirrors changes into the master dataset directory.

## Credits

The original shell scraping scripts and the genesis of this taxonomy (the `urls.txt` structure and RipMe logic) were originally created by [alex000kim](https://github.com/alex000kim/nsfw_data_scraper). This repository expands upon that foundation with the Flask feedback UI, modernized Python 3 training logic, and the manual-correction syncing pipeline.
