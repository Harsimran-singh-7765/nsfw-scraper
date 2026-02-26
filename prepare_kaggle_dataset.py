import os
import shutil
import random
import argparse
import re
from pathlib import Path

def prepare_dataset(raw_data_dir, output_dir, balance_limit=None):
    raw_path = Path(raw_data_dir)
    out_path = Path(output_dir)
    
    if out_path.exists():
        print(f"Cleaning up existing output directory: {output_dir}")
        shutil.rmtree(out_path)
    
    out_path.mkdir(parents=True, exist_ok=True)
    
    categories = ["neutral", "drawings", "sexy", "porn", "hentai"]
    
    total_copied = 0
    
    for cat in categories:
        images_dir = raw_path / cat / "IMAGES"
        if not images_dir.exists():
            print(f"Skipping {cat}, no IMAGES folder found.")
            continue
            
        dest_dir = out_path / cat
        dest_dir.mkdir(parents=True, exist_ok=True)
        
        # Get all files
        files = [f for f in images_dir.iterdir() if f.is_file()]
        print(f"Found {len(files)} files in {cat}.")
        
        if balance_limit and len(files) > balance_limit:
            print(f"Balancing {cat}: randomly selecting {balance_limit} files.")
            selected_files = random.sample(files, balance_limit)
        else:
            selected_files = files
            
        copied_this_cat = 0
        for f in selected_files:
            try:
                # Sanitize filename by removing URL query parameters and invalid characters
                clean_name = f.name.split('?')[0].split('&')[0]
                clean_name = re.sub(r'[^a-zA-Z0-9_.-]', '_', clean_name)
                
                shutil.copy2(f, dest_dir / clean_name)
                copied_this_cat += 1
            except Exception as e:
                print(f"Failed to copy {f.name}: {e}")
                
        print(f"Copied {copied_this_cat} files for {cat}.")
        total_copied += copied_this_cat
        
    print(f"\nSuccessfully prepared Kaggle dataset at '{output_dir}'.")
    print(f"Total images: {total_copied}")
    print("\nNext steps for Kaggle:")
    print(f"1. Run: zip -r {output_dir}.zip {output_dir}")
    print(f"2. Go to Kaggle -> Datasets -> New Dataset -> Upload {output_dir}.zip")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prepare dataset for Kaggle")
    parser.add_argument("--raw_dir", default="raw_data", help="Directory containing scraped data")
    parser.add_argument("--out_dir", default="kaggle_dataset", help="Output directory for Kaggle format")
    parser.add_argument("--balance", type=int, default=None, help="Limit number of images per category to prevent class imbalance")
    args = parser.parse_args()
    
    prepare_dataset(args.raw_dir, args.out_dir, args.balance)
