#!/bin/bash

scripts_dir="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
base_dir="$(dirname "$scripts_dir")"
raw_data_dir="$base_dir/raw_data"

if [ -n "$TARGET_CATEGORY" ] && [ "$TARGET_CATEGORY" != "all" ]; then
	declare -a class_names=("$TARGET_CATEGORY")
else
	declare -a class_names=(
		"neutral"
		"drawings"
		"sexy"
		"porn"
		"hentai"
		)
fi

echo "--- Extracting Frames from Videos/GIFs ---"

for cname in "${class_names[@]}"
do
	images_dir="$raw_data_dir/$cname/IMAGES"
	
	if [ -d "$images_dir" ]; then
		echo "Processing videos in $cname..."
		
		# Find all mp4, gif, webm files
		find "$images_dir" -maxdepth 1 -type f \( -iname \*.mp4 -o -iname \*.gif -o -iname \*.webm \) | while read -r video_file; do
			
			base_name=$(basename "$video_file")
			filename="${base_name%.*}"
			
			echo "Extracting frames from $base_name..."
			
			# Use ffmpeg to extract 1 frame per second (-r 1) and output as JPG
			# -v quiet -hide_banner suppresses the massive ffmpeg output spam
			ffmpeg -v quiet -hide_banner -i "$video_file" -r 1 -vframes 3 "$images_dir/${filename}_frame_%03d.jpg"
			
			# If ffmpeg succeeded (exit code 0), delete the original video to save space
			if [ $? -eq 0 ]; then
				echo "Successfully extracted. Deleting original video $base_name."
				rm "$video_file"
			else
				echo "Failed to extract frames from $base_name."
			fi
		done
	fi
done

echo "Extraction complete."
