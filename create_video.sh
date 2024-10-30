#chmod +x create_video.sh #to make it executable
#./create_video.sh


#!/bin/bash

# Create 'videos' directory if it doesn't exist
mkdir -p videos

# Extract the first and last filenames from images.txt
first_file=$(head -n 1 images.txt | awk -F"'" '{print $2}')
last_file=$(tail -n 1 images.txt | awk -F"'" '{print $2}')

# Extract the timestamp from the filenames (remove the directory and suffix)
first_timestamp=$(basename "$first_file" _sc_constellation.png)
last_timestamp=$(basename "$last_file" _sc_constellation.png)

# Create a dynamic title for the video
video_title="${first_timestamp}_to_${last_timestamp}_video.mp4"

# Save the video inside the 'videos' directory
ffmpeg -f concat -safe 0 -i images.txt -c:v libx264 -pix_fmt yuv420p -r 30 "videos/$video_title"

echo "Video created: videos/$video_title"