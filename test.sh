#!/bin/bash

# Define input and output filenames
output_directory="images_processed"

img_baboon="images/baboon.png"
img_butterfly="images/butterfly.png"
img_city="images/city.png"
img_house="images/house.png"

# Create output directory if it doesn't exist
mkdir -p "$output_directory"

# 1-1 mosaic
python3 1-1-mosaic.py "$img_baboon" "$output_directory/1-1-mosaic.png"

# 1-2 image combination
python3 1-2-combination.py "$img_baboon" "$img_butterfly" 0.5 0.5 "$output_directory/1-2-combination.png"

# 1-3 intensity transformation 
opt=(1 2 3 4 5)

for opt_value in "${opt[@]}"
do
    python3 1-3-intesity-transformation.py "$img_city" "$output_directory/1-3-intensity-transformation-${opt_value}.png" "$opt_value"
done

# 1-4 color 
opt=(1 2)

for opt_value in "${opt[@]}"
do
    python3 1-4-color.py "$img_city" "$output_directory/1-4-color-${opt_value}.png" "$opt_value"
done

# 1-5 bright adjust
opt=(1.5 2.5 3.5)

for opt_value in "${opt[@]}"
do
    python3 1-5-bright-adjust.py "$img_baboon" "$output_directory/1-5-bright-adjust-${opt_value}.png" "$opt_value"
done

# 1-6 quantization
opt=(256 64 32 16 8 4 2)

for opt_value in "${opt[@]}"
do
    python3 1-6-quantization.py "$img_baboon" "$output_directory/1-6-quantization-${opt_value}.png" "$opt_value"
done

# 1-7 bit plane
opt=(0 1 2 3 4 5 6 7)

for opt_value in "${opt[@]}"
do
    python3 1-7-bit-plane.py "$img_baboon" "$output_directory/1-7-bit_plane-${opt_value}.png" "$opt_value"
done

# mask
filters=(h1 h2 h3 h4 h5 h6 h7 h8 h9 h10 h11)

# 
# Loop through each filter
for filter_name in "${filters[@]}"
do
    # Run Python script with filter
    python3 1-8-mask.py "$img_baboon" "$output_directory/1-8-mask-${filter_name}.jpg" "${filter_name: -1}"
done
