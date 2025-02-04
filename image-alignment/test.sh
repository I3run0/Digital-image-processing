#!/bin/bash

# Ensure the script stops if any command fails
set -e

usage() {
    echo "Usage: $0 -m <method> <input_dir> <output_dir>"
    echo "  -m <method>  : Method to use for alignment (HL for Hough Transformation, HP for Horizontal Projection)"
    echo "  <input_dir>  : Directory containing input images"
    echo "  <output_dir> : Directory to save the processed images"
    exit 1
}

while getopts ":m:" opt; do
    case $opt in
        m)
            METHOD=$OPTARG
            ;;
        *)
            usage
            ;;
    esac
done

shift $((OPTIND -1))

# Check if correct number of arguments is provided
if [ "$#" -ne 2 ]; then
    usage
fi

INPUT_DIR=$1
OUTPUT_DIR=$2

# Validate method argument
if [[ "$METHOD" != "HL" && "$METHOD" != "HP" ]]; then
    echo "Error: Invalid method. Choose 'HL' (Hough Transformation) or 'HP' (Horizontal Projection)."
    usage
fi

# Create the output directory if it doesn't exist
mkdir -p "$OUTPUT_DIR"

# Iterate over all image files in the input directory
for IMAGE_PATH in "$INPUT_DIR"/*.{png,jpg,jpeg}; do
    if [ -e "$IMAGE_PATH" ]; then
        IMAGE_NAME=$(basename "$IMAGE_PATH")
        IMAGE_NAME="${IMAGE_NAME%.*}"
        echo 
        if [ "$METHOD" == "HP" ]; then
            OUTPUT_IMAGE_PATH="$OUTPUT_DIR/${IMAGE_NAME}_rotated_hp.png"
            python3 src/horizontal_projection.py --histogram --ocr "$IMAGE_PATH" "$OUTPUT_IMAGE_PATH"
        elif [ "$METHOD" == "HL" ]; then
            OUTPUT_IMAGE_PATH="$OUTPUT_DIR/${IMAGE_NAME}_rotated_hl.png"
            python3 src/hough_transformation.py --draw-lines --ocr "$IMAGE_PATH" "$OUTPUT_IMAGE_PATH"
        fi
    fi
done
