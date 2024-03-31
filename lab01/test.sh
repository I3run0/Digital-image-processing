#!/bin/bash

run_scripts() {
    local input_image="$1"
    local input_text="$2"
    local output_path="$3"
    local bit_plane="$4"

    shift 4
    local bit_plane_to_view=("$@")

    local output_image
    output_image="$(basename "$input_image")"
    output_image="${output_image%.*}_output.png"

    local output_text
    output_text="$(basename "$input_text")"
    output_text="${output_text%.*}_output.txt"

    mkdir -p "$output_path"

    echo "Running encoder.py script..."
    python src/encoder.py "$input_image" "$input_text" "$bit_plane" "$output_path/$output_image"

    echo "Running decoder.py script..."
    python src/decoder.py "$output_path/$output_image" "$bit_plane" "$output_path/$output_text"

    echo "Running bit_plane.py script..."
    python src/bit_plane.py "$input_image" "$output_path" "${bit_plane_to_view[@]}"
    python src/bit_plane.py "$output_path/$output_image" "$output_path" "${bit_plane_to_view[@]}"

    echo "Running message diff"
    diff -q "$input_text" "$output_path/$output_text" 
    echo "Scripts execution completed."
}

# Check if all required parameters are provided
if [ "$#" -lt 5 ]; then
echo "$#" < 5
    echo "Usage: $0 input_image input_text output_path max_bit_plane_to_encode bit_plane_list_to_view"
    exit 1
fi

# Run the function with provided arguments
run_scripts "$@"
