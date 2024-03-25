#!/bin/bash

run_scripts() {
    local input_image="$1"
    local input_text="$2"
    local output_path="$3"
    local bit_plane="$4"

    local output_image
    output_image="$(basename "$input_image")"
    output_image="${output_image%.*}_output.png"

    local output_text
    output_text="$(basename "$input_text")"
    output_text="${output_text%.*}_output.txt"

    mkdir -p "$output_path"

    echo "Running encoder.py script..."
    python encoder.py "$input_image" "$input_text" "$bit_plane" "$output_path/$output_image"

    echo "Running decoder.py script..."
    python decoder.py "$output_path/$output_image" "$bit_plane" "$output_path/$output_text"

    list=()
    for ((i = 1; i <= $bit_plane; i++)); do
        list+=("$i")
    done

    echo "Running bit_plane.py script..."
    python bit_plane.py "$input_image" "$output_path" "${list[@]}"

    echo "Running message diff"
    diff -q "$input_text" "$output_path/$output_text" 
    echo "Scripts execution completed."
}

# Check if all required parameters are provided
if [ "$#" -ne 4 ]; then
    echo "Usage: $0 input_image input_text output_path bit_plane"
    exit 1
fi

# Run the function with provided arguments
run_scripts "$1" "$2" "$3" "$4"
