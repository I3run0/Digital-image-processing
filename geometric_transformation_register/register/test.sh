#!/bin/bash

mkdir -p input_images
mkdir -p output_images
mkdir -p output_images/ORB
mkdir -p output_images/SIFT
mkdir -p output_images/BRIEF

# URL to download images from
BASE_URL="https://www.ic.unicamp.br/~helio/imagens_registro/"

wget -r -np -nH --cut-dirs=3 -A jpg,jpeg,png -P input_images "${BASE_URL}"

if [ $? -ne 0 ]; then
    echo "Failed to download images from ${BASE_URL}"
    exit 1
fi

extract_prefix() {
    echo "$1" | grep -oE '^[^0-9]*[0-9]+'
}

IMAGES=(input_images/*)
methods=("ORB" "SIFT" "BRIEF")

declare -A PREFIX_GROUPS

for IMAGE in "${IMAGES[@]}"; do
    PREFIX=$(extract_prefix "$(basename "$IMAGE")")
    PREFIX_GROUPS["$PREFIX"]+="$IMAGE "
done

for PREFIX in "${!PREFIX_GROUPS[@]}"; do
    IMAGES_WITH_PREFIX=(${PREFIX_GROUPS[$PREFIX]})
    if [ ${#IMAGES_WITH_PREFIX[@]} -eq 2 ]; then
        for method in "${methods[@]}"; do
            IMAGE1="${IMAGES_WITH_PREFIX[0]}"
            IMAGE2="${IMAGES_WITH_PREFIX[1]}"
            OUTPUT_IMAGE="output_images/${method}/panorama_${PREFIX}.jpg"
            python3 "panoramic_join.py" --keypoint-detector ${method} "${IMAGE1},${IMAGE2}" "$OUTPUT_IMAGE"
        done
    fi
done
