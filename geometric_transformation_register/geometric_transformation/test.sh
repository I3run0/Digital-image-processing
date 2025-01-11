#!/bin/bash

# Cria as pastas necessárias
mkdir -p input_image
mkdir -p output_images/scale/native
mkdir -p output_images/scale/opencv
mkdir -p output_images/rotate/native
mkdir -p output_images/rotate/opencv
mdkir -p csv_files/

BASE_URL="https://www.ic.unicamp.br/~helio/imagens_png/"

wget -r -np -nH --cut-dirs=3 -A jpg,jpeg,png -P input_images "${BASE_URL}"

scales=(1.5 2 0.5)
rotate=(90 -90 30 -30)
methods=("nearest" "bilinear" "bicubic" "lagrange")

results_file_scale="csv_files/comparison_results_scale.csv"
echo "Image,Scale,Method,MSE,NMSE" > $results_file_scale

results_file_rotate="csv_files/comparison_results_rotate.csv"
echo "Image,Rotation,Method,MSE,NMSE" > $results_file_rotate

IMAGES=(input_images/*)

for input_image in "${IMAGES[@]}"; do
  image_name=$(basename "$input_image")
  for method in "${methods[@]}"; do
    for scale in "${scales[@]}"; do
      native_output="output_images/scale/native/${image_name%.png}_${scale}_${method}.png"

      python3 src/geometric_transformation.py scale --scale-factor "$scale" --method "$method" "$input_image" "$native_output"
      opencv_output="output_images/scale/opencv/${image_name%.png}_${scale}_${method}.png"

      python3 src/opencv_geometric_transformation.py scale --scale-factor "$scale" --method "$method" "$input_image" "$opencv_output"

      comparison_results=$(python3 src/compare_images.py "$native_output" "$opencv_output")
      mse=$(echo "$comparison_results" | grep "MSE:" | awk '{print $2}')
      nmse=$(echo "$comparison_results" | grep "NMSE:" | awk '{print $2}')
      echo "$image_name,$scale,$method,$mse,$nmse" >> $results_file_scale
    done

    for rotation in "${rotate[@]}"; do
      # Executa o programa geometric_transformation.py
      native_output="output_images/rotate/native/${image_name%.png}_${rotation}_${method}.png"
      python3 src/geometric_transformation.py rotate --angle "$rotation" --method "$method" "$input_image" "$native_output"

      # Executa o programa opencv_geometric_transformation.p
      opencv_output="output_images/rotate/opencv/${image_name%.png}_${rotation}_${method}.png"
      python3 src/opencv_geometric_transformation.py rotate --angle "$rotation" --method "$method" "$input_image" "$opencv_output"
      # Compara as imagens e guarda os resultados no CSV
      comparison_results=$(python3 src/compare_images.py "$native_output" "$opencv_output")
      mse=$(echo "$comparison_results" | grep "MSE:" | awk '{print $2}')
      nmse=$(echo "$comparison_results" | grep "NMSE:" | awk '{print $2}')
      echo "$image_name,$rotation,$method,$mse,$nmse" >> $results_file_rotate
    done
  done
done

echo "Processamento concluído!"
