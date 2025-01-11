# Projeto de Registro de Imagens Panorâmicas

Este projeto tem como objetivo criar imagens panorâmicas a partir de duas ou mais imagens de entrada. Utiliza a biblioteca OpenCV para detectar pontos de interesse, corresponder descritores e alinhar imagens usando homografia.

## Estrutura do Diretório

- `input_images/`: Contém as imagens de entrada utilizadas para gerar as panorâmicas.
- `output_images/`: Contém as imagens de saída geradas após o processamento.
  - `SIFT/`: Imagens panorâmicas e correspondências geradas utilizando o método SIFT.
  - `BRIEF/`: Imagens panorâmicas e correspondências geradas utilizando o método BRIEF.
  - `ORB/`: Imagens panorâmicas e correspondências geradas utilizando o método ORB.
- `panoramic_join.py`: Script principal que realiza o processamento das imagens para gerar a panorâmica.

## Dependências

- Python 3.x
- OpenCV
- NumPy
- Matplotlib

Para instalar as dependências, execute:
```bash
pip install opencv-python-headless numpy matplotlib
```

## Como Utilizar

1. Execute o script `panoramic_join.py` com os parâmetros desejados:
```bash
python panoramic_join.py --keypoint-detector SIFT --show input_images/foto1A.jpg,input_images/foto1B.jpg output_images/panorama.jpg 
```

### Parâmetros do Script

- `input_images_path`: Caminhos das imagens de entrada, separados por vírgula. Exemplo: `foto1A.jpg,foto1B.jpg`.
- `output_image_path`: Caminho onde a imagem panorâmica resultante será salva. Exemplo: `panorama.jpg`.
- `--keypoint-detector`: Tipo de detector de pontos de interesse a ser utilizado. As opções disponíveis são SIFT, ORB e BRIEF, sendo ORB o valor padrão. Exemplo: `--keypoint-detector ORB`.
- `--show`: Se presente, indica que as imagens intermediárias e finais devem ser exibidas durante o processamento. Este parâmetro não requer um valor adicional.

## Resultados

Os resultados gerados pelo script `panoramic_join.py` são salvos na pasta `output_images/` organizada pelos métodos de detecção de pontos de interesse (SIFT, BRIEF, ORB). Cada subpasta contém as imagens panorâmicas e as correspondências de pontos entre as imagens de entrada.