# Projeto de Transformações Geométricas em Imagens

Este projeto tem como objetivo aplicar transformações geométricas (escala e rotação) em imagens utilizando diferentes métodos de interpolação. Utiliza a biblioteca OpenCV para realizar essas operações e comparar os resultados com implementações próprias.

## Estrutura do Diretório

- `input_images/`: Contém as imagens de entrada utilizadas para as transformações.
- `output_images/`: Contém as imagens de saída geradas após o processamento.
  - `rotate/`: Imagens rotacionadas.
    - `native/`: Imagens geradas pelo script `geometric_transformation.py`.
    - `opencv/`: Imagens geradas pelo script `opencv_geometric_transformation.py`.
  - `scale/`: Imagens escalonadas.
    - `native/`: Imagens geradas pelo script `geometric_transformation.py`.
    - `opencv/`: Imagens geradas pelo script `opencv_geometric_transformation.py`.
- `src/`: Contém os scripts utilizados para o processamento das imagens.
  - `geometric_transformation.py`: Script que realiza transformações geométricas utilizando implementações próprias.
  - `opencv_geometric_transformation.py`: Script que realiza transformações geométricas utilizando a biblioteca OpenCV.
  - `compare_images.py`: Script que compara imagens geradas por diferentes métodos de interpolação e calcula métricas de similaridade.

## Dependências

- Python 3.x
- OpenCV
- NumPy

Para instalar as dependências, execute:
```bash
pip install opencv-python-headless numpy
```

## Como Utilizar

### Transformações Geométricas

1. Execute o script `geometric_transformation.py` ou `opencv_geometric_transformation.py` com os parâmetros desejados:

```bash
python src/geometric_transformation.py scale --scale-factor 1.5 --method bilinear input_images/image.jpg output_images/scale/native/image_scaled.jpg
```

ou

```bash
python src/geometric_transformation.py rotate --angle 1.5 --method bilinear input_images/image.jpg output_images/rotate/native/image_scaled.jpg
```

### Comparação de Imagens

Para comparar as imagens geradas pelos diferentes métodos de interpolação, execute o script `compare_images.py`:

```bash
python src/compare_images.py input_images/image.jpg output_images/scale/native/image_scaled.jpg
```

### Parâmetros e argumentos dos Scripts

#### geometric_transformation.py e opencv_geometric_transformation.py

- `scale`: Define que será feito uma transformação de escala.
- `rotate`: Define que será feito uma transformação de rotação;
- `--scale-factor`: Fator de escala (exemplo: 1.5).
- `--width-height`: Altura e largura para uma transformção de escala (exemplo: 400,600). Quando utilizada o fator de escala é ignorado.
- `--angle`: Ângulo de rotação (exemplo: 30).
- `--method`: Método de interpolação (opções: nearest, bilinear, bicubic, lagrange).
- `--help`: Exibe um mensagem de ajuda.
- `input_image`: Caminho para a imagem de entrada.
- `output_image`: Caminho para salvar a imgem de saída
#### compare_images.py

- `imageA`: Caminho da primeira imagem.
- `imageB`: Caminho da segunda imagem.

## Resultados

Os resultados das transformações geométricas são salvos na pasta `output_images/`, organizados em subdiretórios de acordo com o tipo de transformação (rotação ou escala) e a origem do método (nativo ou OpenCV).