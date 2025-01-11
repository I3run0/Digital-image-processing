# Alinhamento de documentos utilizando Projeção horizontal e Transformada de hough

## Informações pessoais

#### `Nome:` Bruno Sobreira França

#### `RA:` 217787

#### `Turma:` Mc920 - 2024S1

## Visão Geral

Este projeto consiste em scripts Python projetados para processamento de imagens, especificamente para rotação de imagens usando várias técnicas para alcançar o alinhamento ideal do texto. As funcionalidades principais incluem detectar o melhor ângulo de rotação para uma imagem, aplicar essa rotação e realizar OCR para comparar o texto antes e depois da rotação.

### Arquivos

1. `hough_transformation.py`
2. `horizontal_projection.py`
3. `alignmnet_utils.py`

## Requisitos

- Python 3.x
- OpenCV
- NumPy
- Matplotlib
- pytesseract
- difflib

Instale as bibliotecas necessárias usando pip:
```sh
pip install opencv-python numpy matplotlib pytesseract
```

## Descrição dos Scripts

### `hough_transformation.py`

Este script detecta linhas em uma imagem usando a Transformada de Hough e calcula o ângulo de rotação ideal com base na principal linha detectadas. Inclui opções para plotar os resultados, desenhar as linhas detectadas e realizar OCR para comparar o texto antes e depois da rotação.

#### Uso

```sh
python hough_transformation.py <caminho_imagem_entrada> <caminho_imagem_saida> [-p | --plot] [-d | --draw-lines] [-c | --ocr]
```

Opções:
- `-p, --plot`: Plota imagens originais e processadas.
- `-d, --draw-line`: Desenha a linha dominante detectada na imagem.
- `-c, --ocr`: Realiza OCR e compara o texto antes e depois da rotação.

### `horizontal_projection.py`

Este script calcula o ângulo de rotação ideal analisando a projeção horizontal da imagem e maximizando a variância das somas das linhas. Inclui opções para plotar os resultados, salvar histogramas das somas das linhas e realizar OCR.

#### Uso

```sh
python horizontal_projection.py <caminho_imagem_entrada> <caminho_imagem_saida> [-p | --plot] [-h | --histogram] [-c | --ocr]
```

Opções:
- `-p, --plot`: Plota imagens originais e processadas.
- `-h, --histogram`: Salva histogramas das somas das linhas.
- `-c, --ocr`: Realiza OCR e compara o texto antes e depois da rotação.

### `alignmnet_utils.py`

Este modulo utilitário fornece funções comuns usadas por `hough_transformation.py` e `horizontal_projection.py`. Inclui funções para rotacionar imagens, realizar comparações OCR e imprimir mensagens de saída.

## Exemplo

```sh
# Rotacionar uma imagem usando a Transformada de Hough
python hough_transformation.py --draw-lines --ocr input.jpg output.jpg

# Rotacionar uma imagem usando Projeção Horizontal
python horizontal_projection.py --histogram --ocr input.jpg output.jpg
```

## Autor

- [Seu Nome]

## Licença

Este projeto é licenciado sob a Licença MIT - veja o arquivo LICENSE para detalhes.