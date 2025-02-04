# Descrição do Projeto

Este projeto consiste em um conjunto de scripts em Python e Bash para realizar a codificação e decodificação de mensagens em imagens utilizando a técnica de esteganografia. A esteganografia é a prática de esconder informações dentro de mídias digitais, como imagens, de forma que a presença da mensagem escondida não seja aparente.

## Informações pessoais

#### `Nome:` Bruno Sobreira França

#### `RA:` 217787

#### `Turma:` Mc920 - 2024S1

## Arquivos do Projeto

### `encoder.py`

Este script codifica uma mensagem de texto em uma imagem. Ele aceita os seguintes parâmetros:

- `input_image`: o caminho da imagem de entrada onde a mensagem será codificada.
- `input_text`: o caminho do arquivo de texto contendo a mensagem a ser codificada.
- `max_bit_plane`: o plano de bits maximo utilizado para codificação, deve ser um valor entre 0 e 2.
- `output_image`: o caminho onde a imagem codificada será salva.

### `decoder.py`

Este script decodifica uma mensagem previamente codificada em uma imagem. Ele aceita os seguintes parâmetros:

- `input_image`: o caminho da imagem codificada.
- `max_bit_plane`: o plano de bits maximo utilizado durante a codificação.
- `output_text`: o caminho onde a mensagem decodificada será salva.

### `bit_plane.py`

Este script realiza a fatiamento da imagem nos planos de bits especificados. Ele aceita os seguintes parâmetros:

- `input_image`: o caminho da imagem de entrada.
- `output_path`: o diretório onde as imagens resultantes serão salvas.
- `bit_planes`: os planos de bits a serem fatiados.

### `test.sh`

Este é um script em bash que automatiza a execução dos scripts Python. Ele aceita os seguintes parâmetros:

- `input_image`: o caminho da imagem de entrada.
- `input_text`: o caminho do arquivo de texto contendo a mensagem a ser codificada.
- `output_path`: o diretório onde os arquivos de saída serão salvos.
- `max_bit_plane`: o plano de bits máximo utilizado para codificação.
- `bit_planes`: os planos de bits a serem fatiados.

## Execução de testes via script test.sh

Para executar o projeto, você deve fornecer os parâmetros adequados ao script `test.sh`. Por exemplo:

```
./test.sh input_image.png input_text.txt output_directory 2 0 1 2 7
```

Este comando executará a codificação da mensagem contida em `input_text.txt` na imagem `input_image.png`, usando o plano de bits `2`, e salvará os arquivos resultantes no diretório `output_directory`. Os 4 últimos números (0, 1, 2, 7) referem-se aos planos de bits ,de cada banda de cor, que serão gerados.

Após a execução do script, os arquivos codificados e decodificados serão salvos no diretório especificado, e o script verificará se a mensagem original e a mensagem decodificada são iguais.

Dentro dos diretórios `inputs/texts/` e `inputs/images/` exite um conjunto de texto e imagens para testes.

Os seguintes restrições devem ser consideradas na execução do programa.

- `input_text`: é preciso utilizar um arquivo `.txt`, no qual todos caracters são ASCII.
- `input_image.png`: o formato de imagem suportado é apenas `.png`

## Agradecimentos 
Agradeço toda atenção!
:)