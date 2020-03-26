# BlinkAnnotator


## Bibliotecas utilizadas
Para a leitura do teclado foi utilizado o **pynput**.
Para tratamento de vídeo e algoritmos de detecção foram utilizados o **OpenCV**, **Dlib** e **tensorflow+keras**.
O **OpenCV** é a biblioteca utilizada na captura de vídeo, detecção de face, manipulação e visualização de imagens.
O **Dlib** foi utilizado para a detecção dos *facial landmarks*.
O **tensorflow** e **keras** são permitem a utilização de *redes neurais*.

Para auxiliar a interação com o usuário, a biblioteca **tqdm** foi usada a fim de inicar o progresso do usuário no script.


## *Scripts* implementados
### Grupo A
- `face_recorder.py`: Permite que o usuário grave sua face e marque (utilizando o teclado) quando seus olhos fecharam
- `avg_annotator.py`: A partir das saídas de `face_recorder.py`, calcula a media dos pixeis na região dos olhos para análise
- `points_annotator.py`: A partir das saídas de `face_recorder.py`, detecta *facial landmarks* na região dos olhos para análise

### Grupo B
- `dataset_recorder.py`: Permite que o usuário tenha sua face gravada e emite duas vezes um sinal sonoro de dois segundos, indicando que o usuário deve fechar os olhos durante esse momento
- `pre_annotator.py`: A partir das saídas de `dataset_recorder.py`, utiliza uma CNN para preanotar se as frames estão com os olhos abertos ou fechados
- `blink_annotator.py`: A partir das saídas de `pre_annotator.py`, o usuário analisa as anotações e corrige aquelas erradas
- `visualizer.py`: Visualizador de anotações de `blink_annotator.py` ou `pre_annotator.py` em vídeo


## Como usar Face Recorder
### Instalação
Basta instalar a biblioteca do OpenCV e do pynpuy para python:
```
pip install opencv-python
pip install pynput
```
### Execução
O script para gravador da face deve ser executado utilizando os seguintes argumentos:
```
-rp --recordsPath : Caminho será salvo arquivo pickle com as informações de cada frame
-fd --framesDir : Caminho para diretório onde serão salvas as frames
```

Para executá-lo:
```
python face_recorder.py -rp RECORDS_PATH -fd FRAMES_DIR
```

Durante a execução, aparecerá a seguinte janela como *feedback*.
<br>
<img src="./assets/open.png" width="250" height="250">
<img src="./assets/closed.png" width="250" height="250">

O texto em verde indica por quanto tempo o programa está rodando. Ao atingir 60 segundos, o programa irá automaticamente terminar.

O texto em azul indica a situação dos olhos de acordo com o usuário. Segurar a tecla `n` indicará que o usuário está com os olhos fechados.

O texto em vermelho indica a quantas frames por segundo (`FPS`) o programa está operando.

O programa pode ser abortado teclando `q`. Isso fará com que o diretório criado com as frames já gravadas seja excluído.

## Como usar Points Annotator
TODO
