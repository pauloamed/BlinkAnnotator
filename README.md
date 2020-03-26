# BlinkAnnotator


## Bibliotecas utilizadas
Para a leitura do teclado foi utilizado o **pynput**.
Para tratamento de vídeo e algoritmos de detecção foram utilizados o **OpenCV**, **Dlib** e **tensorflow+keras**.
O **OpenCV** é a biblioteca utilizada na captura de vídeo, detecção de face, manipulação e visualização de imagens.
O **Dlib** foi utilizado para a detecção dos *facial landmarks*.
O **tensorflow** e **keras** são permitem a utilização de *redes neurais*.

Para auxiliar a interação com o usuário, a biblioteca **tqdm** foi usada a fim de inicar o progresso do usuário no script.


## *Scripts* implementados
### Experiments Scripts
- `face_recorder.py`: Permite que o usuário grave sua face e marque (utilizando o teclado) quando seus olhos fecharam
- `avg_annotator.py`: A partir das saídas de `face_recorder.py`, calcula a media dos pixeis na região dos olhos para análise
- `points_annotator.py`: A partir das saídas de `face_recorder.py`, detecta *facial landmarks* na região dos olhos para análise

### Dataset Scripts
- `dataset_recorder.py`: Permite que o usuário tenha sua face gravada e emite duas vezes um sinal sonoro de dois segundos, indicando que o usuário deve fechar os olhos durante esse momento
- `pre_annotator.py`: A partir das saídas de `dataset_recorder.py`, utiliza uma CNN para preanotar se as frames estão com os olhos abertos ou fechados
- `blink_annotator.py`: A partir das saídas de `pre_annotator.py`, o usuário analisa as anotações e corrige aquelas erradas
- `visualizer.py`: Visualizador de anotações de `blink_annotator.py` ou `pre_annotator.py` em vídeo



## Como usar Points Annotator
TODO
