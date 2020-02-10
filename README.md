# BlinkAnnotator

Repositório contendo executáveis para anotação de piscadas. 

O gravador da face irá salvar fotos da face detectada durante 60 segundos e o usuário deve indicar se os olhos estão fechados ou não utilizando a tecla `n`.

O anotador irá ler o diretório com as frames salvas e o arquivo com as informações anotadas. Para cada frame, os *landmarks* dos olhos serão extraídos, relacionados com as informações referentes àquela imagem e então salvos num arquivo `.csv`.

<img src="https://www.pyimagesearch.com/wp-content/uploads/2017/04/facial_landmarks_68markup-768x619.jpg" width="250" height="250">

## Bibliotecas utilizadas
Foram utilizados o **OpenCV** e o **Dlib**.
O **OpenCV** é a biblioteca utilizada na captura de vídeo, detecção de face, manipulação e visualização de imagens.
O **Dlib** foi utilizado para a detecção dos *facial landmarks*.

## Como usar Face Recorder
### Instalação
Basta instalar a biblioteca do OpenCV para python: 
```
pip install opencv-python
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

O texto em azul indica a situação dos olhos de acordo com o usuário. Para mudá-la, aperte a tecla `n`. `open` indica que os olhos estão abertos e `closed` que os olhos estão fechados. (se você conseguir ler `closed` algo está errado :P)

O texto em vermelho indica a quantas frames por segundo (`FPS`) o programa está operando. 

O programa pode ser abortado teclando `q`. Isso fará com que o diretório criado com as frames já gravadas seja excluído.

## Como usar Points Annotator
TODO
