# Dataset Scripts

## Como usar `blink_annotator.py`
### Uso do *script*
O *script* é dividido em três fases:
- identificação de frames mal preanotadas
- correção de frames identificadas
- identificação e correção de sequências muito curtas

#### Primeira fase
Na primeira fase, o usuário irá indicar as frames que não foram classificadas corretamente. Para isso, serão criados dois grupos: os grupos de frames classificadas como OLHOS_ABERTOS e o outro grupo como OLHOS_FECHADOS.

Todas as imagens de cada grupo serão passadas em sequência e o usuário deverá indicar quando uma imagem não pertence ao determinado grupo. Será utilizada a barra de espaço para isso. Como podem haver atrasos por parte do usuário e por parte do computador, será considerada a frame atual e uma janela contendo as X últimas frames. O valor de X está em função da velocidade que as frames irão passar, essa regulada pelo usuário por argumento de entrada (`timesSlower`).

Note que uma marcação indica a frame atual e uma janela das frames passadas, mas não uma janela das frames futuras. Logo, sempre que for visto uma imagem fora da classe, a barra de espaço deve ser apertada, mesmo que essa já tenha sido poucos instantes antes.

Assim que o usuário executar o programa, a primeira fase começa e o seguinte texto é impresso:
<br><img src="./assets/first_phase.png" width="500" height="250">

Nele, é indicado que a primeira fase está começando, e também são indicados o tempo que cada frame aparecerá (em segundos) e o tamanho da janela de frames passadas que será considerada a fim de tratar o atraso. Ambos valores, como já dito, estão em função do parâmetro de entrada `timesSlower`.

O primeiro grupo que será mostrado será o de **OLHOS_FECHADOS**. Logo, o usuário deve **marcar com barra de espaço todas as frames que encontrar algum olho aberto**.

<br><img src="./assets/first_phase_a.png" width="500" height="250">

Nessa fase, o usuário deve apertar espaço em casos como o seguinte (já que esses casos **não** estão em sua classe correta)
<br>s<img src="./assets/first_phase_a_open.png" width="500" height="250">

E não apertar em casos como o seguinte (já que esses casos estão em sua classe correta)
<br><img src="./assets/first_phase_a_closed.png" width="500" height="250">

O segundo grupo que será mostrado será o de **OLHOS_ABERTOS**. Logo, o usuário deve **marcar com barra de espaço todas as frames que encontrar algum olho fechado**.

<br><img src="./assets/first_phase_b.png" width="500" height="250">

E não apertar em casos como o seguinte (já que esses casos estão em sua classe correta)
<br><img src="./assets/first_phase_b_open.png" width="500" height="250">


O fim dessa fase é indicado com a quantidade de frames cujas classes o usuário indicou como erradas.

#### Segunda fase
Na segunda fase, o usuário irá anotar/corrigir as frames que marcou como erradas. As frames irão aparecer na tela e esperar que o usuário indique se pertencem à classe de olhos abertos (`O`, para *open*) ou fechados (`C`, para *closed*).

<br><img src="./assets/second_phase.png" width="500" height="250">
<br><img src="./assets/second_phase_1.png" width="500" height="250">

O usuário deve digitar `C` ou `O`. A digitação errada será indicada com a mensagem `Wrong input!`.

<br><img src="./assets/second_phase_input.png" width="400" height="100">
<br><img src="./assets/second_phase_wronginput.png" width="500" height="250">
<br><img src="./assets/second_phase_example.png" width="500" height="250">

É necessária uma discussão do time de anotação para essa fase, dado que aparecerão casos não tão simples de serem anotados, como o seguinte

<br><img src="./assets/second_phase_problem.png" width="500" height="250">

#### Terceira fase
A terceira fase irá indicar e corrigir as sequências muito curtas. O que foi classificado até agora será analisado e as sequências de tamanho abaixo de um limite indicado pelo usuário (argumento de entrada `seqThreshold`) serão salvas numa lista. As sequências adjacentes serão unidas numa só lista.

<br><img src="./assets/third_phase.png" width="500" height="250">

Após a extração dessas sequências, essas serão postas novamente para correção. Cada sequência será reproduzida para o usuário, para que esse possa analisá-la melhor. Se for do interesse do usuário, esse pode reproduzir novamente a sequência.

<br><img src="./assets/third_phase_seq.png" width="500" height="250">

Então, a sequência será posta para anotação

<br><img src="./assets/third_phase_ann.png" width="500" height="250">

### Como instalar prerequisitos
Basta instalar as seguintes bibliotecas:
```
pip install opencv-python
pip install pynput
pip install tqdm
```

### Como executar
O script deve ser executado utilizando os seguintes argumentos:
```
-ip --inputPath: caminho para arquivo .csv com as preanotações
-fd --framesDir: caminho para diretório com frames anotadas por INPUT_PATH
-op --outputPath: caminho onde o arquivo .csv com as anotações corrigidas deve ser salvo
-ts --timesSlower: quantas vezes mais lento que a reprodução normal serão passadas as frames na primeira fase
-st --seqThreshold: limite de tamanho de sequência para a terceira fase

timesSlower é um argumento facultativo. Se não indicado, será assumido o valor 3.
seqThreshold é um argumento facultativo. Se não indicado, será assumido o valor 3.
```

Para executá-lo:
```
python face_recorder.py -ip inputPath -fd framesDir -op outputPath -ts timesSlower -st seqThreshold
```
Os argumentos facultativos não precisam ser indicados
```
python face_recorder.py -ip inputPath -fd framesDir -op outputPath
```
