# `Geração Sintética de Dados Aplicado a Reconhecimento de Atividades Humanas (HAR)`
# `Synthetic Data Generation for Human Activity Recognition (HAR)`

## Apresentação

O presente projeto foi originado no contexto das atividades da disciplina de pós-graduação *IA376N - IA generativa: de modelos a aplicações multimodais*,
oferecida no segundo semestre de 2024, na Unicamp, sob supervisão da Profa. Dra. Paula Dornhofer Paro Costa, do Departamento de Engenharia de Computação e Automação (DCA) da Faculdade de Engenharia Elétrica e de Computação (FEEC).

|Nome  | RA | Especialização|
|--|--|--|
| Bruno Guedes da Silva  | 203657  | Eng. de Computação|
| Amparo Díaz  | 152301  | Aluna especial|

## Resumo (Abstract)



## Descrição do Problema/Motivação

O projeto tem como tema a geração de dados sintéticos de sensores para utilização em tarefas de reconhecimento de atividades humanas (HAR). Esse trabalho surge no contexto do Hub de Inteligência Artificial e Arquiteturas Cognitivas (HIAAC) do qual os integrantes do grupo fazem parte. Um dos objetos de estudos do HIAAC tem sido a tarefa de reconhecimento de atividades a partir de sensores de smartphones e foi observado a discordância entre diferentes datasets e metodologias da área. Assim, foi identificado uma oportunidade de avanço da área na criação de novos datasets e métodos de geração de dados sintéticos para aprimorar o desempenho de modelos para HAR.

Atualmente, há dois principais problemas relacionados aos dados existentes para o treinamento de modelos na tarefa HAR:

- **Falta de Dados:** A escassez de dados relevantes e diversos é um desafio significativo para o treinamento e avaliação de modelos de HAR. A coleta desse tipo de dados requer a participação de diversas pessoas em diferentes cenários e atividades. Embora a janela de tempo de cada captura de dados seja relativamente pequena (cerca de 1 a 15 minutos) o tempo de preparo do participante e deslocamento entre os locais em que as atividades são realizadas pode ser grande. Além disso, deve-se garantir que todos os sensores funcionem corretamente durante o experimento e que os dados sejam coretamente sincronizados e anonimizados. Diferentemente de dados como imagens, áudios e textos que são abundantemente presentes na internet, dados de sensores são mais escassos.
- **Heterogeneidade:** A variabilidade nas classes de atividade, na posição dos sensores e nas características das pessoas cria dificuldades para criar um dataset representativo e generalizável. A quantidade de atividades que uma pessoa pode realizar é imensa (subir escadas, pular, nadar, andar, correr) e pode ser modulada por diferentes fatores externos (clima, elevação, angulação do chão). Além disso, as características físicas do participante (altura, idade, peso, etc.) influenciam o comportamento dos dados. Esses fatores tornam difícil a construção de um dataset com classes bem definidas e variedade de participantes de forma a ser representativo o suficiente para generalização de modelos de aprendizado.

## Objetivo 

Diante do contexto e motivação apresentados, temos como objetivo geral a implementação e avaliação de modelos generativos para dados de sensores de acelerômetro e giroscópio correspondentes a diferentes atividades humanas, buscando obter dados sintéticos que sejam representativos da distribuição de dados reais e possam ser utilizados para a melhoria de modelos de classificação HAR.


## Metodologia

Neste trabalho, utilizamos os dados de 5 datasets (MotionSense[ref], KuHAR[ref], RealWorld[ref], UCI[ref] e WISDM[ref]) para reconhecimento de atividades humanas a partir de dados de acelerômetro e giroscópio. Ao invés dos dados brutos de cada dataset, são utilizados os dados processados do repositório DAGHAR[ref]. 

Três arquiteturas de geração de séries temporais são comparadas: DoppelGAN, TimeGAN e BioDiffusion. Esses modelos foram escolhidos por serem projetados para geração de séries temporais e implementam diferentes técnicas para alcançar isso.

### DoppelGAN

A DoppelGANger é um modelo generativo adversarial para séries temporais, o qual compila diversas técnicas da literatura para adereçar diferentes problemas que GANs sofrem durante o treinamento, especificamente de séries temporais. As principais técnicas aplicadas são:
Separação de informações categóricas (denominados metadados e que não se alteram durante o tempo para uma mesma amostra) de informações temporais;
Geração de metadados como condicionamento da geração de dados temporais;
Uso de uma heurística de auto-normalização, onde cada amostra é normalizada individualmente pelo seu máximo e mínimo e esses parâmetros são aprendidos como metadados da amostra. Para isso, um gerador de máximo e mínimo é utilizado especificamente para a geração desses metadados sintéticos;
Geração da série temporal em batches condicionada pelos metadados. A rede recorrente utilizada para geração da série temporal fornece, a cada iteração, $s$ instantes de tempo, ao invés de somente 1, como tradicionalmente se utiliza as redes recorrentes;

A implementação do modelo está disponível na biblioteca Gretel Synthetics.

A configuração utilizada nos experimentos é a padrão, com os seguintes hiperparâmetros:
| Parâmetro| Valor|
|---|---|
|Dimensão do vetor de ruído | 10|
|Número de camadas gerados de metadados | 3|
|Tamanho das camadas do gerador de metadados | 100|
|Número de camadas LSTM para o gerador de série temporal | 1|
|Tamanho das camadas LSTM | 100|
|Range da normalização das amostras |[0, 1]|
|Coeficiente de penalidade do gradiente | 10.0|
|Coeficiente de ponderação do erro do discriminador auxiliar | 1.0|
Learning rate (gerador, discriminador auxiliar e discriminador) | 0.001|
|Beta 1 para Adam (gerador, discriminador auxiliar e discriminador) | 0.5|
|Batch size | 32|

### BioDiffusion

O BioDiffusion é um modelo generativo de difusão para séries temporais multidimensionais de dados médicos. A arquitetura consiste em um modelo U-Net adaptado para séries temporais, onde camadas de convolução 1D são adicionadas no início e final do pipeline da rede. O modelo BioDiffusion pode ser utilizado com dados condicionados (presença de uma classe ou outro sinal ‘guia’) ou não-condicionados. 

A implementação utilizada é uma reprodução do código disponível no repositório do artigo [ref].

A configuração utilizada nos experimentos é a padrão, com os seguintes hiperparâmetros:
| Parâmetro| Valor|
|---|---|
|Erro | L2|
|Scheduler de ruído | cosine|
|Total de passo de difussão | 1000|
|Otimizador | AdamW|
|Learning Rate | 0.0003|
|Número de canais Conv1D | 32 |
|Multiplicadores de número de canais por bloco | [1, 2, 4, 8, 8]|
|Camadas ResNet por bloco | 3|

### TimeGAN

A proposta apresentada pelo modelo TimeGAN é a união dos métodos de treinamento de GANs e modelos autorregressivos para o aprendizado de um espaço latente representativo. Amostras reais são representadas em espaço latente por um embeder e dados sintéticos também são gerados diretamente na dimensão do espaço latente. O discriminador é treinado com base nas representações dos dados projetados no espaço latente e um reconstrutor é treinado para recuperar os dados na representação original a partir de sua projeção no espaço latente. Por fim, uma tarefa de supervisão é treinada conjuntamente, cujo objetivo é prever o próximo instante de tempo de um dado, real ou sintético, que foi projetado para o espaço latente.

### Bases de Dados e Evolução

|Base de Dados | Endereço na Web | Resumo descritivo|
|----- | ----- | -----|
|KuHAR | https://data.mendeley.com/datasets/45f952y38r/5 | Dados de 90 participantes capturados de smartphones em 18 atividades diferentes. |
| RealWorld |https://www.uni-mannheim.de/dws/research/projects/activity-recognition/dataset/dataset-realworld/ | Dados de 15 participantes capturados de sensores IMU em 8 atividades diferentes. |
| UCI | https://archive.ics.uci.edu/dataset/341/smartphone+based+recognition+of+human+activities+and+postural+transitions | Dados de 30 participantes capturados de smartphones em 6 atividades diferentes. |
| WISDM | https://archive.ics.uci.edu/dataset/507/wisdm+smartphone+and+smartwatch+activity+and+biometrics+dataset | Dados de 51 participantes capturados de smartphones e smartwatches em 18 atividades diferentes. |
| MotionSense | https://www.kaggle.com/datasets/malekzadeh/motionsense-dataset | Dados de 90 participantes capturados de smartphones em 18 atividades diferentes. |

Para este trabalho, inicialmente utilizaremos os conjuntos de dados fornecidos pela equipe da Meta 4 do grupo HIAAC, disponíveis em https://zenodo.org/records/11992126 . A versão balanceada destes conjuntos de dados foi denominada View Balanceada. Os dados foram balanceados de forma que todas as classes apresentassem o mesmo número de amostras, evitando que as diferentes proporções entre os rótulos afetassem a avaliação do desempenho. Essa abordagem visa garantir uma distribuição equitativa das classes, permitindo uma análise mais precisa das metodologias de avaliação implementadas para a geração de dados.
Adicionalmente, os subconjuntos de treino, teste e validação foram organizados de maneira que as amostras de um determinado participante não estivessem presentes em dois subconjuntos distintos. Essa estratégia é fundamental para evitar o vazamento de informações entre os conjuntos, assegurando que a validação do modelo seja realizada de forma justa e confiável.


### Workflow

#### Configuração do Ambiente

- Instalação de requisitos:
  - Python versão `3.9`
  - É recomendado o uso de um ambiente virtual do python (`venv` ou `conda`)
  - Instalação de dependências `pip install -r requirements.txt`
- Definir as configurações principais para realização do experimento em um arquivo YAML. Os arquivos de configuração são armazenados no diretório `/tests`. Um arquivo de configuração define:
  - Datasets a serem utilizados;
  - Transformações aplicadas sobre todos dados carregados;++
  - Modelos a serem treinados e seus parâmetros (a depender do modelo);++
  - Avaliações a serem realizadas;
  - Diretórios onde serão salvos os dados sintéticos e Arquivos de Report;
- Definição manual das seeds de geração aleatória dos diferentes frameworks utilizados (PyTorch, NumPy e Random).

#### Preparação dos Dados

- Download do repositório de dados padronizados do DAGHAR [ref], caso não exista no diretório `/data`;
- Leitura dos dados referente aos datasets selecionados no arquivo de configuração YAML;
  - Cada amostra consiste de 60 instantes de tempo dos sensores de acelerômetro (eixos x, y e z) e giroscópio (eixos x, y e z) e uma classe de atividade;
  - O conjunto de dados é organizado em uma tabela com uma amostra por linha e 361 dados por amostra (6 eixos x 60 instantes de tempo + 1 classe)
- Se definido no arquivo de configuração, é aplicada uma transformação nos dados (no momento, nenhum experimento é realizado com transformações de dados)

#### Treinamento de Modelos

- O modelo definido pelo arquivo de configuração é inicializado e treinado:
- DGAN [ref]
  - Os dados possuem dimensões (n_amostras, 361) e são organizados para um `np.array` de tamanho (n_amostras, 6, 60) contendo os dados das séries temporais e um `np.array` tamanho (n_amostras, ) com as classes das amostras;
  - A implementação da DGAN da biblioteca Gretel Synthetics [ref] é utilizada para treinamento do modelo;
- TimeGAN [ref]
  - Os dados possuem dimensões (n_amostras, 361) e são organizados para um `np.array` de tamanho (n_amostras, 6, 60) contendo os dados das séries temporais e um `np.array` tamanho (n_amostras, ) com as classes das amostras;
  - Os dados são separados por classe e é treinado um modelo TimeGAN incondicional por classe, ou seja, obtêm-se 6 diferentes modelos TimeGAN, cada um treinado para gerar dados de uma classe de atividade diferente;
  - A implementação do repositório XX foi reproduzida para o treinamento do modelo TimeGAN;
- BioDiffusion [https://www.mdpi.com/2306-5354/11/4/299]:
  - Os dados possuem dimensões (n_amostras, 361) e são organizados para um `np.array` de tamanho (n_amostras, 6, 60) contendo os dados das séries temporais e um `np.array` tamanho (n_amostras, ) com as classes das amostras;
  - Os dados são separados por classe e é treinado um modelo BioDiffusion incondicional por classe, ou seja, obtêm-se 6 diferentes modelos BioDiffusion, cada um treinado para gerar dados de uma classe de atividade diferente;
  - A implementação do repositório [https://github.com/imics-lab/biodiffusion/tree/main] foi reproduzido para o treinamento do modelo BioDiffusion e uma normalização por média e desvio padrão é aplicada nos dados de treino antes de serem passados para o modelo. Os parâmetros de normalização para cada classe são salvos para aplicação nos dados sintéticos gerados pelo modelo;

#### Avaliação de Modelos

- O modelo treinado é utilizado para gerar um conjunto de dados sintéticos;
- Os conjuntos de dados de treino, teste e sintéticos são passados para diferentes técnicas de avaliação selecionadas no arquivo de configuração;
- Estatísticas dos Dados:
  - Avaliação do conjunto de treino e/ou sintético;
  - Gera gráficos para:
    - Número de amostras por classe;
    - Projeção dos dados em 2 dimensões utilizando t-SNE;
    - Amostras aleatórias para cada classe separados por acelerômetro e giroscópio;
- Gráficos t-SNE:
  - Comparação das projeções dos conjuntos de treino e sintético:
    - Projeção conjunta de todos dados de treino e teste, identificando cada categoria. Previamente a projeção todos os dados são normalizados para média 0 e desvio padrão 1;
    - Projeção conjunta dos dados de treino e sintéticos separados por classe. Previamente a projeção todos os dados são normalizados para média 0 e desvio padrão 1;
- Avaliação de Utilidade:
  - Modelos de aprendizado de máquina para classificação são utilizados para comparar a utilidade dos dados sintéticos em auxiliar na melhoria da performance da classificação.
  - Para cada tipo de modelo (no momento são testados SVM e Random Forest), três classificadores são treinados e então avaliados no conjunto de teste:
    - Um classificador treinado com o conjunto de treino;
    - Um classificador treinado com o conjunto sintético;
    - Um classificador treinado com os conjuntos de treino e sintético;
  - Para cada classificador é reportados a acurácia, precisão, recall e pontuação f1;

### Ferramentas Utilizadas

|Ferramenta| Uso|
|---|---|
|Python| Linguagem de programação utilizada para implementação de todos os códigos|
|NumPy| Manipulação dos dados|
|PyTorch| Definição e treinamento do modelo BioDiffusion|
|Scikit Learn| Projeção de dados com t-SNE, treinamento de classificadores para métricas de utilidade e normalização de dados|
|Matplotlib| Geração de gráficos para comparação de amostras e histogramas|
|Gretel Synthetics| Implementação e treinamento do modelo DoppelGAN|
|Pandas| Manipulação, leitura e escrita de conjuntos de dados|
|Seaborn| Geração de gráficos para dados projetados por t-SNE|


## Experimentos, Resultados e Discussão dos Resultados



## Conclusão



## Referências Bibliográficas

HUANG, S.; CHEN, P.-Y.; MCCANN, J. DiffAR: Adaptive Conditional Diffusion Model for Temporal-augmented Human Activity Recognition. Proceedings of the Thirty-Second International Joint Conference on Artificial Intelligence. Anais... Em: THIRTY-SECOND INTERNATIONAL JOINT CONFERENCE ON ARTIFICIAL INTELLIGENCE {IJCAI-23}. Macau, SAR China: International Joint Conferences on Artificial Intelligence Organization, ago. 2023. Disponível em: <https://www.ijcai.org/proceedings/2023/424>

MALEKZADEH, M. et al. Protecting Sensory Data against Sensitive Inferences. Proceedings of the 1st Workshop on Privacy by Design in Distributed Systems. Anais... Em: EUROSYS ’18: THIRTEENTH EUROSYS CONFERENCE 2018. Porto Portugal: ACM, 23 abr. 2018. Disponível em: <https://dl.acm.org/doi/10.1145/3195258.3195260>.

NORGAARD, S. et al. Synthetic Sensor Data Generation for Health Applications: A Supervised Deep Learning Approach. 2018 40th Annual International Conference of the IEEE Engineering in Medicine and Biology Society (EMBC). Anais... Em: 2018 40TH ANNUAL INTERNATIONAL CONFERENCE OF THE IEEE ENGINEERING IN MEDICINE AND BIOLOGY SOCIETY (EMBC). Honolulu, HI: IEEE, jul. 2018. Disponível em: <https://ieeexplore.ieee.org/document/8512470/>.

RAVURU, C.; SAKHINANA, S. S.; RUNKANA, V. Agentic Retrieval-Augmented Generation for Time Series Analysis. arXiv, , 18 ago. 2024. Disponível em: <http://arxiv.org/abs/2408.14484>.

VAIZMAN, Y.; ELLIS, K.; LANCKRIET, G. Recognizing Detailed Human Context in the Wild from Smartphones and Smartwatches. IEEE Pervasive Computing, v. 16, n. 4, p. 62–74, out. 2017. 

ydataai/ydata-profiling. YData, , 9 set. 2024. Disponível em: <https://github.com/ydataai/ydata-profiling>.
