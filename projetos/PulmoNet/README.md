# `PulmoNet: Rede Neuronal Generativa para Imagens Tomográficas Pulmonares`
# `PulmoNet: Generative Neuronal Network for Pulmonary Tomographic Images`

## Apresentação

O presente projeto foi originado no contexto das atividades da disciplina de pós-graduação *IA376N - IA generativa: de modelos a aplicações multimodais*, 
oferecida no segundo semestre de 2024, na Unicamp, sob supervisão da Profa. Dra. Paula Dornhofer Paro Costa, do Departamento de Engenharia de Computação e Automação (DCA) da Faculdade de Engenharia Elétrica e de Computação (FEEC).

 |Nome  | RA | Especialização|
 |--|--|--|
 | Arthur Matheus Do Nascimento | 290906 | Eng. Elétrica |
 | Júlia Castro de Paula | 219193 | Eng. Elétrica |
 | Letícia Levin Diniz | 201438  | Eng. Elétrica |

## Tabela de Conteúdos

1. [Resumo](#resumo-abstract)
2. [Links Importantes](#links-importantes)
3. [Descrição do Problema / Motivação](#descrição-do-problemamotivação)
4. [Objetivo](#objetivo)
5. [Metodologia](#metodologia)
    1. [Materiais de Referência](#materiais-de-referência)
    2. [Modelo Proposto](#modelo-proposto)
    3. [Bases de Dados e Evolução](#bases-de-dados-e-evolução)
    4. [Workflow](#workflow)
    5. [Ferramentas Relevantes](#ferramentas-relevantes)
    6. [Métricas de Avaliação](#métricas-de-avaliação)
        1. [Análise Qualitativa](#análise-qualitativa)
        2. [Análise Quantitativa](#análise-quantitativa)
        3. [Análise de Utilidade](#análise-de-utilidade)
    7. [Cronograma](#cronograma)
    8. [Ambiente Computacional](#ambiente-computacional)
6. [Experimentos, Resultados e Discussão dos Resultados](#experimentos-resultados-e-discussão-dos-resultados)
    1. [Resultados preliminares com 10 mil dados de treinamento da GAN](#resultados-preliminares-com-10-mil-dados-de-treinamento-da-gan)
    2. [Resultados com 60 mil dados de treinamento da GAN](#resultados-com-60-mil-dados-de-treinamento-da-gan)
7. [Conclusão](#conclusão)
    1. [Perspectivas Futuras](#perspectivas-futuras)
8. [Referências Bibliográficas](#referências-bibliográficas)

**ANEXOS**:
1. [Explorações de arquiteturas e dados](#explorações-de-arquiteturas-e-dados)
2. [Como rodar os modelos](HowToRun.md?raw=true).

## Links Importantes
Links para apresentações de slides e vídeos para entregas E1, E2 e E3 para a disciplina:

[Link para o vídeo de apresentação E1](https://drive.google.com/file/d/1TlpQOlCh_lAI0-jPPMPWOzGZ_werCo3d/view?usp=sharing)

[Link para a apresentação de slides E1](https://docs.google.com/presentation/d/1b8W0Cw1eiTbWlJ0CJJ8eMRA4zyu2iLhYvggi55-mOb0/edit?usp=sharing)

[Link para a apresentação de slides E2](https://docs.google.com/presentation/d/1QH5_WpeTp7kQPSVB78ukK7msn-Tx09pZoM_3dWmeqC4/edit?usp=sharing)

[Link para a apresentação de slides E3](https://docs.google.com/presentation/d/1YcYpWPjaEHAoT9k7YVTDgA9t4VoGU5SL2fJ26_SG3ok/edit?usp=sharing)

## Resumo (Abstract)

As tomografias computadorizadas (CT) pulmonares e a segmentação das vias aéreas são essenciais para o diagnóstico preciso de doenças pulmonares, e para auxiliar nesse procedimento, propõe-se a PulmoNet, uma rede generativa que produz imagens artificiais 2D de CTs pulmonares para auxiliar a ampliar bancos de dados existentes, possibilitando melhorar a qualidade de redes neurais de segmentação das vias aéreas. Utilizando a base de dados ATM'22, implementa-se uma arquitetura GAN com gerador Pix2Pix e discriminador PatchGAN, que sintetiza novas amostras de tomografia a partir de segmentações pulmonares, com foco na amplificação de informações das vias aéreas. Avalia-se a rede qualitativamente, quantitativamente (métricas FID e SSIM) e em utilidade. Os resultados finais demonstraram uma evolução na qualidade das imagens ao ampliar a quantidade de imagens utilizadas e otimizar os parâmetros de treino, mas demonstraram dificuldade em gerar conteúdo das vias aéreas em si. A FID obtidas nos três melhores modelos variou entre 290 e 310, enquanto o SSIM ficou entre 0.61 e 0.77, dependendo se o enfoque era nas vias aéreas ou na imagem como um todo. Também foram realizados testes para analisar a possibilidade de utilizar os pesos da GAN para treinar redes de segmentação, mas sem resultados relevantes alcançados.

## Descrição do Problema/Motivação
As tomografias computadorizadas (CT) pulmonares, juntamente com a segmentação das vias aéreas, desempenham um papel crucial no diagnóstico preciso de doenças pulmonares. Ao gerar imagens detalhadas da região torácica, a tomografia permite que médicos mapeiem a anatomia das vias aéreas antes de procedimentos cirúrgicos, avaliando a extensão de lesões e facilitando o acompanhamento da progressão de doenças respiratórias [[2]](#2). Além disso, a CT é fundamental para monitorar a eficácia de tratamentos e detectar seus possíveis efeitos colaterais [[5]](#5).

A complexidade e diversidade do corpo humano, bem como o custo e acesso a CT, limitam a obtenção de grandes volumes de dados que sejam representativos das diversas condições anatômicas. Essa escassez de dados limita a performance de modelos de aprendizado de máquina, como redes neurais, que utilizam de tais dados para promover ferramentas que auxiliem a equipe médica. A limitação de tais modelos pode levar a diagnósticos imprecisos, e comprometer a qualidade do atendimento a pacientes [[6]](#6). Com as redes generativas é possível criar dados de forma a compensar essa escassez, potencialmente aprimorando a performance de modelos treinados com o suporte desses dados sintéticos.

## Objetivo
Este projeto visa gerar imagens sintéticas de tomografia computadorizada (CT) da região torácica. A priori, o modelo generativo proposto terá como saída imagens em duas dimensões (2D) de CT da região do tórax. Busca-se um grau de realismo suficiente para auxiliar redes de segmentação de vias aéreas. 
Além disso, este trabalho também serve como uma primeira etapa de um projeto maior e mais ambicioso, no qual buscar-se-á a geração de volumes (3D) de tomografias pulmonares, uma combinação de imagens que juntas formam o equivalente a um exame real.

## Metodologia
### Materiais de Referência
O trabalho desenvolvido em [[1]](#1), propõe duas arquiteturas baseadas em GANs para a síntese de imagens CT pulmonares a partir de máscaras binárias que segmentam a região pulmonar. No artigo em questão, as imagens sintéticas se limitam a região pulmonar, sem produzir elementos ao seu entorno, como os músculos torácicos e a coluna vertebral. Em [[3]](#3), desenvolve-se uma GAN condicional para a geração de imagens CT pulmonares a partir de imagens de ressonância magnética. Já o trabalho em [[4]](#4) utiliza um modelo baseado em GAN para a segmentação do pulmão em imagens CT que contém anomalias no tecido pulmonar.

### Modelo Proposto
Trabalhos correlatos ao nosso projeto indicam que a estratégia predominante para a síntese de CTs pulmonares e conversão imagem para imagem corresponde a aplicação de GANs (redes adversárias generativas). A estrutura de uma GAN é composta por uma rede neural "geradora", responsável por sintetizar as distribuições de entrada e retornar saídas similares aos dados reais, e uma rede neural "discriminadora", que deve ser capaz de classificar corretamente suas entradas como "reais" ou "falsas". Com isso, uma boa rede generativa deve ser capaz de enganar o discriminador, ao passo que um bom discriminador deve identificar corretamente os dados sintéticos em meio aos dados reais [[11]](#11). Idealmente, o gerador e o discriminador jogam um jogo, no qual o primeiro minimiza um critério, enquanto o segundo o maximiza. Com o treinamento, espera-se obter um "equilíbrio de Nash", onde cada estrutura tem 50% de chance de ganhar. 

Este projeto se inspira no trabalho desenvolvido em [[1]](#1). Das duas arquiteturas propostas no artigo de referência, inspira-se na arquitetura Pix2Pix, na qual o gerador é composto de um *encoder* que aumenta a profundidade da imagem enquanto diminui suas dimensões, seguido de um *decoder* que realiza o processo oposto. Tal arquitetura também utiliza conexões residuais (*skip connections*), que concatenam camadas da rede codificadora com a decodificadora (Fig. 1). Além disso, na arquitetura proposta, o discriminador segue a arquitetura 30 × 30 PatchGAN, sendo composto por cinco camadas convolucionais, onde as quatro primeiras são seguidas por uma camada de ativação *LeakyReLu*, enquanto a última é seguida de uma função *sigmoide* (Fig. 2).
Nesta configuração, cada pixel da imagem de saída do discriminador - uma imagem 1 x 62 x 62 - está associado a um patch de pixels da imagem de entrada, assumindo um valor binário (0 ou 1) correspondente à classificação em "real" ou "falso" do patch associado.

Em [[1]](#1), a entrada do gerador corresponde a uma máscara binária com o formato de um pulmão, e, sua saída corresponde a uma imagem onde o pulmão está preenchido conforme seria em uma CT. Neste trabalho, a entrada do gerador é a mesma da proposta pela referência, no entanto, ao invés de simplesmente obter o preenchimento do pulmão na saída, deseja-se que a saída contenha tanto o interior do pulmão quanto os elementos ao seu entorno, i.e., uma imagem de saída equivalente a que se tem em uma CT real. Ainda em [[1]](#1), a rede considerada é uma *conditional GAN*, na qual o discriminador recebe tanto a imagem CT (real ou sintética) quanto a máscara binária de segmentação do pulmão. Ambas estruturas foram inicialmente recomendadas por [[8]](#8).
As duas imagens abaixo ilustram as arquiteturas do gerador e discriminador, respectivamente.

![Arquitetura Pix2Pix proposta para gerador.](figs/arquitetura_gen.png?raw=true)

*Figura 1: Arquitetura Pix2Pix proposta para gerador.*

![Arquitetura PatchGAN proposta para discriminador.](figs/arquitetura_disc.png?raw=true)

*Figura 2: Arquitetura PatchGAN proposta para discriminador.*

Em [[1]](#1), a função de *loss* aplica um critério similar à *Binary Cross Entropy*, com regularização por MAE (*Mean Absolute Error*), conforme a seguinte a equação matemática:

$$arg\ min_{𝐺}\ max_{𝐷}\ E_{𝑥,𝑦}[log 𝐷(𝑥, 𝑦)] + E_{𝑥,𝑧}[log(1 − 𝐷(𝑥, 𝐺(𝑥, 𝑧)))] + 𝜆E_{𝑥,𝑦,𝑧}[‖𝑦 − 𝐺(𝑥, 𝑧)‖_{1}],$$

onde $x$ corresponde a máscara pulmonar, $z$ corresponde ao ruído (aplicado a $x$ ou imposto pelo gerador $G$ pelo uso de *dropout*) [[8]](#8), e $y$, a imagem CT real. Nota-se que a regularização se aplica apenas ao gerador. 

No trabalho em questão, considera-se variações da *loss* apresentada acima. Tais variações incluem: regularização por MSE (*Mean Squared Error*) ao invés de MAE, regularização apenas na região da máscara que representa o interior do pulmão e regularização apenas na região da máscara que representa o exterior do pulmão. As variações da *loss* foram testadas durante o processo de busca pelos hiperparâmetros da rede, como será abordado na seção [Workflow](#workflow). Idealmente, neste trabalho, busca-se uma *loss* que permita a síntese de imagens onde tanto as estruturas externas ao pulmão (músculos torácicos e a coluna vertebral), como as internas ao mesmo (vias aéreas) sejam bem representadas.

Ao modelo proposto e desenvolvido, da-se o nome de PulmoNet.

### Bases de Dados e Evolução
Apesar da metodologia ser baseada no artigo [[1]](#1), o desenvolvimento deste projeto utiliza a base de dados fornecida pelo Desafio ATM'22 (Tab. 1)[[2]](#2). 
Essa alteração se deve a necessidade de utilizar um banco de dados contendo informações sobre as vias aéreas e não somente sobre o pulmão, possibilitando melhores comparações e avaliações sobre as aplicações propostas.
O banco de dados utilizado é composto por 500 volumes/exames de CT da região torácica juntamente com máscaras de segmentação das vias aéreas, tendo essas sido feitas e validadas por três radiologistas experientes. Cada volume é simplesmente um empilhamento de múltiplas fatias/imagens 2D do tórax.
Dos 500 volumes, 350 foram obtidos do Hospital de Peito de Xangai, com aprovação ética de número KS(Y)21328. Os 150 volumes restantes foram obtidos de desafios anteriores ou de bancos de dados públicos.
Sua divisão foi feita destinando 300 volumes para treino, 50 para validação e 150 para teste.

*Tabela 1: Descrição e acesso a base de dados ATM'22.*
|Base de Dados | Endereço na Web | Resumo descritivo|
|----- | ----- | -----|
|ATM'22 | https://atm22.grand-challenge.org/  | Esta base contém 500 volumes CTs pulmonares, nos quais as vias aéreas foram completamente anotadas, i.e., delimitadas, por especialistas. Esta base de dados foi utilizada para um desafio de segmentação automática de vias aéreas em volumes de CT da região pulmonar [[2]](#2).|

Os dados desta base são arquivos com extensão `*.nii.gz`, um formato comum para imagens médicas por possibilitar armazenar outras informações relevantes para estudos clínicos, e contêm todo o volume pulmonar obtido durante um exame de tomografia. Os arquivos são fornecidos em pares, com os volumes de CT sendo acompanhados pelas anotações das vias aéreas feitas por especialistas da totalidade do volume. Tais dados podem ser lidos com auxílio da biblioteca `SimpleITK`, conforme feito pelas classes em `datasets.py` neste repositório.

Dado que este trabalho centra-se na geração de imagens sintéticas 2D de CTs pulmonares, as fatias de cada um dos volumes pulmonares foram separadas, assim como ilustrado na imagem abaixo, permitindo utilizá-las de forma independente, resultando em imagens 2D de 512x512, aumentando o tamanho dos conjuntos de dados disponíveis para treinamento, validação e testes.

![Exemplo de fatia de CT pulmonar obtida a partir da base de dados ATM'22.](figs/dataset_exemplo_fatia.png?raw=true)

*Figura 3: Exemplo de fatia de CT pulmonar obtida a partir da base de dados ATM'22.*

Como a entrada da rede geradora são máscaras pulmonares, apenas fatias contendo uma quantidade significativa de pulmão foram selecionadas para o desenvolvimento deste projeto. Para fazer essa seleção, utiliza-se a biblioteca em Python `lungmask` [[7]](#7), que realiza a  segmentação automática dos CTs pulmonares. Considerando a distribuição da quantidade de imagens em função da quantidade de pixels presentes na região pulmonar (Fig. 4) e a área da imagem ocupada pelo pulmão em função da quantidade de pixels segmentados (Fig. 5), definiu-se de modo empirico um limite inferior de 25mil pixels para a região pulmonar. Imagens cujas máscaras correspondentes continham menos pixels do que o limite estabelecido foram descartadas, resultando em 90 mil imagens disponíveis para o desenvolvimento deste projeto. 

![Histrograma da quantidade de pixels das fatias selecionadas após segmentação das CTS pulmonares da base de dados ATM'22.](figs/histograma_fatias.png?raw=true)

*Figura 4: Histrograma da quantidade de pixels das fatias selcionadas após segmentação das CTS pulmonares da base de dados ATM'22.*

![Exemplos de fatias das CTS pulmonares da base de dados ATM'22 segmentadas.](figs/exemplos_pixels.png?raw=true)

*Figura 5: Exemplos de fatias das CTS pulmonares da base de dados ATM'22 segmentadas.*

Essas 90 mil imagens foram dividas em conjuntos de treinamento, validação e testes. Para permitir uma comparação justa com os resultados quantitativos obtidos em [[1]](#1) (FID e SSIM), opta-se por utilizar a mesma quantidade de dados de teste que o artigo de referência: 7 mil imagens. Além disso, deseja-se realizar um teste de utilidade do modelo, o qual envolve o treinamento e avaliação de uma rede de segmentação (detalhes em [Métricas de Avaliação](#métricas-de-avaliação)). Desse modo, define-se 14 mil dados para o treino da rede de segmentação, e 2 mil dados para sua validação. Considerando que o conjunto de testes da GAN para as outras métricas não tem relação com os dados da rede de segmentação, reaproveita-se este conjunto para obtenção das métricas do teste de utilidade (DICE). Assim, o conjunto de teste da rede de segmentação é o mesmo proposto para a obtenção das métricas quantitativas da GAN. O restante dos dados são divididos em 60 mil para treino da GAN e 7 mil para validação da mesma. Com isso, em uma visão geral, separa-se dois terços (cerca de 66,7%) da base de dados completa para o treinamento da GAN, 7,8% para a validação da GAN e 25,6% para todos os testes (incluindo a análise qualitativa, análise quantitativa e o teste de utilidade) (Fig. 6). Nota-se que, devido ao reaproveitamento do conjunto de testes entre a GAN e a rede de segmentação, tem-se ao final cinco (5) conjuntos na saída desta etapa (Fig. 7).

![Separação da base de dados completa em conjuntos de treinamento, validação e testes. Visão geral desta separação, em porcentagem.](figs/Dados_porcentagem.png?raw=true)

*Figura 6: Separação da base de dados completa em conjuntos de treinamento, validação e testes. Visão geral desta separação, em pocentagem.*

![Separação da base de dados completa em conjuntos de treinamento, validação e testes. Visão geral desta separação.](figs/Dados_visao_geral.png?raw=true)

*Figura 7: Separação da base de dados completa em conjuntos de treinamento, validação e testes. Visão geral desta separação.*

Desconsiderando os testes de utilidade e focando apenas nos testes qualitativos e quantitativos, tem-se 60 mil dados para treinamento da GAN, 7 mil para validação e 7 mil para testes. Isso representa um proporção próxima a 80-10-10, uma das mais clássicas na literatura para treinamento de redes neurais (Fig. 8). 

![Separação da base de dados em conjuntos de treinamento, validação e testes, considerando apenas os testes qualitativos e quantitativos para avaliação da GAN.](figs/Dados_GAN.png?raw=true)

*Figura 8: Separação da base de dados em conjuntos de treinamento, validação e testes, considerando apenas os testes qualitativos e quantitativos para avaliação da GAN.*

Por sua vez, para a rede de segmentação, tem-se uma proporção de conjuntos de treino-validação-teste de 60-10-30, o que também é bem comum na literatura e é a proporção utilizada no desafio de segmentação da base ATM'22 [[2]](#2) (Fig. 9).

![Separação da base de dados em conjuntos de treinamento, validação e testes, considerando apenas o teste de utilidade (rede neural para segmentação das vias aéreas pulmonares).](figs/Dados_seg.png?raw=true)

*Figura 9: Separação da base de dados em conjuntos de treinamento, validação e testes, considerando apenas o teste de utilidade (rede neural para segmentação das vias aéreas pulmonares).*

Por fim, ressalta-se que além da segmentação dos dados e seleção das fatias, a base de dados também passa pelas etapas de normalização e de transformação para `numpy arrays`, antes de ser utilizada pelas GANs implementadas neste projeto (Fig. 10).

![Fluxograma para processamento da base de dados.](figs/Fluxo_proc_dados.png?raw=true)

*Figura 10: Fluxograma para processamento da base de dados.*

### Workflow
Em uma perspectiva geral do projeto, a metodologia se divide em três grandes estágios:
1. Preparação da base de dados;
2. Treinamento e fine-tunning de modelos de síntese;
3. Avaliação dos modelos gerados.

No que diz respeito à preparação da base de dados, aplica-se o fluxo descrito na Figura 10, da seção anterior, na qual os dados são obtidos de uma fonte pública, processados e separados em conjuntos de treinamento, validação e testes. A saída desta etapa são 90 mil trios (fatia da CT pulmonar, segmentação feita por especialistas e máscara binária da região do pulmão), com dimensão 1 x 512 x 512 cada.

Quanto a segunda etapa, implementa-se a arquitetura de uma GAN, descrita na seção [Modelo Proposto](#modelo-proposto), que foi concebida tomando como base o artigo [[1]](#1). Para o treinamento da GAN, existem diversos hiper-parâmetros que impactam na performance da rede. Um dos maiores problemas das GANs é a instabilidade do treinamento das mesmas, o que releva uma sensibilidade aos hiper-parâmetros de treinamento. Para guiar o desenvolvimento deste projeto, alguns destes hiper-parâmetros foram mantidos fixos (otimizador, tamanho dos batches), enquanto outros foram variados segundo a Tabela 2, de forma a encontrar a melhor combinação para gerar imagens sintéticas de CTs pulmonares mais realistas.

*Tabela 2: Identificação e variação de hiper-parâmetros de treino.*
|Parâmetros | Possibilidades |
|----- | ----- |
|Passos de atualização do discriminador | 1 a 4 |
|Passos de atualização do gerador | 1 a 4 |
|Tipo de ruído | [Uniforme, Gaussiano] |
|Localização do ruído | Na imagem completa ou apenas na região do pulmão|
|Intensidade do ruído | 0.3 a 1 |
|Média para ruído Gaussiano | 0.5 a 1 |
|Desvio-padrão para ruído Gaussiano | 0.1 a 0.5 |
|Loss | BCE ou MSE |
|Regularizador | MAE ou MSE |
|Nível de regularização | 1 a 15 |
|Região de regularização | Imagem completa, dentro do pulmão ou fora do pulmão |
|Learning Rate do otimizador | $1 \times 10^{-4}$ a $4 \times 10^{-4}$ |
|Parâmetro beta do otimizador |0.4 a 0.9 |

Um ponto importante a ser destacado com relação a esta varredura é a diferença do tipo e nível de ruído aplicado na máscara de entrada do gerador. Como é possível observar na tabela acima, duas distribuições foram testadas: uniforme e Gaussiana. Mais ainda, o nível e a localização do ruído também foram variados. Tais mudanças impactam na entrada recebida pelo gerador e, portanto, podem interferir no desempenho e qualidade do processo de síntese (Fig. 11).

![Exemplos de entradas com diferentes tipos e níveis de ruídos.](figs/imagem_ruidos.png?raw=true)

*Figura 11: Exemplos de entradas com diferentes tipos e níveis de ruídos.*

Outro ponto interessante que merece ser mencionado é a variação dos passos de atualização do gerador e do discriminador.
Um problema típico de treinamento de GANs é a velocidade de aprendizado em diferentes ritmos do gerador e do discriminador, isto é, uma destas redes pode aprender mais rápido do que a outra, resultando em uma baixa qualidade na tarefa de síntese.
Uma estratégia para tentar solucionar este problema é a variação na taxa de atualização dos pesos neurais, por exemplo: o gerador é atualizado a cada iteração ao passo que o discriminador é atualizado a cada três iterações.

Por limitações de tempo e de *hardware*, esta varredura inicial é feita com apenas 10 mil dados e considerando apenas 40 épocas de treino com *learning rate* fixa. Com apoio da ferramenta Weights & Biases, combinou-se aleatoriamente estes parâmetros para obter quinze modelos. Durante o treinamento, avalia-se em cada época a evolução da *loss* do gerador do dicscriminador, de forma que, ao final do treino retém-se o modelo da última época e o modelo que levou ao menor valor da *loss* do gerador no conjunto de validação. Para cada um dos quinze modelos, esse último é analisado no conjunto de testes de maneira qualitativa (análise subjetiva dos alunos quanto aos resultados) e quantitativa (cálculo das métricas FID e SSIM) (resultados em [Resultados preliminares com 10 mil dados de treinamento da GAN](#resultados-preliminares-com-10-mil-dados-de-treinamento-da-gan)). 

A partir desta análise inicial, seleciona-se três modelos para prosseguir com o treinamento com todos os dados disponíveis (60 mil). Os três melhores modelos são treinados mantendo os hiper-parâmetros que lhes foram atribuídos na primeira etapa, com exceção da *learning rate* cujo valor inicial é fixado em 0,0002. Partindo dos modelos obtidos no treinamento inicial, cada modelo é treinado por mais 50 épocas, com uma *learning rate* que descresce linearmente a partir da época 10. Os modelos obtidos na última época desse treino adicional são então avaliados de forma quantitativa e qualitativa. Devido a limitações de hardware, apenas um dos três melhores modelos segue para o teste de utilidade. Tais testes serão descritos em mais detalhes na seção [Métricas de Avaliação](#métricas-de-avaliação).

Em suma, o fluxo de trabalho proposto por este projeto está ilustrado na Figura 12. Inicia-se com a obtenção da base de dados ATM'22 e seu devido tratamento, conforme detalhado na seção anterior.
Utilizando estes dados, alimenta-se a rede generativa com as fatias segmentadas (máscaras binárias). Já a rede discriminadora recebe os dados reais (sem segmentação) e os dados sintéticos, devendo classificar cada um como "real" ou "falso".
Após o treinamento, avalia-se os dados sintéticos a partir de três perspectivas: análise qualitativa, análise quantitativa e análise de utilidade, as quais serão descritas em detalhes nas próximas seções deste relatório.

![Fluxo para treinamento da PulmoNet.](figs/workflow_completo.png?raw=true)

*Figura 12: Fluxo para treinamento da PulmoNet.*

Destaca-se ainda que, em operação (após a fase treinamento), espera-se que o modelo receba máscaras binárias com o formato do pulmão somadas a um ruído, retornando imagens CT que preencham a região pulmonar, bem como seu entorno. Uma mesma máscara binária pode gerar imagens sintéticas distintas, devido ao ruído aleatório adicionado na entrada do modelo. Os dados sintéticos devem ser bons o suficiente para ajudarem no treinamento de modelo de segmentação das vias aéreas (teste de utilidade) e potencialmente substituir o uso de dados reais, para a preservação da privacidade dos pacientes.

### Ferramentas Relevantes
A ferramenta escolhida para o desenvolvimento da arquitetura dos modelos e de treinamento é o **PyTorch**, em função de sua relevância na área e familiaridade por parte dos integrantes do grupo. Para o versionamento dos modelos e para ajustar seus hiperparâmetros, decidiu-se pela ferramenta **Weights & Biases (Wandb AI)** dentre as opções disponíveis no mercado. E, além disso, a ferramenta do **GitHub** também auxilia no versionamento dos algoritmos desenvolvidos.

### Métricas de Avaliação
Para avaliar a qualidade dos resultados obtidos com o modelo de síntese, propõe-se três tipos de avaliação: análise qualitativa, análise quantitativa e análise de utilidade.

#### Análise Qualitativa
Esta estratégia consiste na própria observação e avaliação dos dados sintéticos por meio dos estudantes. Os dados sintéticos são comparados aos reais, avaliando se eles se mostram muito distantes de uma CT pulmonar ou se o modelo está se encaminhando para bons resultados. Uma vez que os estudantes não possuem conhecimento clínico para julgar imagens sintéticas muito próximas das reais, essa etapa faz sentido num momento inicial, onde as diferenças são facilmente percebidas. Em uma fase posterior, em que as amostras reais e sintéticas são difíceis de serem distinguidas por leigos, a simples avaliação dos estudantes não se torna relevante, e é preciso buscar conhecimento especializado. 

#### Análise Quantitativa
A análise quantitativa trata de uma avaliação sobre as imagens a partir dos métodos **Fréchet Inception Distance (FID)** e **Structural Similarity Index (SSIM)**, os quais são utilizados para avaliação da qualidade das imagens sintéticas e da similaridade com os dados reais. Ambas estratégias foram utilizadas pelos pesquisadores do artigo [[1]](#1), permitindo comparar os resultados produzidos no projeto proposto frente a esse trabalho utilizado como base [[1]](#1).

A métrica FID avalia o desempenho da rede generativa e é calculada utilizando uma rede neural pré-treinada *InceptionV3*, que extrai *features* das fatias pulmonares geradas e das fatias originais. Com isso, as distribuições dos dados sintéticos e dos dados reais, obtidas pelo *encoder* desta rede, são usadas para calcular a FID e, assim, avaliar a qualidade da imagem gerada.
A expressão matemática que descreve o cálculo da FID entre duas distribuições Gaussianas criadas pelas *features* da última camada de *pooling* do modelo *Inception-v3* é dada por:

$$FID = ‖𝜇_{𝑟} − 𝜇_{𝑔}‖^{2} + Tr(\sum_{𝑟} + \sum_{𝑔} − 2(\sum_{𝑟}\sum_{𝑔})^{1∕2})$$

onde $𝜇_{𝑟}$ e $𝜇_{𝑔}$ são as médias entre as imagens reais e sintéticas, e $\sum_{𝑟},\ \sum_{𝑔}$ são as matrizes de covariância para os vetores de *features* dos dados reais e gerados, respectivamente.
Quanto menor for o FID, maior a qualidade da imagem gerada.

Por sua vez, a métrica SSIM compara a imagem gerada com seu respectivo *ground-truth* com base em três características: luminância, distorção de contraste e perda de correlação estrutural.
As expressões matemáticas usadas para o cálculo desta métrica são:

$$SSIM(𝑥, 𝑦) = l(𝑥, 𝑦) \times 𝑐(𝑥, 𝑦) \times 𝑠(𝑥, 𝑦)$$

$$
l(x, y) = \frac{2\mu_{x}\mu_{y} + C_{1}}{\mu_{x}^{2} + \mu_{y}^{2} + C_{1}}
$$

$$
c(x, y) = \frac{2\sigma_{x}\sigma_{y} + C_{2}}{\sigma_{x}^{2} + \sigma_{y}^{2} + C_{2}}
$$

$$𝑠(𝑥, 𝑦) = \frac{𝜎_{𝑥𝑦} + 𝐶_{3}}{𝜎_{𝑥}𝜎_{𝑦} + 𝐶_{3}}$$

onde $𝜇_{𝑥}$, $𝜇_{𝑦}$, $𝜎_{𝑥}$, $𝜎_{𝑦}$, e $𝜎_{𝑥𝑦}$ são as médias locais, variâncias e covariâncias cruzadas para as imagens 𝑥, 𝑦, respectivamente. $𝐶_{1}$, $𝐶_{2}$ $𝐶_{3}$ são constantes.

No caso do cálculo do SSIM, como o foco do projeto está associado com uma boa geração de vias aéreas pulmonares, esta métrica é calculada considerando tanto a saída completa (imagem 512 x 512) quanto apenas a região central (imagem 256 x 256).

#### Análise de Utilidade
Nesta etapa, avalia-se a utilidade do gerador, em termos de **transfer learning**. Isto é, tomando como inspiração a abordagem explorada em [[9]](#9), implementa-se uma rede similar à U-Net - com a mesma estrutura da rede geradora Pix2Pix da PulmoNet - para realizar a segmentação das vias aéreas e compara-se o desempenho desta U-Net com uma outra rede que utiliza o aprendizado do nosso gerador.
Para isto, coloca-se na entrada da rede de segmentação imagens completas de pulmões e compara-se as saídas geradas pela U-Net com a própria segmentação presente na base de dados ATM'22 feita por especialistas, conforme ilustrado no fluxograma abaixo (Fig. 13).
A seleção entre o tipo de modelo (inicialização aleatória ou pesos transferidos do nosso gerador) é definida em um arquivo YAML, bem como outros parâmetros de configuração.
Para a avaliação de desempenho destas redes, calcula-se o coeficiente DICE (obtido a partir da precisão e *recall* da predição), tomando como referência o artigo [[2]](#2).
Além disso, optou-se pela DiceLoss como função de *loss*, dado que é tipicamente utilizada em tarefas de segmentação de imagens médicas [[12]](#12).

![Fluxo para treinamento da rede de segmentação de vias aérea para o teste de utilidade.](figs/workflow_unet.png?raw=true)

*Figura 13: Fluxo para treinamento da rede de segmentação de vias aérea para o teste de utilidade.*

A métrica Dice varia entre 0 e 1, sendo que quanto maior for este resultado, melhor, e a função DiceLoss também varia entre 0 e 1, mas quanto menor for este valor, melhor, isto é, busca-se minimizar a DiceLoss. Elas são calculadas seguindo as seguintes expressões matemáticas:

![Equação matemática para cálculo da métrica Dice.](figs/dice_metric.png?raw=true)

*Figura 14: Equação matemática para cálculo da métrica Dice. Retirado de https://medium.com/@nghihuynh_37300/understanding-evaluation-metrics-in-medical-image-segmentation-d289a373a3f*

$$
DiceLoss = 1 - Dice
$$

Para aproveitar os pesos iniciais da GAN para a tarefa de segmentação, realiza-se o seguinte processo de *transfer learning*: congela-se apenas a parte da rede codificadora do gerador, retreinando somente o decodificador (Fig. 15), utilizando apenas 60% dos dados separados para esse propósito. Com isso, espera-se demonstrar a capacidade de mapeamento da nossa GAN para um espaço latente adequado, que contenha informações acerca das vias aéreas e que tais informações ajudem a aprimorar esta tarefa.

![Arquitetura da rede de segmentação das vias aéreas. Modelo do gerador da PulmoNet com camadas congeladas na rede codificadora para a aplicação do transfer learning.](figs/arquitetura_unet.png?raw=true)

*Figura 15: Arquitetura da rede de segmentação das vias aéreas. Modelo do gerador da PulmoNet com camadas congeladas na rede codificadora para a aplicação do transfer learning.*

Por fim, é importante destacar o caminho a ser seguido para a avaliação da rede generativa para as saídas em 3D, caso seja possível implementá-las dentro do prazo do projeto. Para esta aplicação, geraríamos um volume sintético e passaríamos esta saída pela rede de segmentação *medpseg* [[10]](#10). Feito isso, compararíamos as vias aéreas segmentadas com o *ground-truth* estabelecido na própria base de dados ATM'22.

### Cronograma
O projeto é implementado seguindo o seguinte fluxo lógico:

![Fluxo lógico das ativaidades para desenvolvimento da PulmoNet.](figs/fluxo_logico.png?raw=true)

*Figura 16: Fluxo lógico das ativaidades para desenvolvimento da PulmoNet.*

Dado este fluxo, estipulamos o seguinte cronograma para desenvolvimento do projeto:

| Nº da Tarefa | Descrição                                                                 | Data Prevista de Finalização | Semanas Entre Etapas |
|--------------|---------------------------------------------------------------------------|------------------------------|----------------------|
| 1            | Leitura de artigos, familiarização com a base de dados e GANs             | 10/09                        |                      |
| 2            | Primeira versão da GAN (inspirada no artigo de referência)                | 24/09                        | 2 semanas            |
| 3            | Estrutura de avaliação bem delimitada                                     | 07/10                        | 2 semanas            |
| 4            | E2                                                                        | 08/10                        | 1 dia                |
| 5            | Primeiros resultados com imagens segmentadas e valores para validação     | 15/10                        | 1 semana             |
| 6            | Fine-tuning e aperfeiçoamento do modelo                                   | 29/10                        | 2 semanas            |
| 7            | Evoluir para redes 3D ou continuar aperfeiçoando o modelo                 | 05/11                        | 1 semana             |
| 8            | E3                                                                        | 25/11                        | 3 semanas            |



### Ambiente Computacional

Os modelos da GAN foram treinados em uma máquina com uma GPU NVIDIA GeForce RTX 3060 com 12GB de VRAM, 64GB de memória RAM e um processador AMD Ryzen 5 5600. Nessa máquina foram realizados todos os treinamentos da GAN, sendo que aqueles que utilizaram 10k imagens demoraram 10 horas por modelo, enquanto os treinos com as 60k imagens levaram cerca de 30 horas por modelo.

Já o modelo da rede de segmentação, para o teste de utilidade, foi treinado em um computador pessoal que tinha uma GPU NVIDIA GeForce RTX 3050 com 4GB de VRAM, 16GB de memória RAM e processador Intel i5 de 12ª geração. Nesse computador foram rodadas algumas versões da rede de segmentação utilizadas para avaliação, sendo que o tempo necessário por batch foi de cerca de 10s, com o tempo total variando conforme a quantidade de dados e os critérios de early stopping.


## Experimentos, Resultados e Discussão dos Resultados

### Resultados preliminares com 10 mil dados de treinamento da GAN
A PulmoNet - o modelo de GAN proposto em nosso projeto - passou por uma etapa de busca pelas configurações e hiperparâmetros de treinamentos ótimos, a fim de encontrar uma combinação que gerasse tomografias pulmonares mais realistas. Para isto, testou-se quinze configurações distintas, com uma parcela dos dados selecionados para o treinamento da GAN. Tanto as  configurações de cada modelo avaliado nesta varredura quanto as métricas qualitativas e quantitativas obtidas estão apresentadas na Tabela 3.

*Tabela 3: Resultados varredura inicial de hiper parâmetros. A cada modelo, atribuí-se um nome aleatório.*
| Modelo |	Relação Passos (Disc/Gen) |	Ruído |	Ruído só no pulmão |	Intensidade do Ruído	|Média Ruído para Gaussiano |Desvio Ruído para Gaussiano	| Critério	| Regularizador	| Nível Regularização	| Learning Rate	| Beta | Época de menor *loss* de validação (gerador)| Análise Qualitativa |
| ----- | ----- | -----   | ----- | -----       | -----         | -----         |   -----  | ----- | -----| -----   |   -----       | -----| -----   |
|Sweep10|	4/2	|Gaussiano|	Falso |	0,316 |	0,747 |	0,178 |	BCELoss|	MSE|	8|	3,11E-04|	0,460| 17 | Bom |
|Sweep205|	3/1	|Gaussiano|	Verdadeiro|	0,557 |	0,512 |	0,390|	MSELoss|	MAE|	10|	2,85E-04|	0,755| 11 | Bom |
|Sweep412|	1/1| Gaussiano|	Falso| 0,757 |	0,525 |	0,475|	MSELoss|	MAE|	4	|1,70E-04	|0,881| 6 | Bom |
|Sweep64	|1/2	|Gaussiano	|Verdadeiro	|0,818	|0,560	|0,223	|MSELoss	|MAE	|3	|3,75E-04	|0,866| 10 | Médio |
|Sweep123	|2/1	|Gaussiano	|Verdadeiro	|0,332	|0,653	|0,335 |MSELoss	|MAE	|4	|1,55E-04	|0,625| 6 | Médio |
|Sweep284	|1/2	|Gaussiano	|Verdadeiro	|0,488	|0,872	|0,447 |MSELoss	|MSE	|4	|2,24E-04	|0,678| 9 | Médio |
|Sweep394	|2/1	|Gaussiano	|Falso	|0,371	|0,700 |0,287	|BCELoss	|MAE	|1	|3,40E-04	|0,479| 34 | Médio |
|Sweep497	|1/1	|Gaussiano	|Verdadeiro	|0,304	|0,875	|0,290	|MSELoss	|MSE	|15	|1,32E-04	|0,841| 6 | Médio |
|Sweep522	|4/2	|Gaussiano	|Falso	|0,877	|0,693	|0,379|MSELoss	|MSE_mask	|13	|3,40E-04	|0,573| 29 | Médio |
|Sweep71	|2/1	|Gaussiano	|Verdadeiro	|0,817	|0,549	|0,326	|BCELoss	|MSE_mask	|1	|2,82E-04	|0,526| 32 | Ruim |
|Sweep185	|4/1	|Uniforme	|Verdadeiro	|0,356 |	0,590 |	0,216|	MSELoss|	MAE_mask|	5|	2,82E-04|	0,424| 38 | Ruim |
|Sweep186	|2/1	|Uniforme	|Verdadeiro	|0,979 |	0,531	|0,262	|BCELoss	|MAE_mask	|4	|1,87E-04	|0,607| 40 | Ruim |
|Sweep256	|1/2	|Gaussiano	|Verdadeiro	|0,308|0,681	|0,135	|MSELoss	|MAE_mask	|8	|3,16E-04	|0,470| 1 | Ruim |
|Sweeo279	|4/2	|Gaussiano	|Falso	|0,682	|0,968 |0,102	|MSELoss	|MAE_mask	|15	|2,58E-04	|0,647| 1 | Ruim |
|Sweep464	|2/2	|Gaussiano	|Verdadeiro	|0,986	|0,993	|0,101	|MSELoss	|MSE_mask	|1	|2,91E-04	|0,439| 38 | Ruim |


Com relação à análise qualitativa, cada um dos três membros deste projeto examinaram algumas imagens sintéticas e classificaram o modelo entre três categorias: "Bom", "Médio" e "Ruim".
Nesta análise qualitativa, considerou-se a definição das bordas e da região externa ao pulmão, além do preenchimento na região com as vias aéreas.
Alguns exemplos destas imagens são apresentados em anexo, em [Varredura dos parâmetros da GAN para 10 mil dados](docs/modelos_sweep.pdf?raw=true).

Considerando a média das avaliações qualitativas, apenas um modelo recebeu três votos "Bom" (Sweep 205), um modelo recebeu dois votos "Bom" (Sweep 10) e dois modelos receberam apenas um voto "Bom" (Sweep 412 e 123), de modo que filtrou-se 4 dos 15 modelos. Destes dois modelos com apenas um voto "Bom", comparou-se o FID e o SSIM para selecionar a configuração que prosseguiria com o treinamento. Em função destas métricas, considerou-se que o Sweep 412 tinha mais potencial para aprimorar e gerar boas imagens sintéticas. Assim, foram escolhidas as configurações **Sweep 205, Sweep 10 e Sweep 412** para a realização do treinamento com toda a base de dados disponível.

Ainda sobre a análise qualitativa dos resultados, é pertinente observar que todos os modelos que foram treinados com regularização apenas na região do pulmão (interior da máscara binária) não tiveram bons resultados, e foram julgados como "Ruim".

Ressalta-se também que esta etapa preliminar de seleção e varredura da combinação de parâmetros da treinamento do modelo é primordial para potencializar bons resultados, visto a instabilidade dos treinos das GANs. Além disso, esta estratégia também economiza tempo e recursos, já que previne que o treinamento completo do modelo seja alocado em uma configuração potencialmente ruim.

### Resultados com 60 mil dados de treinamento da GAN

**Análise Qualitativa**

As figuras 17 a 19 trazem exemplos de imagens geradas pelos três modelos selecionados após o treinamento complementar dos mesmos com os 60 mil dados disponíveis para a tarefa de treinamento da GAN. Como referência, ao lado de cada imagem gerada, apresenta-se a imagem real cuja máscara foi utilizada como entrada para o gerador. Todos os modelos inovam pouco na geração, e têm dificuldade na geração das estruturas internas ao pulmão. 

Visualmente, dentre os três modelos considerados, o Sweep10 se mostra o modelo com mais dificuldade de gerar imagens nítidas, de forma que as bordas das estruturas parecem "borradas" (Animação 1). Os modelos Sweep205 e Sweep412 são mais difíceis de serem comparados, visto que as imagens geradas parecem apresentar o mesmo nível de qualidade, e é necessário conhecimento especialista para indicar qual é melhor. Dos quatro exemplos apresentados para cada modelo, tanto o Sweep205 e Sweep412 apresentam duas imagens sintéticas onde o interior do pulmão não aparece vazio. A nitidez das imagens geradas por tais modelos também é similar, e, quanto a capacidade de inovação dos modelos, para os dois, as quatro imagens geradas apresentam variações com relação as imagens originais. As variações mais perceptivas estão relacionadas a presença ou ausência de "furos" na região cinzenta, ou uma alteração do formato dos mesmos. Tais variações podem indicar uma capacidade do modelo de inovar e criar imagens sintéticas que não se limitem a replicar as imagens usadas no treino, ou podem indicar mudanças que não correspondem a anatomia humana, e, logo, inúteis ao propósito de auxiliar no treino de outras redes para aplicação clínica. 

![Imagens geradas pelo modelo Sweep10 após segunda etapa de treino. A esquerda de cada imagem sintética, apresenta-se a imagem real a partir da qual extraiu-se a máscara pulmonar.](figs/sweep10.png?raw=true)

*Figura 17: Imagens geradas pelo modelo Sweep10 após segunda etapa de treino. A esquerda de cada imagem sintética, apresenta-se a imagem real a partir da qual extraiu-se a máscara pulmonar.*

![Imagens geradas pelo modelo Sweep205 após segunda etapa de treino. A esquerda de cada imagem sintética, apresenta-se a imagem real a partir da qual extraiu-se a máscara pulmonar.](figs/sweep205.png?raw=true)

*Figura 18: Imagens geradas pelo modelo Sweep205 após segunda etapa de treino. A esquerda de cada imagem sintética, apresenta-se a imagem real a partir da qual extraiu-se a máscara pulmonar.*

![Imagens geradas pelo modelo Sweep412 após segunda etapa de treino. A esquerda de cada imagem sintética, apresenta-se a imagem real a partir da qual extraiu-se a máscara pulmonar.](figs/sweep412.png?raw=true)

*Figura 19: Imagens geradas pelo modelo Sweep412 após segunda etapa de treino. A esquerda de cada imagem sintética, apresenta-se a imagem real a partir da qual extraiu-se a máscara pulmonar.*

![](figs/ezgif.com-animated-gif-maker.gif?raw=true)
*Animação 1: Animação das imagens geradas pelos modelos Sweep10, 205 e 412.*

****
**Análise Quantitativa**

Para a análise quantitativa, os três melhores modelos obtidos foram comparados por meio da FID e do SSIM, comparando os treinamentos feitos com apenas 10 mil dados e o treinamento feito com os 60 mil. 

É possível notar que a utilização de uma maior quantidade de dados reduziu o valor da FID nos três modelos apresentados, demonstrando que as imagens passaram a ter mais qualidade, mas os resultados se mantiveram elevados para os três. A SSIM various entre os modelos: enquanto no Sweep10 ela se elevou ao utilizar mais dados, ela caiu para o Sweep205 e para o 412. Em geral, os valores de SSIM tiveram resultados similares, com a Sweep10 gerando resultados mais similares em relação aos dados reais, enquanto os outros modelos apresentaram resultados mais "criativos", no entanto, é difícil confirmar a qualidade dessa informação devido a elevada FID dos três modelos em conjunto com a proximidade dos valores obtidos da SSIM. 

Também é possível notar que a SSIM realizada na imagem total obteve valores maiores do que a realizada apenas no centro da imagem gerada. Considerar todo o pulmão reduziu a similaridades das imagens nos três modelos analisados em comparação com as reais, demonstrando que a região central, onde deviam ser geradas as vias aéreas, teve uma baixa variação, confirmando a dificuldade em gerar dados relativos a elas.

*Tabela 4: Resultados para os modelos treinados com 60 mil dados.*
| Modelo | FID (10k) | FID (60k) | SSIM completo (10k) | SSIM completo (60k) | SSIM central (10k) | SSIM central (60k) |
| ---- | ---- | ---- | ---- | ---- | ---- | ---- |
| Sweep10 | $335.24$ | $293.99$ | $0.62 \pm 0.24$ | $0.68 \pm 0.23$ | $0.70 \pm 0.21$ | $0.77 \pm 0.18$ |
| Sweep205 | $327.52$ | $311.54$ | $0.64 \pm 0.23$ | $0.63 \pm 0.23$ | $0.74 \pm 0.19$ | $0.72 \pm 0.18$ |
| Sweep412 | $320.07$ | $304.82$ | $0.69 \pm 0.23$ | $0.61 \pm 0.23$ | $0.78 \pm 0.17$ | $0.70 \pm 0.19$ |

A comparação com a literatura base é incerta por não estarem sendo utilizados os mesmos conjuntos de dados e por terem objetivos diferentes: o artigo [[1]](#1) gera apenas o conteúdo interior do pulmão, enquanto a metodologia proposta gera a tomografia como um todo. Mesmo assim, a comparação de FID e SSIM podem ser vistas nas tabelas abaixo, onde o FID mostrou a maior divergência, com a literatura atingindo valores muito inferiores aos alcançados no projeto, enquanto a SSIM foi mais similar, com a literatura tendo resultados maiores/mais similaridade para a imagem completa, o que faz sentido considerando que não ela não foca em gerar conteúdo externo ao pulmão, enquanto para a região central o resultado da SSIM foi inferior ao alcançado neste trabalho, mostrando que os resultados da literatura foram mais criativos e menos similares ao conjunto original na geração das vias aéreas.

*Tabela 5: Comparação da FID obtida em comparação com o artigo de referência.*
| Modelo | $FID_{InceptionV3}$ |
| ------- | ------- |
| $Sweep10$ | 293.994 |
| $Sweep205$ | 311.541 |
| $Sweep412$ | 304.826 |
| $P2P_{𝐿𝐼𝐷𝐶}$ (Mendes et al., 2023) | 12.82 |


*Tabela 7: Comparação da SSIM para as imagens completas(512x512) e com o corte centralizado (256x256).*
| Modelo | $ SSIM_{512} $ | $ SSIM_{256} $ |
| ------- | ------- | ------- |
| | $𝜇 \pm 𝜎$ | $𝜇 \pm 𝜎$ |
| $Sweep10$ | $0.682 \pm 0.238$ |$0.773 \pm 0.180$ |
| $Sweep205$ | $0.635 \pm 0.235$ | $0.725 \pm 0.184$ |
| $Sweep412$ | $0.616 \pm 0.237$ | $0.709 \pm 0.1912$ |
| $P2P_{𝐿𝐼𝐷𝐶}$ (Mendes et al., 2023) | $0.803 \pm 0.122$ | $0.651 \pm 0.083$ |

****
**Teste de Utilidade**

A arquitetura do gerador descrito na seção [Modelo Proposto](#modelo-proposto) é utilizada para a tarefa de segmentação de vias aéreas em duas etapas: inicialização aleatória dos pesos de aprendizado da rede neural e aplicação da técnica de *transfer learning* a partir da melhor versão da PulmoNet.
Em ambas as estratégias, foram encontradas dificuldades em gerar segmentações de qualidade, de modo que frequentemente os modelos passam a reproduzir imagens complemente pretas, não apresentando nenhuma via aérea.
Apesar de testes não formais terem sido implementados para tentar aprimorar os parâmetros da nossa arquitetura, este comportamento já era esperado, dado que a tarefa de segmentação é extremamente complexa e seria necessário uma alocação de recursos e esforços mais dedicados na compreensão e busca de aprimoramento desta tarefa, do que simplesmente aplicar a arquitetura do gerador da PulmoNet.
Ainda assim, foi possível treinar e testar modelos para a segmentação das vias aéreas, conforme será descrito nos próximos parágrafos.

A rede treinada com pesos iniciais aleatórios (sem as camadas de dropout na arquitetura do gerador) atingiu uma *loss* de validação mínima de 0.9899.
Nas figuras abaixo, apresenta-se alguns exemplos de vias aéreas segmentadas durante a etapa de testes deste modelo.
Nota-se que parte das vias aéras está presente nas imagens geradas, mas também há uma região considerável de alucinação, isto é, a rede cria vias aéreas que não existem de fato.

![Exemplos de saídas da rede de segmentação - from zero.](figs/unet_results/model_trained_from_zero/example_generated_epoch_1.png?raw=true)
![Exemplos de saídas da rede de segmentação - from zero.](figs/unet_results/model_trained_from_zero/example_generated_epoch_3.png?raw=true)
![Exemplos de saídas da rede de segmentação - from zero.](figs/unet_results/model_trained_from_zero/example_generated_epoch_5.png?raw=true)
![Exemplos de saídas da rede de segmentação - from zero.](figs/unet_results/model_trained_from_zero/example_generated_epoch_9.png?raw=true)
![Exemplos de saídas da rede de segmentação - from zero.](figs/unet_results/model_trained_from_zero/example_generated_epoch_15.png?raw=true)

*Figura 20: Exemplos de saídas da rede de segmentação, treinada com pesos iniciais aleatórios.*

Já a rede pré-treinada para a síntese de CTs pulmonares (modelo Sweep 412), passou pela etapa de transferência de aprendizado com uma quantidade reduzida de dados com relação ao treinamento da U-Net a partir do zero.
Na arquitetura deste modelo, as camadas de dropout também foram removidas e os pesos da rede codificadora se mantiveram inalterados.
Exemplos de figuras sintéticas geradas com este processo estão ilustradas abaixo.
Nota-se que nas imagens com baixo contraste entre o pulmão e a região externa, a rede gerada tinha dificuldades em identificar as vias aéreas, ao passo que nas imagens com alto contraste entre a região do pulmão e os músculos torácicos o modelo de segmentação alucinou.
Isto é, em uma análise qualitativa, a rede não teve bons resultados quando aplicado o método de *transfer learning*.

![Exemplos de saídas da rede de segmentação - transfer learning.](figs/unet_results/model_transfer_learning/example_generated_epoch_3.png?raw=true)
![Exemplos de saídas da rede de segmentação - transfer learning.](figs/unet_results/model_transfer_learning/example_generated_epoch_6.png?raw=true)
![Exemplos de saídas da rede de segmentação - transfer learning.](figs/unet_results/model_transfer_learning/example_generated_epoch_14.png?raw=true)
![Exemplos de saídas da rede de segmentação - transfer learning.](figs/unet_results/model_transfer_learning/example_generated_epoch_18.png?raw=true)
![Exemplos de saídas da rede de segmentação - transfer learning.](figs/unet_results/model_transfer_learning/example_generated_epoch_20.png?raw=true)

*Figura 21: Exemplos de saídas da rede de segmentação, treinada via tranfer learning do modelo Sweep412.*

Por fim, calculando a métrica DICE para ambos os modelos, tem-se os resultados da tabela abaixo.

*Tabela 7: Resultados para o teste de utilidade.*
| Modelo | Melhor época | Dice |
| ------ | ------------ | ---- |
| Modelo com pesos inciais aleatórios | 1 | 0.02|
| Transfer learning a partir do modelo Sweep412 | 29 | 0.02|

Fica evidente que o método de *transfer learning* não teve impacto positivo significativo na tarefa de segmentação, retornando um Dice próximo à U-Net inicializada com pesos aleatórios.
Este comportamento pode ter ocorrido devido ao baixo preenchimento da região interna pulmonar por parte da PulmoNet, conforme já comentado anteriormente neste relatório.
Em outras palavras, como o gerador do nosso modelo não preencheu detalhadamente a área interna do pulmão, onde ficam as vias aéreas, é possível que a rede codificadora do gerador não tenha aprendido tão bem como codificar essa região anatômica e, portanto, não contribua com a tarefa de segmentação.

## Conclusão
O projeto da PulmoNet busca a geração de fatias de CTs pulmonares a partir de máscaras binárias, em duas dimensões, baseada em GANs. Esta rede utiliza uma arquitetura Pix2Pix para o gerador e uma PatchGAN para o discriminador. São usados dados da base pública ATM'22, cujos dados correspondem a volumes pulmonares de tomografias e segmentações das vias aéreas feitas por especialistas. Para a avaliação da qualidade da rede, propõe-se métodos qualitativos, quantitativos e análises de utilidade.

Nosso projeto conseguiu implementar modelos de síntese com qualidade considerável, frente a uma base de dados distinta daquela usada tipicamente nas tarefas de síntese de CTs pulmonares. Mais ainda, as imagens sintéticas geradas pela PulmoNet incluem tanto a região interna quanto a região externa do pulmão, fornecendo uma saída completa de uma tomografia (fatiada transversalmente). Destarte, nossa pesquisa pode ser considerada um passo inicial na busca de melhores modelos de síntese de CTs pulmonares, estabelecendo uma metodologia e uma aplicação da base ATM'22 para esta outra finalidade.

Por fim, ressalta-se que os resultados gerados não são ideais e há ainda uma boa margem para aprimorar. Por exemplo, a métrica FID resultante ainda está demasiadamente elevada frente a trabalhos similares da literatura. Ademais, a análise de utilidade não forneceu resultados promissores para nosso modelo, o que indica que há um caminho a ser percorrido para geração de imagens que contenham informações anatômicas significativas sobre a região das vias aéreas.

### Perspectivas Futuras
Novos estudos poderão ser elaborados a partir da nossa pesquisa, tais como:
- Aplicação de um método de fine-tunning do nosso melhor modelo, focado apenas na região interna do pulmão. Com isso, buscar-se-ia um modelo que realiza-se um preenchimento melhor desta região. Caso este novo modelo tenha sucesso nesta tarefa, espera-se que o teste de utilidade aplicado a ele seja melhor do que os resultados que obbtivemos nesta pesquisa.
- Estudar arquiteturas alternativas para a tarefa de síntese de CTs pulmonares, tais como modelos de difusão.
- Aprofundar este modelo para a geração de CTs pulmonares em três dimensões.

## Referências Bibliográficas

<a id="1">[1]</a> : José Mendes et al., Lung CT image synthesis using GANs, Expert Systems with Applications, vol. 215, 2023, pp. 119350., https://www.sciencedirect.com/science/article/pii/S0957417422023685.

<a id="2">[2]</a> : Minghui Zhang et al., Multi-site, Multi-domain Airway Tree Modeling (ATM'22): A Public Benchmark for Pulmonary Airway Segmentation, https://arxiv.org/abs/2303.05745.

<a id="3">[3]</a> :  Jacopo Lenkowicz et al., A deep learning approach to generate synthetic CT in low field MR-guided radiotherapy for lung cases, Radiotherapy and Oncology, vol. 176, 2022, pp. 31-38, https://www.sciencedirect.com/science/article/pii/S0167814022042608.

<a id="4">[4]</a> : Swati P. Pawar and Sanjay N. Talbar, LungSeg-Net: Lung field segmentation using generative adversarial network, Biomedical Signal Processing and Control, vol. 64, 2021, 102296, https://www.sciencedirect.com/science/article/pii/S1746809420304158.

<a id="5">[5]</a> : Tekatli, Hilâl et al. “Artificial intelligence-assisted quantitative CT analysis of airway changes following SABR for central lung tumors.” Radiotherapy and oncology : journal of the European Society for Therapeutic Radiology and Oncology vol. 198 (2024): 110376. doi:10.1016/j.radonc.2024.110376, https://pubmed.ncbi.nlm.nih.gov/38857700/

<a id="6">[6]</a> : Zhang, Ling et al. “Generalizing Deep Learning for Medical Image Segmentation to Unseen Domains via Deep Stacked Transformation.” IEEE transactions on medical imaging vol. 39,7 (2020): 2531-2540. doi:10.1109/TMI.2020.2973595, https://www.ncbi.nlm.nih.gov/pmc/articles/PMC7393676/

<a id="7">[7]</a> : Hofmanninger, J., Prayer, F., Pan, J. et al. Automatic lung segmentation in routine imaging is primarily a data diversity problem, not a methodology problem. Eur Radiol Exp 4, 50 (2020). https://doi.org/10.1186/s41747-020-00173-2

<a id="8">[8]</a> : Isola, P., Zhu, J. Y., Zhou, T., & Efros, A. A. (2017). Image-to-image translation with conditional adversarial networks. In Proceedings - 30th IEEE conference on computer vision and pattern recognition, CVPR 2017. http://dx.doi.org/10.1109/CVPR.2017. 632, arXiv:1611.07004.

<a id="9">[9]</a> : Radford, A., Metz, L., and Chintala, S., “Unsupervised Representation Learning with Deep Convolutional Generative Adversarial Networks”, <i>arXiv e-prints</i>, Art. no. arXiv:1511.06434, 2015. doi:10.48550/arXiv.1511.06434.

<a id="10">[10]</a> : Carmo, D. S., “MEDPSeg: Hierarchical polymorphic multitask learning for the segmentation of ground-glass opacities, consolidation, and pulmonary structures on computed tomography”, <i>arXiv e-prints</i>, Art. no. arXiv:2312.02365, 2023. doi:10.48550/arXiv.2312.02365.

<a id="11">[11]</a> : Goodfellow, I. J., Pouget-Abadie, J., Mirza, M., Xu, B., Warde-Farley, D., Ozair, S., Courville, A., Bengio, Y., “Generative Adversarial Networks”, arXiv e-prints, Art. no. arXiv:1406.2661, 2014. doi:10.48550/arXiv.1406.2661.

<a id="12">[12]</a> : A. Keshavarzi and E. Angelini, "Few-Shot Airway-Tree Modeling Using Data-Driven Sparse Priors," 2024 IEEE International Symposium on Biomedical Imaging (ISBI), Athens, Greece, 2024, pp. 1-5, doi: 10.1109/ISBI56570.2024.10635527.

Documento com as referências extras identificadas: [Artigos](docs/Artigos%20Projeto%20Redes%20Generativas.pdf?raw=true).


# Anexos

## Explorações de arquiteturas e dados

Considerando a dificuldade do modelo em sintetizar estruturas no interior do pulmão, duas outras abordagens foram testadas. A primeira envolveu apenas uma mudança no processamento dos dados de treino, e a segunda uma mudança na arquitetura da PulmoNet.

Quanto a mudança dos dados, ao invés de simplesmente enviar ao gerador a máscara da região pulmonar, testou-se a ideia de somar a máscara do pulmão com a máscara das vias aéreas. Para isso, 17 mil imagens dos dados de treino da GAN que continham vias aéreas foram selecionadas e criou-se um novo dataset de treino, onde a entrada do gerador corresponde a $0,5 \times {Máscara\ Pulmonar} + 0,5 \times Máscara\ das\ Vias\ Aéreas$. Para aproveitar o conhecimento da região externa e forçar o modelo a focar na região interna ao pulmão, continuou-se o treinamento do modelo Sweep412 por mais 40 épocas, utilizando as mesmas configurações de *loss* e ruído, porém com esse novo dataset de treino. Como teste, comparou-se as imagens sintéticas geradas pelo Sweep412 e pelo mesmo após esse treino adicional, considerando como entrada de ambos apenas a máscara do pulmão (Fig. A1). O impacto desse novo treino foi mínimo, com a adição de poucas ou nenhuma estrutura no interior do pulmão. Além disso, o novo modelo pareceu mais propenso a erros, gerando estruturas indesejáveis na região externa ao pulmão (Fig. A1, exemplos 2 e 3). Desse modo, conclui-se que a estratégia de *fine-tuning* explorada nesse experimento causa uma melhora irrelevante no interior do pulmão e não justifica o custo computacional da abordagem. 

![Exemplos de imagens sintéticas geradas pelo modelo Sweep412 e pelo mesmo após treino adicional com as máscaras de pulmão e vias aéreas combinadas. Ambos modelos receberam como entrada a máscara somente do pulmão para geração dos exemplos. Ao lado de cada imagem gerada, apresenta-se a imagem original a partir da qual obteve-se a máscara pulmonar.](figs/sweep_com_mask.png?raw=true)
*Figura A1: Exemplos de imagens sintéticas geradas pelo modelo Sweep412 e pelo mesmo após treino adicional com as máscaras de pulmão e vias aéreas combinadas. Ambos modelos receberam como entrada a máscara somente do pulmão para geração dos exemplos. Ao lado de cada imagem gerada, apresenta-se a imagem original a partir da qual obteve-se a máscara pulmonar.*

Finalmente, outra alternativa explorada pelo grupo foi a modificação da arquitetura da pulmonet para geração simultânea da imagem CT e da segmentação das vias aéreas. Nessa abordagem, o gerador continuaria a receber apenas a máscara pulmonar, mas sua saída seria composta por dois canais, um primeiro com a imagem CT sintética, e um segundo com as vias aéreas da imagem sintética. O discriminador, por sua vez, ao invés de receber uma imagem com dois canais contendo a máscara pulmonar e uma imagem CT, receberia uma imagem de três canais: o primeiro com a máscara pulmonar, o segundo com as máscaras de vias aéreas (real ou sintética), e, o terceiro, com uma imagem CT (real ou sintética). Fora as alterações com relação a dimensões, outra alteração necessária para comportar essa mudança é que a *loss* do gerador deve considerar a regularização tanto da imagem CT quanto da segmentação das vias aéreas.

Considerando a abordagem descrita acima, propôs-se um treino com as mesmas configurações utilizadas para o Sweep412 (tipo e intensidade de ruído, *loss*, parâmetro do otimizador) partindo de pesos aleatórios e composto por 40 épocas com *learning rate* inicial de 0,0002, que decai linearmente após a época 10. Devido a limitações dos recursos computacionais, tal treino foi realizando usando apenas um terço dos daods disponíveis para o treino da GAN. Apesar de algumas variações terem sido testadas, os resultados dessa abordagem não foram animadores (Fig. A2), e, devido ao tempo, não pôde-se aprofundar nessa ideia e testar os limites das mudanças propostas. Tais mudanças estão disponíveis no código fornecido para que maiores explorações futuras possam ser feitas. 

![Exemplo de imagens sintéticas geradas pelo modelo treinado para geração de imagens CT e da segmentação das vias aéreas simultâneamente.](figs/anexo_gen_segm.png?raw=true)
*Figura A2: Exemplo de imagens sintéticas geradas pelo modelo treinado para geração de imagens CT e da segmentação das vias aéreas simultâneamente.*
