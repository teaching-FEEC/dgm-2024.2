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
7. [Conclusão](#conclusão)
    1. [Próximos Passos](#próximos-passos)
8. [Referências Bibliográficas](#referências-bibliográficas)

**ANEXOS**:
1. [Varredura dos parâmetros da GAN para 10 mil dados]()
2. [Testes adicionais com outras arquiteturas]()
3. [Como rodar os modelos](#how-to-run)

## Links Importantes
Links para apresentações de slides e vídeos para entregas E1, E2 e E3 para a disciplina:

[Link para o vídeo de apresentação E1](https://drive.google.com/file/d/1TlpQOlCh_lAI0-jPPMPWOzGZ_werCo3d/view?usp=sharing)

[Link para a apresentação de slides E1](https://docs.google.com/presentation/d/1b8W0Cw1eiTbWlJ0CJJ8eMRA4zyu2iLhYvggi55-mOb0/edit?usp=sharing)

[Link para a apresentação de slides E2](https://docs.google.com/presentation/d/1QH5_WpeTp7kQPSVB78ukK7msn-Tx09pZoM_3dWmeqC4/edit?usp=sharing)

> TODO: link E3

## Resumo (Abstract)

> TODO: Update

As tomografias computadorizadas (CT) pulmonares e a segmentação das vias aéreas são essenciais para o diagnóstico preciso de doenças pulmonares. Propõe-se a PulmoNet, uma rede para síntese de imagens 2D de CTs pulmonares, visando apoiar redes de segmentação e gerar dados sintéticos para bases de dados de outras redes neurais, como classificadores de tumores. Utilizando a base ATM'22, implementa-se uma arquitetura GAN com gerador Pix2Pix e discriminador PatchGAN, que preencherá máscaras binárias do pulmão com vias aéreas. Avalia-se a rede qualitativamente, quantitativamente (métricas FID e SSIM) e em utilidade. Resultados parciais indicam problemas no treinamento devido à velocidade de aprendizado do discriminador.

## Descrição do Problema/Motivação
As tomografias computadorizadas (CT) pulmonares, juntamente com a segmentação das vias aéreas, desempenham um papel crucial no diagnóstico preciso de doenças pulmonares. Ao gerar imagens detalhadas da região torácica, a tomografia permite que médicos mapeiem a anatomia das vias aéreas antes de procedimentos cirúrgicos, avaliando a extensão de lesões e facilitando o acompanhamento da progressão de doenças respiratórias [[2]](#2). Além disso, a CT é fundamental para monitorar a eficácia de tratamentos e detectar seus possíveis efeitos colaterais [[5]](#5).

A complexidade e diversidade do corpo humano, bem como o custo e acesso a CT, limitam a obtenção de grandes volumes de dados que sejam representativos das diversas condições anatómicas. Essa escassez de dados limita a performance de modelos de aprendizado de máquina, como redes neurais, que utilizam de tais dados para promover ferramentas que auxiliem a equipe médica. A limitação de tais modelos pode levar a diagnósticos imprecisos, e comprometer a qualidade do atendimento a pacientes [[6]](#6). Com as redes generativas é possível criar dados de forma a compensar essa escassez, potencialmente aprimorando a performance de modelos treinados com o suporte desses dados sintéticos.

## Objetivo
Este projeto visa gerar imagens sintéticas de tomografia computadorizada (CT) da região torácica. A priori, o modelo generativo proposto terá como saída imagens em duas dimensões (2D) de CT da região do tórax. Busca-se um grau de realismo suficiente para auxiliar redes de segmentação de vias aéreas. 
Além disso, este trabalho também serve como uma primeira etapa de um projeto maior e mais ambicioso, no qual buscar-se-á a geração de volumes (3D) de tomografias pulmonares, uma combinação de imagens que juntas formam o equivalente a um exame real.

## Metodologia
### Materiais de Referência
Este projeto usará como inspiração inicial o trabalho desenvolvido em [[1]](#1), o qual propõe duas arquiteturas baseadas em GANs para a síntese de imagens CT pulmonares a partir de máscaras binárias que segmentam a região pulmonar. Das arquiteturas propostas, inspirar-se-á na arquitetura Pix2Pix, na qual o gerador é composto de um encoder que aumenta a profundidade da imagem enquanto diminui suas dimensões, seguido de um decoder que realiza o processo oposto. Tal arquitetura também utiliza conexões residuais. Na arquitetura Pix2Pix, o discriminador é composto por cinco camadas convolucionais, onde as quatro primeiras são seguidas por uma camada de ativação *LeakyReLu*, enquanto a última é seguida de uma função *sigmoide*. 

Além do artigo [[1]](#1), também serão considerados os trabalhos realizados em [[3]](#3) e [[4]](#4). No primeiro, desenvolveu-se uma GAN condicional para a geração de imagens CT pulmonares a partir de imagens de ressonância magnética. Já no segundo, utiliza-se um modelo baseado em GAN para a segmentação do pulmão em imagens CT que contém anomalias no tecido pulmonar. Apesar dos objetivos de tais trabalhos não serem os mesmos objetivos propostos para o presente projeto, eles servirão de apoio para proposição de modificações na arquitetura, estratégias de treino e de validação de resultados.   

### Modelo Proposto

> TODO: Atualizar arquiteturas + descrever melhor a loss

Conforme já discutido na seção anterior, após um estudo de outros artigos correlatos ao nosso projeto, verificamos que a estratégia predominante para a síntese de CTs pulmonares e conversão imagem para imagem corresponde a aplicação de GANs (redes adversárias generativas).
Em uma GAN, temos uma rede neural "geradora", responsável por sintetizar as distribuições de entrada e retornar saídas similares aos dados reais, e uma rede neural "discriminadora", que deve ser capaz de classificar corretamente suas entradas como "reais" ou "falsas". Com isso, uma boa rede generativa deve ser capaz de enganar o discriminador, ao passo que um bom discriminador deve identificar corretamente os dados sintéticos em meio aos dados reais.

No caso específico da nossa aplicação, utilizaremos como referência principal as arquiteturas propostas em [[1]](#1). Neste trabalho, uma rede Pix2Pix é utilizada pelo gerador, recebendo uma máscara binária com o formato de um pulmão em um CT e retornando esta imagem 2D preenchida com as vias aéras de um pulmão. Já a rede discriminadora segue a arquitetura 30 × 30 PatchGAN. Ambas estas estruturas foram inicialmente recomendadas por [[8]](#8).
As duas imagens abaixo ilustram as arquiteturas do gerador e discriminador, respectivamente.

![Arquitetura Pix2Pix proposta para gerador.](figs/arquitetura_gen.png?raw=true)

*Figura 1: Arquitetura Pix2Pix proposta para gerador.*

![Arquitetura PatchGAN proposta para discriminador.](figs/arquitetura_disc.png?raw=true)

*Figura 2: Arquitetura PatchGAN proposta para discriminador.*

A função de *loss* aplica um critério similar à *Binary Cross Entropy*, com regularização por MAE (*Mean Absolute Error*), conforme a seguinte a equação matemática:

$$arg\ min_{𝐺}\ max_{𝐷}\ E_{𝑥,𝑦}[log 𝐷(𝑥, 𝑦)] + E_{𝑥,𝑧}[log(1 − 𝐷(𝑥, 𝐺(𝑥, 𝑧)))] + 𝜆E_{𝑥,𝑦,𝑧}[‖𝑦 − 𝐺(𝑥, 𝑧)‖_{1}]$$

> Idealmente, deseja-se que a função de *loss* do gerador e a do discriminador encontrem um equilíbrio em torno de 0.5 (referência Goodfellow)

### Bases de Dados e Evolução
Apesar de inspirar-se no artigo [[1]](#1), o desenvolvimento deste projeto utilizará a base de dados ATM'22, cuja descrição está na tabela abaixo. Tal base de dados não foi usada no desenvolvimento do projeto em [[1]](#1), mas foi escolhida no presente projeto devido a sua amplitude, a presença de dados volumétricos e em razão das imagens possuírem a delimitação das vias aéreas obtidas através de especialistas. Os volumes da base ATM'22 foram adquiridos em diferentes clínicas e considerando diferentes contextos clínicos. Construída para a realização de um desafio de segmentação automática de vias aéria utilizando IA, a base de dados está dividida em 300 volumes para treino, 50 para validação e 150 para teste.

|Base de Dados | Endereço na Web | Resumo descritivo|
|----- | ----- | -----|
|ATM'22 | https://zenodo.org/records/6590774 e https://zenodo.org/records/6590775  | Esta base contém 500 volumes CTs pulmonares, nos quais as vias aéreas estão completamente anotadas, i.e., delimitadas. Tais volumes serão fatiados em imagens 2-D, segmentados e transformados. Esta base de dados foi utilizada para um desafio de segmentação [[2]](#2).|

Os dados desta base são arquivos com extensão `*.nii.gz`, em um formato característico de imagens médicas, e contêm todo o volume pulmonar obtido durante um exame de tomografia. Cada arquivo com um volume pulmonar é acompanhado por um outro arquivo de mesma extensão contendo as anotações feitas por especialistas.
Tais dados são lidos com auxílio da biblioteca `SimpleITK`, conforme feito pelas classes em `datasets.py` neste repositório.

Dado que este trabalho centrará-se na geração de imagens sintéticas em duas dimensões de CTs pulmonares, estes volumes pulmonares serão fatiados no eixo transversal, assim como ilustrado na imagem abaixo. Como resultado, fatiaremos os 500 volumes pulmores em uma quantidade muito maior de imagens 2D, aumentando o tamanho dos conjuntos de dados disponíveis para treinamento, validação e testes.

![Exemplo de fatia de CT pulmonar obtida a partir da base de dados ATM'22.](figs/dataset_exemplo_fatia.png?raw=true)

*Figura 3: Exemplo de fatia de CT pulmonar obtida a partir da base de dados ATM'22.*

A quantia exata de dados que serão utilizados depende da configuração da fatia obtida. Isto é, não serão utilizadas todas as fatias do volume pulmonar, mas sim apenas as imagens que apresentarem o pulmão completo e cercado por tecidos. A partir desta condição, as fatias serão selecionadas e utilizadas como entrada da rede geradora. Ressalta-se que esta seleção é necessária, uma vez que é uma restrição da biblioteca em Python `lungmask` [[7]](#7), utilizada para segmentação automática de CTs pulmonares.
Também é pertinente destacar que esta segmentação é uma etapa essencial do workflow, posto que os dados de entrada da rede geradora da GAN serão máscaras pulmonares, tal como feito em [[1]](#1).

O gráfico abaixo ilustra o histograma da base de dados após a seleção das fatias. Para a construção deste histograma, calculou-se a quantidade de pixels de cada imagem que descrevem a região pulmonar (a parte em branco após a máscara de segmentação). Nota-se que temos muitas imagens com até 2 mil pixels para compor o pulmão, depois temos uma queda nesta quantidade de imagens até algo em torno de 20 mil pixels, seguido por uma nova região de máximo - temos a maior concentração das imagens usadas pela rede generativa com o pulmão ocupando entre 30 e 40 mil pixels. Depois disso, a quantidade exemplares com mais pixels vai diminuindo gradualmente até pouco mais de 100 mil pixels.
Um ponto importante a ser mencionado é que apesar do histograma começar em zero, a menor quantia de pixels no conjunto após segmentação é de 100 pixels. Ademais, dado que são imagens com dimensão 512 x 512 e, portanto, têm mais de 260 mil pixels, as imagens com a maior quantidade de pixels para a região do pulmão não ocupam nem metade de todos os pixels disponíveis.

![Histrograma da quantidade de pixels das fatias selcionadas após segmentação das CTS pulmonares da base de dados ATM'22.](figs/histograma_fatias.png?raw=true)

*Figura 4: Histrograma da quantidade de pixels das fatias selcionadas após segmentação das CTS pulmonares da base de dados ATM'22.*

A figura abaixo apresenta exemplos de fatias em regiões distintas deste histograma para podermos visualizar a variabilidade dos dados de entrada da rede.
Nota-se que as fatias com menos de 10 mil pixels para descrever o pulmão praticamente não têm região suficiente para ser preenchida com vias aéreas, ao passo que as imagens com mais pixels para a região do pulmão são aquelas mais próximas de uma fatia no meio do pulmão, exibindo a maior área util deste órgão.
Com base nestas análises, descarta-se as imagens com menos 25 mil pixels para o pulmão, realizando uma segunda etapa de filtragem da base de dados.

![Exemplos de fatias das CTS pulmonares da base de dados ATM'22 segmentadas.](figs/exemplos_pixels.png?raw=true)

*Figura 5: Exemplos de fatias das CTS pulmonares da base de dados ATM'22 segmentadas.*

Deste modo, ao selecionar apenas as imagens com mais de 25 mil pixels com a região do pulmão, conseguimos construir uma base de dados com pouco mais de 90 mil figuras. Tais imagens devem, então, ser divididas em conjuntos de treinamento, validação e testes.
Para facilitar o cálculo desta separação, opta-se por fixar a base de dados em 90 mil amostras.

Considerando que a parte de testes do modelo envolve uma etapa de teste de utilidade, em que uma rede de segmentação será treinada e avaliada, o tamanho do conjunto de testes total não pode ser pequeno demais.
Além disso, para uma comparação mais justa com o nosso artigo de referência, optamos por utilizar a mesma quantidade de dados de teste para cálculo das demais métricas sobre a qualidade da GAN obtida (FID e SSIM), de maneira a fixar 7 mil dados para cálculo destas métricas no conjunto de testes.
Em função disso, levando em consideração os testes qualitativos, quantitativos e de utilidade, separa-se cerca de um quarto de toda a base de dados para todos os testes.

Ademais, para evitar um conjunto de validação da GAN maior do que o conjunto para obtenção das métricas do modelo (FID e SSIM), opta-se por manter a mesma quantidade de dados para esta tarefa, isto é, 7 mil dados.
Com isso, em uma visão geral, separa-se dois terços (cerca de 66,7%) da base de dados completa para o treinamento da GAN, 7,8% para a validação da GAN e 25,6% para todos os testes (incluíndo a análise qualitativa, análise quantitativa e o teste de utilidade).
Uma representação gráfica desta separação está ilustrada na figura abaixo.

![Separação da base de dados completa em conjuntos de treinamento, validação e testes. Visão geral desta separação, em porcentagem.](figs/Dados_porcentagem.png?raw=true)

*Figura 6: Separação da base de dados completa em conjuntos de treinamento, validação e testes. Visão geral desta separação, em pocentagem.*

Desconsiderando os testes de utilidade e focando apenas nos testes qualitativos e quantitativos, temos 60 mil dados para treinamento da GAN, 7 mil para validação e 7 mil para testes. Isso representa um proporção próxima a 80-10-10, uma das mais clássicas na literatura para treinamento de redes neurais.
A figura abaixo ilustra graficamente esta proporção de dados.

![Separação da base de dados em conjuntos de treinamento, validação e testes, considerando apenas os testes qualitativos e quantitativos para avaliação da GAN.](figs/Dados_GAN.png?raw=true)

*Figura 7: Separação da base de dados em conjuntos de treinamento, validação e testes, considerando apenas os testes qualitativos e quantitativos para avaliação da GAN.*

Por sua vez, considerando apenas o teste de utilidade, teríamos 16 mil dados disponíveis para o treinamento, validação e teste destes modelos de segmentação de vias aéreas (23 mil dados do conjuto de testes total menos 7 mil dados do conjunto de testes qualitativos e quantitativos da GAN).
Todavia, considerando que o conjunto de testes da GAN para as outras métricas não tem relação com os dados da rede de segmentação, podemos reaproveitar este conjunto para obtenção das métricas do teste de utilidade.
Com isso, restam 16 mil dados para serem divididos em conjuntos de treinamento e validação das redes de segmentação. Opta-se por uma quantidade maior de dados de treinamento, considerando o tamanho deste conjunto total, de maneira a selecionar 14 mil dados para treinamento e 2 mil para validação.
Com isso, para a rede de segmentação, teríamos uma proporção de conjuntos próxima a 60-10-30, o que também é bem comum na literatura e é a proporção utilizada no desafio de segmentação da base ATM'22 [[2]](#2).
A figura abaixo ilustra a divisão deste conjunto.

![Separação da base de dados em conjuntos de treinamento, validação e testes, considerando apenas o teste de utilidade (rede neural para segmentação das vias aéreas pulmonares).](figs/Dados_seg.png?raw=true)

*Figura 8: Separação da base de dados em conjuntos de treinamento, validação e testes, considerando apenas o teste de utilidade (rede neural para segmentação das vias aéreas pulmonares).*

Em suma, a separação da base de dados completa em conjuntos de treinamento para GAN, validação da GAN, treinamento da rede de segmentação, validação da rede de segmentação e testes da GAN e da rede de segmentação está ilustrada na figura abaixo. Nota-se que, devido ao reaproveitamento do conjunto de testes entre a GAN e a rede de segmentação, temos ao final cinco (5) conjuntos na saída desta etapa.

![Separação da base de dados completa em conjuntos de treinamento, validação e testes. Visão geral desta separação.](figs/Dados_visao_geral.png?raw=true)

*Figura 9: Separação da base de dados completa em conjuntos de treinamento, validação e testes. Visão geral desta separação.*

Por fim, ressalta-se que além da segmentação dos dados e seleção das fatias, a base de dados também passa pelas etapas de normalização e de transformação para `numpy arrays`, antes de ser utilizada pelas GANs implementadas neste projeto.
A figura abaixo resume esta etapa de tratamento dos dados por completo.

![Fluxograma para processamento da base de dados.](figs/Fluxo_proc_dados.png?raw=true)

*Figura 10: Fluxograma para processamento da base de dados.*

### Workflow

> TODO: Incluir mais detalhes da metodologia

Em uma perspectiva geral do projeto, a metodologia se divide em três grandes estágios:
1. Preparação da base de dados;
2. Treinamento e fine-tunning de modelos de síntese;
3. Avaliação dos modelos gerados.

No que diz respeito à preparação da base de dados, aplica-se o fluxo descrito na Figura 10, da seção anterior, na qual os dados são obtidos de uma fonte pública, processados e separados em conjuntos de treinamento, validação cruzada e testes. A saída desta etapa são 90 mil trios (fatia da CT pulmonar, segmentação feita por especialistas e máscara binária da região do pulmão), com dimensão 1 x 512 x 512 cada.

Quanto a segunda etapa, implementa-se a arquitetura de uma GAN, descrita na seção [Modelo Proposto](#modelo-proposto), que foi concebida tomando como base o artigo [[1]](#1). Sob esta arquitetura, realiza-se uma busca pelos parâmetros ótimos de treinamento da rede conforme a tabela abaixo, a fim de encontrar a melhor combinação para gerar imagens sintéticas de CTs pulmonares mais realistas.
Esta varredura inicial é feita com apenas 10 mil dados e analisada no conjunto de testes de maneira qualitativa (análise subjetiva dos alunos quanto aos resultados) e quantitativa (cálculo das métricas FID e SSIM).
A partir desta análise inicial, seleciona-se três modelos para prosseguir com o treinamento com todos os dados disponíveis.

|Parâmetros | Possibilidades |
|----- | ----- |
|Passos Disc | 1 a 10 |
|Passos Gen | 1 a 10 |
|Tipo de ruído | [Uniforme, Gaussiano] |
|Localização do ruído | Na imagem completa ou apenas na região do pulmão|
|Regularização | [0, 10] |
|Beta |Entre 1989 e 2000 |
|blabla | blabla |

Dadas as restrições de tempo e capacidade computacional, não foram testadas todas as combinações de parâmetros da tabela acima. Com apoio da ferramenta Weights & Biases, combinou-se aleatoriamente estes parâmetros em quinze modelos, descritos na tabela abaixo.
A configuração destes parâmetros é feita em um arquivo YAML.

[jogar para resultados -----------------]

|Modelo |	Relação Passos (Disc/Gen) |	Ruído |	Ruído só no pulmão|	Intensidade	|Média Ruído	|Desvio Ruído	|Criterion	|Regularizador	|Nível Regularização	|Learning Rate	|Beta|
| ----- | ----- | -----   | ----- | -----       | -----         | -----         |   -----  | ----- | -----| -----   |   -----       |
|Sweep10|	4/2	|Gaussiano|	Falso |	0,3157719473|	0,7469069764|	0,1784581512|	BCELoss|	MSE|	8|	3,11E-04|	0,4597517629|
|Sweep205|	3/1	|Gaussiano|	Verdadeiro|	0,5566831094|	0,5120044953|	0,3903814624|	MSELoss|	MAE|	10|	2,85E-04|	0,7555202559|
|Sweep412|	1/1| Gaussiano|	Falso| 0,757255249|	0,5250495573|	0,4755411392|	MSELoss|	MAE|	4	|1,70E-04	|0,8811316699|
|Sweep64	|1/2	|Gaussiano	|Verdadeiro	|0,81851453	|0,5597838196	|0,2229110595	|MSELoss	|MAE	|3	|3,75E-04	|0,8659691523|
|Sweep123	|2/1	|Gaussiano	|Verdadeiro	|0,3320755603	|0,652635058	|0,3347731658	|MSELoss	|MAE	|4	|1,55E-04	|0,6252443893|
|Sweep284	|1/2	|Gaussiano	|Verdadeiro	|0,4882098594	|0,872090533	|0,4466720449	|MSELoss	|MSE	|4	|2,24E-04	|0,6781061686|
|Sweep394	|2/1	|Gaussiano	|Falso	|0,3715918515	|0,6996284578	|0,2871496533	|BCELoss	|MAE	|1	|3,40E-04	|0,4792751887|
|Sweep497	|1/1	|Gaussiano	|Verdadeiro	|0,3039449554	|0,8749711247	|0,2897599163	|MSELoss	|MSE	|15	|1,32E-04	|0,840671948|
|Sweep522	|4/2	|Gaussiano	|Falso	|0,8766142328	|0,6935412609	|0,3790460335	|MSELoss	|MSE_mask	|13	|3,40E-04	|0,5728743005|
|Sweep71	|2/1	|Gaussiano	|Verdadeiro	|0,8172635438	|0,548984276	|0,3265456309	|BCELoss	|MSE_mask	|1	|2,82E-04	|0,52631016|
|Sweep185	|4/1	|Uniforme	|Verdadeiro	|0,3563791549|	0,5899638112|	0,2158650277|	MSELoss|	MAE_mask|	5|	2,82E-04|	0,4240341338|
|Sweep186	|2/1	|Uniforme	|Verdadeiro	|0,9795390854|	0,5310213915	|0,2623582226	|BCELoss	|MAE_mask	|4	|1,87E-04	|0,6069949071|
|Sweep256	|1/2	|Gaussiano	|Verdadeiro	|0,3085178607	|0,6810390549	|0,1347611367	|MSELoss	|MAE_mask	|8	|3,16E-04	|0,4703302188|
|Sweeo279	|4/2	|Gaussiano	|Falso	|0,6821396703	|0,9681958035	|0,1024100341	|MSELoss	|MAE_mask	|15	|2,58E-04	|0,6470046351|
|Sweep464	|2/2	|Gaussiano	|Verdadeiro	|0,9864110063	|0,9929413808	|0,1007233152	|MSELoss	|MSE_mask	|1	|2,91E-04	|0,4393293661|

Após esta etapa, passa-se os três melhores modelos para a etapa de avaliação de desempenho e qualidade dos resultados. Gera-se imagens sintéticas a partir de máscaras binárias de CTs pulmonares com ruído e realiza-se três testes: qualitativo, quantitativo e de utilidade. Tais testes serão descritos em mais detalhes na seção [Métricas de Avaliação](#métricas-de-avaliação).

Um ponto importante a ser destacado é a diferença do tipo de ruído aplicado na máscara de entrada do gerador. Como é possível observar nas tabelas acima, duas distribuições foram testadas: uniforme e gaussiana. xxxxxx

Em suma, o fluxo de trabalho proposto por este projeto, ilustrado na figura a seguir, inicia-se com a obtenção da base de dados ATM'22 e seu devido tratamento, conforme detalhado na seção anterior.
Utilizando estes dados, alimenta-se a rede generativa com as fatias segmentadas (máscaras binárias). Já a rede discriminadora recebe os dados reais (sem segmentação) e os dados sintéticos, devendo classificar cada um como "real" ou "falso".
Após o treinamento, avalia-se os dados sintéticos a partir de três perspectivas: análise qualitativa, análise quantitativa e análise de utilidade, as quais serão descritas em detalhes nas próximas seções deste relatório.

![Fluxo para treinamento da PulmoNet.](figs/workflow_completo.png?raw=true)

*Figura 11: Fluxo para treinamento da PulmoNet.*

Destaca-se que, em operação (após a fase treinamento), espera-se que o modelo receba máscaras binárias com o formato do pulmão somadas a um ruído, retonando o preenchimento da área interna do pulmão.
Uma mesma máscara binária poderá gerar imagens sintéticas distintas, devido ao ruído aleatório adicionado na entrada do modelo.
Os dados sintéticos deverão ser bons o suficiente para ajudarem no treinamento de modelo de segmentação das vias aéreas e potencialmente substituir o uso de dados reais, para a preservação da privacidade dos pacientes.

### Ferramentas Relevantes
A ferramenta escolhida para o desenvolvimento da arquitetura dos modelos e de treinamento é o **PyTorch**, em função de sua relevância na área e familiaridade por parte dos integrantes do grupo.
Ademais, para o desenvolvimento inicial e colaborativo dos modelos entre os estudantes, opta-se pela ferramenta de programação **Google Collaboratory**.
Já para o versionamento dos modelos e para ajustar seus hiperparâmetros, decidiu-se pela ferramenta **Weights & Biases (Wandb AI)** dentre as opções disponíveis no mercado. E, além disso, a ferramenta do **GitHub** também auxiliará no versionamento dos algoritmos desenvolvidos.

### Métricas de Avaliação
Para avaliar a qualidade dos resultados obtidos com o modelo de síntese, propõe-se três tipos de avaliação: análise qualitativa, análise quantitativa e análise de utilidade.

#### Análise Qualitativa
Esta estratégia será utilizada apenas nas etapas iniciais do desenvolvimento do projeto, na qual os próprios estudantes irão observar os resultados sintéticos, sejam eles imagens e/ou  volumes, e compararão com os dados reais esperados. Com isto, faz-se uma avaliação se a imagem gerada estaria muito distante de uma CT pulmonar ou se o modelo já estaria se encaminhando para bons resultados. Após esta etapa, as avaliações do modelo serão feitas por meio das análises quantitativa e de utiliddade.

#### Análise Quantitativa
A análise quantitativa trata de uma avaliação sobre as imagens a partir dos métodos **Fréchet Inception Distance (FID)** e **Structural Similarity Index (SSIM)**, os quais são utilizados para avaliação de qualidade das imagens sintéticas e de similaridade com dados reais. Ambas estratégias foram utilizadas pelos pesquisadores do artigo [[1]](#1), o que permite uma avaliação dos nossos resultados frente a esta outra pesquisa.

Entrando em mais detalhes, a métrica FID avalia o desempenho da rede generativa e será calculada utilizando uma rede neural pré-treinada *InceptionV3*, que extrairá *features* das fatias pulmonares geradas e das fatias originais. Com isso, as distribuições dos dados sintéticos e dos dados reais, obtidas pelo encoder desta rede, são usadas para calcular a FID e, assim, avaliar a qualidade da imagem gerada.
A expressão matemática que descreve o cálculo da FID entre duas distribuições gaussianas criadas pelas *features* da última camada de *pooling* do modelo *Inception-v3* é dada por:

$$FID = ‖𝜇_{𝑟} − 𝜇_{𝑔}‖^{2} + Tr(\sum_{𝑟} + \sum_{𝑔} − 2(\sum_{𝑟}\sum_{𝑔})^{1∕2})$$

onde $𝜇_{𝑟}$ e $𝜇_{𝑔}$ são as médias entre as imagens reais e sintéticas, e $\sum_{𝑟},\ \sum_{𝑔}$ são as matrizes de convariância para os vetores de *features* dos dados reais e gerados, respectivamente.
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
Dado que o objetivo do projeto é gerar imagens sintéticas (2D) de CTs pulmonares realistas, avalia-se nesta etapa duas perspectivas. A primeira delas trata da segmentação das fatias sintéticas por meio da biblioteca *lungmask* e comparação desta saída com a máscara binária original que gerou esta imagem sintética. Isto é feito para avaliar se o gerador conseguiu manter o formato do pulmão original ou algo próximo a isso. Utiliza-se o SSIM para comparação destas duas fatias pulmonares segmentadas.

Já a segunda perspectiva trata da utilidade do gerador, em termos de **feature extraction**. Isto é, tomando como inspiração a abordagem explorada em [[9]](#9), implementaremos uma U-Net, com a mesma estrutura da rede geradora Pix2Pix da PulmoNet, para realizar a segmentação das vias aéreas e compararemos o desempenho desta U-Net com uma outra rede que utiliza as *features* extraídas pelo nosso gerador. Esta comparação será avaliada ao comparar as saídas com a própria segmentação presente na base de dados ATM'22, feita por especialistas. Além disso, será calculado o coeficiente DICE (obtido a partir da precisão e *recall* da predição), tomando como referência o artigo [[2]](#2), e considera-se também calcular o tempo de processamento das redes U-Net e U-Net com *features* extraídos pela nossa Pix2Pix, a fim de verificar se também há uma otimização neste quesito.

Ressalta-se que foram escolhidas duas funções de *loss* para esta tarefa: BCEWithLogitsLoss e DICELoss, tipicamente utilizadas em tarefas de segmentação de imagens médicas.
Além disso, para aproveitar os pesos iniciais da GAN para a tarefa de segmentação, são feitas três variações no processo de *fine-tunning*:
1. Retreina-se todos os pesos da arquitetura, utilizando o conhecimento adquirido pela GAN apenas como uma inicialização não aleatória para o treinamento da rede de segmentação;
2. Congela-se apenas a parte da rede codificadora do gerador, retreinando somente o decodificador;
3. Congela-se todas as camadas do gerador, com excessão da última camada.

Por fim, é importante destacar o caminho a ser seguido para a avaliação da rede generativa para as saídas em 3D, caso seja possível implementá-las dentro do prazo do projeto. Para esta aplicação, geraríamos um volume sintético e passaríamos esta saída pela rede de segmentação *medpseg* [[10]](#10). Feito isso, compararíamos as vias aéreas segmentadas com o *ground-truth* estabelecido na própria base de dados ATM'22.

### Cronograma
O projeto será implementado seguindo o seguinte fluxo lógico:

![Fluxo lógico das ativaidades para desenvolvimento da PulmoNet.](figs/fluxo_logico.png?raw=true)

*Figura 12: Fluxo lógico das ativaidades para desenvolvimento da PulmoNet.*

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
> TODO: Falar sobre a máquina usada para treinar a GAN (quantidade de memória, tipo de GPU etc) e para treinar a rede de segmentação

Os modelos da GAN foram treinados em uma máquina com uma GPU NVIDIA GeForce RTX 3060.
Já o modelo da rede de segmentação, para o teste de utilidade, foi treinado em um computador pessoal que tinha uma GPU 	NVIDIA GeForce RTX 3050, 4G de memória de GPU, 16G de memória RAM e processador Intel I5 de 12ª geração.


## Experimentos, Resultados e Discussão dos Resultados
> TODO: Atualizar com dados da E3

Para a entrega parcial do projeto (E2), já foi feito um estudo de artigos na literatura no contexto do nosso projeto. Além disso, seguindo o cronograma do projeto, também foi finalizada a etapa de análise da base de dados e a definição das etapas de pré-processamento, conforme já discutido brevemente na seção sobre a base de dados. Mais ainda, também foi realizada a implementação da arquitetura inicial das GANs escolhidas para o projeto, tomando como base o desenvolvimento em [[1]](#1), e iniciou-se a etapa de treinamento deste modelo.

Atualmente, estamos enfrentando dificuldades nesta etapa de treinamento, já que notamos que o discriminador estava ficando muito bom rápido demais, não permitindo que o gerador conseguisse avançar em seu aprendizado. Para solucionar este problema, tentaremos usar a estratégia de atualizar a *loss* do gerador com mais frequência do que a do discriminador (a priori, atualizaremos a loss do discriminador a cada 3 batches de atualização da loss do gerador).

O resultado atual do nosso treinamento é apresentado na figura abaixo. Nota-se que a saída do gerador ainda está distante do esperado e precisa ser aprimorada.

![Fatia original, fatia segmentada, saída do gerador e saída do discriminador.](figs/resultado_parcial_e2.jpeg?raw=true)

*Figura 13: Fatia original, fatia segmentada, saída do gerador e saída do discriminador.*

Ademais outros problemas que estamos enfrentando durante a etapa do treinamento tratam do tamanho da nossa base de dados, que é bem grande e resulta em um processamento demorado, e o uso de recursos em GPU.

## Conclusão
> TODO: Atualizar com dados da E3

O projeto da rede PulmoNet busca a geração de fatias de CTs pulmonares a partir de máscaras binárias, em duas dimensões, baseada em GANs. Esta rede utiliza uma arquitetura Pix2Pix para o gerador e uma PatchGAN para o discriminador. São usados dados da base pública ATM'22, cujos dados correspondem a volumes pulmonares de tomografias e segmentações das vias aéreas feitas por especialistas. Para a avaliação da qualidade da rede, propõe-se métodos qualitativos, quantitativos e análises de utilidade.

Seguindo o cronograma do projeto, as etapas até a entrega E2 foram cumpridas, de maneira que estamos atualmente na fase de treinamento do modelo e implementação dos métodos de avaliação. No caso do treinamento, estamos enfrentando algumas dificuldades que estão afetando a qualidade das saídas da rede, principalmente no quesito da velocidade de aprendizado do discriminador frente a do gerador.

Os próximos passos do projeto tratam da finalização do treinamento do modelo, análise das métricas de avaliação e fine-tunning e aperfeiçoamento do modelo. Caso tenhamos tempo disponível, buscaremos a geração de volumes 3D de CTs pulmonares.

### Próximos Passos
> TODO

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

Documento com as referências extras identificadas: https://docs.google.com/document/d/1uatPj6byVIEVrvMuvbII6J6-5usOjf8RLrSxLHJ8u58/edit?usp=sharing

# How To Run
Como uma observação adicional, incluimos uma descrição de como executar as funções propostas neste projeto.

**Processamento da base de dados:**

`1.` Baixar a base de dados ATM'22 na internet

`2.` Fazer a leitura inicial dos dados por meio da classe `rawCTData`


**Treinamento da GAN:**

1. Configurar parâmetros do modelo no arquivo `config.yaml` e a localização da pasta com os dados processados.

2. Executar comando em seu terminal:

```
py training_pipeline.py
```

3. Selecionar o arquivo YAML de configuração desejado:

'''
 config.yaml
'''

**Obtenção das métricas da GAN:**

`1.` Configurar parâmetros do modelo no arquivo `config_eval.yaml` e a localização da pasta com os dados processados.

`2.` Executar comando em seu terminal:

```
test_pipeline.py config_eval.yaml
```

**Treinamento da rede de segmentação:**

`1.` Configurar parâmetros do modelo no arquivo `config_segmentation.yaml` e a localização da pasta com os dados processados.

`2.` Executar comando em seu terminal:

```
segmentation_pipeline.py config_segmentation.yaml
```
