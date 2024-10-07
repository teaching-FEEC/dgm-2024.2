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

## Resumo (Abstract)
As tomografias computadorizadas (CT) pulmonares e a segmentação das vias aéreas desempenham um papel crucial no diagnóstico preciso de doenças pulmonares. 
Propõe-se o desenvolvimento da PulmoNet, uma rede para síntese de imagens 2D de CTs pulmonares, com o intuito de apoiar redes de segmentação e gerar dados sintéticos para incorporação em bases de dados para outras redes neurais (e.g. classificadores de tumores).
Utilizando a base de dados ATM'22, implementa-se uma arquitetura GAN, sendo o gerador uma rede Pix2Pix e o discriminador uma PatchGAN, que receberá uma máscara binária do pulmão e preencherá esta fatia com as vias aéreas.
Tal rede será avaliada em três análises: qualitativa (observação dos resultados no estágio inicial do projeto),  quantitativa (métricas FID e SSIM) e utilidade (aplicação do gerador como *feature extractor*).
Os resultados parciais até o momento não foram bem-sucedidos, uma vez que se enfrenta problemas no treinamento, principalmente com relação a velocidade de aprendizado do discriminador comparada ao do gerador.

## Descrição do Problema/Motivação
As tomografias computadorizadas (CT) pulmonares, juntamente com a segmentação das vias aéreas, desempenham um papel crucial no diagnóstico preciso de doenças pulmonares. Ao gerar imagens detalhadas da região torácica, ela permite que médicos mapeiem a anatomia das vias aéreas antes de procedimentos cirúrgicos, avaliando a extensão de lesões e facilitando o acompanhamento da progressão de doenças respiratórias [[2]](#2). Além disso, a CT é fundamental para monitorar a eficácia de tratamentos e detectar seus possíveis efeitos colaterais [[5]](#5).

A complexidade e diversidade do corpo humano dificultam a obtenção de grandes volumes de dados médicos para treinar modelos de aprendizado de máquina, como as redes neurais. Essa escassez de dados pode levar a diagnósticos imprecisos, comprometendo a qualidade do atendimento aos pacientes [[6]](#6). Com as redes generativas é possível criar dados de forma a compensar essa escassez, permitindo que as redes aprendam muito mais detalhes do que utilizando apenas aqueles obtidos de exames reais.

[Link para o vídeo de apresentação E1](https://drive.google.com/file/d/1TlpQOlCh_lAI0-jPPMPWOzGZ_werCo3d/view?usp=sharing)

[Link para a apresentação de slides E1](https://docs.google.com/presentation/d/1b8W0Cw1eiTbWlJ0CJJ8eMRA4zyu2iLhYvggi55-mOb0/edit?usp=sharing)

[Link para a apresentação de slides E2](https://docs.google.com/presentation/d/1QH5_WpeTp7kQPSVB78ukK7msn-Tx09pZoM_3dWmeqC4/edit?usp=sharing)

## Objetivo
Este projeto visa gerar imagens sintéticas de tomografia computadorizada (CT) da região torácica de alta fidelidade, também produzindo máscaras de segmentação das vias aéreas. A priori, o modelo generativo proposto terá como saída imagens em duas dimensões (2D) de CT da região do tórax, com grau de realismo suficiente e que possa auxiliar redes de segmentação de vias aéreas. 
Além disso, este trabalho também serve como uma primeira etapa de um projeto maior e mais ambicioso, no qual buscar-se-á a geração de volumes (imagens 3D) de tomografias pulmonares, uma combinação de fatias que juntas formarão o equivalente a um exame real.

## Metodologia
### Materiais de Referência
Este projeto usará como inspiração inicial o trabalho desenvolvido em [[1]](#1), o qual propõe duas arquiteturas baseadas em GANs para a síntese de imagens CT pulmonares a partir de máscaras binárias que segmentam a região pulmonar. Das arquiteturas propostas, inspirar-se-á na arquitetura Pix2Pix, na qual o gerador é composto de um encoder que aumenta a profundidade da imagem enquanto diminui suas dimensões, seguido de um decoder que realiza o processo oposto. Tal arquitetura também utiliza conexões residuais. Na arquitetura Pix2Pix, o discriminador é composto por cinco camadas convolucionais, onde as quatro primeiras são seguidas por uma camada de ativação *LeakyReLu*, enquanto a última é seguida de uma função *sigmoide*. 

Além do artigo [[1]](#1), também serão considerados os trabalhos realizados em [[3]](#3) e [[4]](#4). No primeiro, desenvolveu-se uma GAN condicional para a geração de imagens CT pulmonares a partir de imagens de ressonância magnética. Já no segundo, utiliza-se um modelo baseado em GAN para a segmentação do pulmão em imagens CT que contém anomalias no tecido pulmonar. Apesar dos objetivos de tais trabalhos não serem os mesmos objetivos propostos para o presente projeto, eles servirão de apoio para proposição de modificações na arquitetura, estratégias de treino e de validação de resultados.   

### Modelo Proposto
Conforme já discutido na seção anterior, após um estudo de outros artigos correlatos ao nosso projeto, verificamos que a estratégia predominante para a síntese de CTs pulmonares e conversão imagem para imagem corresponde a aplicação de GANs (redes adversárias generativas).
Em uma GAN, temos uma rede neural "geradora", responsável por sintetizar as distribuições de entrada e retornar saídas similares aos dados reais, e uma rede neural "discriminadora", que deve ser capaz de classificar corretamente suas entradas como "reais" ou "falsas". Com isso, uma boa rede generativa deve ser capaz de enganar o discriminador, ao passo que um bom discriminador deve identificar corretamente os dados sintéticos em meio aos dados reais.

No caso específico da nossa aplicação, utilizaremos como referência principal as arquiteturas propostas em [[1]](#1). Neste trabalho, uma rede Pix2Pix é utilizada pelo gerador, recebendo uma máscara binária com o formato de um pulmão em um CT e retornando esta imagem 2D preenchida com as vias aéras de um pulmão. Já a rede discriminadora segue a arquitetura 30 × 30 PatchGAN. Ambas estas estruturas foram inicialmente recomendadas por [[8]](#8).
As duas imagens abaixo ilustram as arquiteturas do gerador e discriminador, respectivamente.

![Arquitetura Pix2Pix proposta para gerador.](figs/arquitetura_gen.png?raw=true)

*Figura 1: Arquitetura Pix2Pix proposta para gerador.*

![Arquitetura PatchGAN proposta para discriminador.](figs/arquitetura_disc.png?raw=true)

*Figura 2: Arquitetura PatchGAN proposta para discriminador.*

A função de loss aplica o critério de *Binary Cross Entropy*, conforme a seguinte a equação matemática:

$arg\ min_{𝐺}\ max_{𝐷}\ E_{𝑥,𝑦}[log 𝐷(𝑥, 𝑦)] + E_{𝑥,𝑧}[log(1 − 𝐷(𝑥, 𝐺(𝑥, 𝑧)))] + 𝜆E_{𝑥,𝑦,𝑧}[‖𝑦 − 𝐺(𝑥, 𝑧)‖_{1}]$

### Bases de Dados e Evolução
Apesar de inspirar-se no artigo [[1]](#1), o desenvolvimento deste projeto utilizará a base de dados ATM'22, cuja descrição está na tabela abaixo. Tal base de dados não foi usada no desenvolvimento do projeto em [[1]](#1), mas foi escolhida no presente projeto devido a sua amplitude, a presença de dados volumétricos e em razão das imagens possuírem a delimitação das vias aéreas obtidas através de especialistas. Os volumes da base ATM'22 foram adquiridos em diferentes clínicas e considerando diferentes contextos clínicos. Construída para a realização de um desafio de segmentação automática de vias aéria utilizando IA, a base de dados está dividida em 300 volumes para treino, 50 para validação e 150 para teste.

|Base de Dados | Endereço na Web | Resumo descritivo|
|----- | ----- | -----|
|ATM'22 | https://zenodo.org/records/6590774 e https://zenodo.org/records/6590775  | Esta base contém 500 volumes CTs pulmonares, nos quais as vias aéreas estão completamente anotadas, i.e., delimitadas. Tais volumes serão fatiados em imagens 2-D, segmentados e transformados. Esta base de dados foi utilizada para um desafio de segmentação [[2]](#2).|

Os dados desta base são arquivos com extensão *.nii.gz, e contêm todo o volume pulmonar obtido durante um exame de tomografia. Cada arquivo com um volume pulmonar é acompanhado por um outro arquivo de mesma extensão contendo as anotações feitas por especialistas.
Dado que este trabalho centrará-se na geração de imagens sintéticas em duas dimensões de CTs pulmonares, estes volumes pulmonares serão fatiados no eixo transversal, assim como ilustrado na imagem abaixo. Como resultado, fatiaremos os 500 volumes pulmores em uma quantidade muito maior de imagens 2D, aumentando o tamanho dos conjuntos de dados disponíveis para treinamento, validação e testes.

![Exemplo de fatia de CT pulmonar obtida a partir da base de dados ATM'22.](figs/dataset_exemplo_fatia.png?raw=true)

*Figura 3: Exemplo de fatia de CT pulmonar obtida a partir da base de dados ATM'22.*

A quantia exata de dados que serão utilizados depende da configuração da fatia obtida. Isto é, não serão utilizadas todas as fatias do volume pulmonar, mas sim apenas as imagens que apresentarem o pulmão completo e cercado por tecidos. A partir desta condição, as fatias serão selecionadas e utilizadas como entrada da rede geradora. Ressalta-se que esta seleção é necessária, uma vez que é uma restrição da biblioteca em Python *lungmask* [[7]](#7), utilizada para segmentação automática de CTs pulmonares.
Também é pertinente destacar que esta segmentação é uma etapa essencial do workflow, posto que os dados de entrada da rede geradora da GAN serão máscaras pulmonares, tal como feito em [[1]](#1).

O gráfico abaixo ilustra o histograma da base de dados após a seleção das fatias. Para a construção deste histograma, calculou-se a quantidade de pixels de cada imagem que descrevem a região pulmonar (a parte em branco após a máscara de segmentação). Nota-se que temos muitas imagens com até 2 mil pixels para compor o pulmão, depois temos uma queda nesta quantidade de imagens até algo em torno de 20 mil pixels, seguido por uma nova região de máximo - temos a maior concentração das imagens usadas pela rede generativa com o pulmão ocupando entre 30 e 40 mil pixels. Depois disso, a quantidade exemplares com mais pixels vai diminuindo gradualmente até pouco mais de 100 mil pixels.
Um ponto importante a ser mencionado é que apesar do histograma começar em zero, a menor quantia de pixels no conjunto após segmentação é de 100 pixels. Ademais, dado que imagens 512 x 512 têm mais de 260 mil pixels, as imagens com a maior quantidade de pixels para a região do pulmão não ocupam nem metade de todos os pixels da imagem.

![Histrograma da quantidade de pixels das fatias selcionadas após segmentação das CTS pulmonares da base de dados ATM'22.](figs/histograma_fatias.png?raw=true)

*Figura 4: Histrograma da quantidade de pixels das fatias selcionadas após segmentação das CTS pulmonares da base de dados ATM'22.*

A figura abaixo apresenta exemplos de fatias em regiões distintas deste histograma para podermos visualizar a variabilidade dos dados de entrada da rede.
Nota-se que as fatias com menos de 10 mil pixels para descrever o pulmão praticamente não têm região suficiente para ser preenchida com vias aéreas, ao passo que as imagens com mais pixels para a região do pulmão são aquelas mais próximas de uma fatia no meio do pulmão, exibindo a maior área util deste órgão.
Com base nestas análises, considera-se descartar imagens com poucos pixels para o pulmão.

![Exemplos de fatias das CTS pulmonares da base de dados ATM'22 segmentadas.](figs/exemplos_pixels.png?raw=true)

*Figura 5: Exemplos de fatias das CTS pulmonares da base de dados ATM'22 segmentadas.*

Além da segmentação dos dados e seleção das fatias, a base de dados também passa pelas etapas de normalização e de transformação para *numpy arrays*, antes de ser utilizada pelas GANs implementadas neste projeto.

### Workflow
O fluxo de trabalho proposto por este projeto, ilustrado na figura a seguir, inicia-se com a obtenção da base de dados ATM'22 e seu devido tratamento, conforme detalhado na seção anterior.
Utilizando estes dados, alimenta-se a rede generativa com as fatias segmentadas (máscaras binárias). Já a rede discriminadora recebe os dados reais (sem segmentação) e os dados sintéticos, devendo classificar cada um como "real" ou "falso".
Após o treinamento, avalia-se os dados sintéticos a partir de três perspectivas: análise qualitativa, análise quantitativa e análise de utilidade, as quais serão descritas em detalhes nas próximas seções deste relatório.

![Fluxo para treinamento da PulmoNet.](figs/workflow_completo.png?raw=true)

*Figura 6: Fluxo para treinamento da PulmoNet.*

Destaca-se que, em operação (após a fase treinamento), espera-se que o modelo receba máscaras binárias com o formato do pulmão somadas a um ruído, retonando o preenchimento da área interna do pulmão.
Uma mesma máscara binária poderá gerar imagens sintéticas distintas, devido ao ruído aleatório adicionado na entrada do modelo.
Os dados sintéticos deverão ser bons o suficiente para ajudarem no treinamento de modelo de segmentação das vias aéreas e potencialmente substituir o uso de dados reais, para a preservação da privacidade dos pacientes.

Ademais, na fase atual do projeto, ainda não estamos somando um ruído aleatório às fatias segmentadas na entrada do gerador, mas este passo está mapeado para as próximas etapas do projeto.

### Ferramentas Relevantes
A ferramenta escolhida para o desenvolvimento da arquitetura dos modelos e de treinamento é o **PyTorch**, em função de sua relevância na área e familiaridade por parte dos integrantes do grupo.
Ademais, para o desenvolvimento colaborativo dos modelos entre os estudantes, opta-se pela ferramenta de programação **Google Collaboratory**.
Já para o versionamento dos modelos e para ajustar seus hiperparâmetros, decidiu-se pela ferramenta **Weights & Biases (Wandb AI)** dentre as opções disponíveis no mercado. E, além disso, a ferramenta do **GitHub** também auxiliará no versionamento dos algoritmos desenvolvidos.

### Métricas de Avaliação
Para avaliar a qualidade dos resultados obtidos com o modelo de síntese, propõe-se três tipos de avaliação: análise qualitativa, análise quantitativa e análise de utilidade.

#### Análise Qualitativa
Esta estratégia será utilizada apenas nas etapas iniciais do desenvolvimento do projeto, na qual os próprios estudantes irão observar os resultados sintéticos, sejam eles imagens e/ou  volumes, e compararão com os dados reais esperados. Com isto, faz-se uma avaliação se a imagem gerada estaria muito distante de uma CT pulmonar ou se o modelo já estaria se encaminhando para bons resultados. Após esta etapa, as avaliações do modelo serão feitas por meio das análises quantitativa e de utiliddade.

#### Análise Quantitativa
A análise quantitativa trata de uma avaliação sobre as imagens a partir dos métodos **Fréchet Inception Distance (FID)** e **Structural Similarity Index (SSIM)**, os quais são utilizados para avaliação de qualidade das imagens sintéticas e de similaridade com dados reais. Ambas estratégias foram utilizadas pelos pesquisadores do artigo [[1]](#1), o que permite uma avaliação dos nossos resultados frente a esta outra pesquisa.

Entrando em mais detalhes, a métrica FID avalia o desempenho da rede generativa e será calculada utilizando uma rede neural pré-treinada *InceptionV3*, que extrairá *features* das fatias pulmonares geradas e das fatias originais. Com isso, as distribuições dos dados sintéticos e dos dados reais, obtidas pelo encoder desta rede, são usadas para calcular a FID e, assim, avaliar a qualidade da imagem gerada.
A expressão matemática que descreve o cálculo da FID entre duas distribuições gaussianas criadas pelas *features* da última camada de *pooling* do modelo *Inception-v3* é dada por:

$FID = ‖𝜇_{𝑟} − 𝜇_{𝑔}‖^{2} + Tr(\sum_{𝑟} + \sum_{𝑔} − 2(\sum_{𝑟}\sum_{𝑔})^{1∕2})$

onde $𝜇_{𝑟}$ e $𝜇_{𝑔}$ são as médias entre as imagens reais e sintéticas, e $\sum_{𝑟},\ \sum_{𝑔}$ são as matrizes de convariância para os vetores de *features* dos dados reais e gerados, respectivamente.
Quanto menor for o FID, maior a qualidade da imagem gerada.

Por sua vez, a métrica SSIM compara a imagem gerada com seu respectivo *ground-truth* com base em três características: luminância, distorção de contraste e perda de correlação estrutural.
Casos as imagens sejam iguais, o resultado desta métrica será igual a 1, ao passo que se as imagens forem completamente diferentes, o SSIM será nulo.
Ressalta-se que não queremos que esta métrica fique em nenhum deste extremos, mas sim em um valor intermediário.
As expressões matemáticas usadas para o cálculo desta métrica são:

$SSIM(𝑥, 𝑦) = l(𝑥, 𝑦) \times 𝑐(𝑥, 𝑦) \times 𝑠(𝑥, 𝑦)$

$l(𝑥, 𝑦) = \frac{2𝜇_{𝑥}𝜇_{𝑦} + 𝐶_{1}}{𝜇^{2}_{𝑥}+ 𝜇^{2}_{𝑦} + 𝐶_{1}}$

$𝑐(𝑥, 𝑦) = \frac{2𝜎_{𝑥}𝜎_{𝑦} + 𝐶_{2}}{𝜎^{2}_{𝑥} + 𝜎^{2}_{𝑦} + 𝐶_{2}}$

$𝑠(𝑥, 𝑦) = \frac{𝜎_{𝑥𝑦} + 𝐶_{3}}{𝜎_{𝑥}𝜎_{𝑦} + 𝐶_{3}}$

onde $𝜇_{𝑥}$, $𝜇_{𝑦}$, $𝜎_{𝑥}$, $𝜎_{𝑦}$, e $𝜎_{𝑥𝑦}$ são as médias locais, variâncias e covariâncias cruzadas para as imagens 𝑥, 𝑦, respectivament. $𝐶_{1}$, $𝐶_{2}$ $𝐶_{3}$ são constantes.

#### Análise de Utilidade
Dado que o objetivo do projeto é gerar imagens sintéticas (2D) de CTs pulmonares realistas, avalia-se nesta etapa duas perspectivas. A primeira delas trata da segmentação das fatias sintéticas por meio da biblioteca *lungmask* e comparação desta saída com a máscara binária original que gerou esta imagem sintética. Isto é feito para avaliar se o gerador conseguiu manter o formato do pulmão original ou algo próximo a isso. Utiliza-se o SSIM para comparação destas duas fatias pulmonares segmentadas.

Já a segunda perspectiva trata da utilidade do gerador, em termos de **feature extraction**. Isto é, tomando como inspiração a abordagem explorada em [[9]](#9), implementaremos uma U-Net, com a mesma estrutura da rede geradora Pix2Pix da PulmoNet, para realizar a segmentação das vias aéreas e compararemos o desempenho desta U-Net com uma outra rede que utiliza as *features* extraídas pelo nosso gerador. Esta comparação será avaliada ao comparar as saídas com a própria segmentação presente na base de dados ATM'22, feita por especialistas. Além disso, será calculado o coeficiente DICE (obtido a partir da precisão e *recall* da predição), tomando como referência o artigo [[2]](#2), e considera-se também calcular o tempo de processamento das redes U-Net e U-Net com *features* extraídos pela nossa Pix2Pix, a fim de verificar se também há uma otimização neste quesito.

Por fim, é importante destacar o caminho a ser seguido para a avaliação da rede generativa para as saídas em 3D, caso seja possível implementá-las dentro do prazo do projeto. Para esta aplicação, utilizaríamos 5 fatias de CTs pulmonares sequenciais, removeríamos a segunda e a quarta fatias e sinetizaríamos esas fatias faltantes. Feito isso, analisaríamos o volume formado em comparação com o volume original. Com isso, seria possível avaliar se a PulmoNet é capaz de gerar imagens relevantes e realistas, além de possibilitar sua implementação no auxílio a **interpolação de CTs pulmonares**.

### Cronograma
O projeto será implementado seguindo o seguinte fluxo lógico:

![Fluxo lógico das ativaidades para desenvolvimento da PulmoNet.](figs/fluxo_logico.png?raw=true)

*Figura 7: Fluxo lógico das ativaidades para desenvolvimento da PulmoNet.*

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



## Experimentos, Resultados e Discussão dos Resultados
Para a entrega parcial do projeto (E2), já foi feito um estudo de artigos na literatura no contexto do nosso projeto. Além disso, seguindo o cronograma do projeto, também foi finalizada a etapa de análise da base de dados e a definição das etapas de pré-processamento, conforme já discutido brevemente na seção sobre a base de dados. Mais ainda, também foi realizada a implementação da arquitetura inicial das GANs escolhidas para o projeto, tomando como base o desenvolvimento em [[1]](#1), e iniciou-se a etapa de treinamento deste modelo.

Atualmente, estamos enfrentando dificuldades nesta etapa de treinamento, já que notamos que o discriminador estava ficando muito bom rápido demais, não permitindo que o gerador conseguisse avançar em seu aprendizado. Para solucionar este problema, tentaremos usar a estratégia de atualizar a *loss* do gerador com mais frequência do que a do discriminador (a priori, atualizaremos a loss do discriminador a cada 3 batches de atualização da loss do gerador).

O resultado atual do nosso treinamento é apresentado na figura abaixo. Nota-se que a saída do gerador ainda está distante do esperado e precisa ser aprimorada.

![Fatia original, fatia segmentada e saída da PulmoNet na terceira época de treinamento.](figs/example_generated_epoch_3.png?raw=true)

*Figura 8: Fatia original, fatia segmentada e saída da PulmoNet na terceira época de treinamento.*

Ademais outros problemas que estamos enfrentando durante a etapa do treinamento tratam do tamanho da nossa base de dados, que é bem grande e resulta em um processamento demorado, e o uso de recursos em GPU.

## Conclusão
O projeto da rede PulmoNet busca a geração de fatias de CTs pulmonares a partir de máscaras binárias, em duas dimensões, baseada em GANs. Esta rede utiliza uma arquitetura Pix2Pix para o gerador e uma PatchGAN para o discriminador. São usados dados da base pública ATM'22, cujos dados correspondem a volumes pulmonares de tomografias e segmentações das vias aéreas feitas por especialistas. Para a avaliação da qualidade da rede, propõe-se métodos qualitativos, quantitativos e análises de utilidade.

Seguindo o cronograma do projeto, as etapas até a entrega E2 foram cumpridas, de maneira que estamos atualmente na fase de treinamento do modelo e implementação dos métodos de avaliação. No caso do treinamento, estamos enfrentando algumas dificuldades que estão afetando a qualidade das saídas da rede, principalmente no quesito da velocidade de aprendizado do discriminador frente a do gerador.

Os próximos passos do projeto tratam da finalização do treinamento do modelo, análise das métricas de avaliação e fine-tunning e aperfeiçoamento do modelo. Caso tenhamos tempo disponível, buscaremos a geração de volumes 3D de CTs pulmonares.

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

Documento com as referências extras identificadas: https://docs.google.com/document/d/1uatPj6byVIEVrvMuvbII6J6-5usOjf8RLrSxLHJ8u58/edit?usp=sharing
