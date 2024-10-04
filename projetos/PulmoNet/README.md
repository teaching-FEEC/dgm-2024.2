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
> Resumo do objetivo, metodologia e resultados obtidos (na entrega E2 é possível relatar resultados parciais). Sugere-se máximo de 100 palavras.

## Descrição do Problema/Motivação
As tomografias computadorizadas (CT) pulmonares, juntamente com a segmentação das vias aéreas, desempenham um papel crucial no diagnóstico preciso de doenças pulmonares. Ao gerar imagens detalhadas da região torácica, ela permite que médicos mapeiem a anatomia das vias aéreas antes de procedimentos cirúrgicos, avaliando a extensão de lesões e facilitando o acompanhamento da progressão de doenças respiratórias [[2]](#2). Além disso, a CT é fundamental para monitorar a eficácia de tratamentos e detectar seus possíveis efeitos colaterais [[5]](#5).

A complexidade e diversidade do corpo humano dificultam a obtenção de grandes volumes de dados médicos para treinar modelos de aprendizado de máquina, como as redes neurais. Essa escassez de dados pode levar a diagnósticos imprecisos, comprometendo a qualidade do atendimento aos pacientes [[6]](#6). Com as redes generativas é possível criar dados de forma a compensar essa escassez, permitindo que as redes aprendam muito mais detalhes do que utilizando apenas aqueles obtidos de exames reais.

[Link para o vídeo de apresentação E1](https://drive.google.com/file/d/1TlpQOlCh_lAI0-jPPMPWOzGZ_werCo3d/view?usp=sharing)

[Link para a apresentação de slides E1](https://docs.google.com/presentation/d/1b8W0Cw1eiTbWlJ0CJJ8eMRA4zyu2iLhYvggi55-mOb0/edit?usp=sharing)

## Objetivo
Este projeto visa gerar imagens sintéticas de tomografia computadorizada (CT) da região torácica de alta fidelidade, também produzindo máscaras de segmentação das vias aéreas. O modelo generativo proposto terá como saída volumes de CT da região do tórax, ou seja, uma combinação de fatias que juntas formarão o equivalente a um exame real.

Caso este resultado se concretize antes do prazo estipulado pelo cronograma e ainda reste tempo para o aprofundamento do projeto, buscar-se-á a geração de imagens 3D de tomografias pulmonares, isto é, espera-se aumentar o escopo do projeto para gerar volumes com a mesma estratégia da síntese de imagens, com as devidas adequações necessárias a esta nova estrutura.

## Metodologia
### Materiais de Referência
Este projeto usará como inspiração inicial o trabalho desenvolvido em [[1]](#1), o qual propõe duas arquiteturas baseadas em GANs para a síntese de imagens CT pulmonares a partir de máscaras binárias que segmentam a região pulmonar. Das arquiteturas propostas, inspirar-se-á na arquitetura Pix2Pix, na qual o gerador é composto de um encoder que aumenta a profundidade da imagem enquanto diminui suas dimensões, seguido de um decoder que realiza o processo oposto. Tal arquitetura também utiliza conexões residuais. Na arquitetura Pix2Pix, o discriminador é composto por cinco camadas convolucionais, onde as quatro primeiras são seguidas por uma camada de ativação *LeakyReLu*, enquanto a última é seguida de uma função *sigmoide*. 

Além do artigo [[1]](#1), também serão considerados os trabalhos realizados em [[3]](#3) e [[4]](#4). No primeiro, desenvolveu-se uma GAN condicional para a geração de imagens CT pulmonares a partir de imagens de ressonância magnética. Já no segundo, utiliza-se um modelo baseado em GAN para a segmentação do pulmão em imagens CT que contém anomalias no tecido pulmonar. Apesar dos objetivos de tais trabalhos não serem os mesmos objetivos propostos para o presente projeto, eles servirão de apoio para proposição de modificações na arquitetura, estratégias de treino e de validação de resultados.   

### Modelo Proposto
> [comentar sobre como está nossa arquitetura da rede (dimensões e camadas) e sobre as funções de loss que vamos utilizar durante o treinamento. Também colocar uma figura parecida com a imagem 2 do artigo 1]

A rede generativa implementada neste trabalho segue a arquitetura de uma rede adversária generativa (GAN), a qual é composta por duas redes com estratégias conflitantes. Há uma rede gerativa, que busca gerar dados sintéticos similares aos dados reais, e há uma rede discriminadora, que deve ser capaz de classificar corretamente suas entradas como "reais" ou "falsas". Com isso, uma boa rede generativa deve ser capaz de enganar o discriminador, ao passo que um bom discriminador deve identificar corretamente os dados sintéticos em meio aos dados reais.

No caso da aplicação para imagens, várias arquiteturas surgiram na literatura, a depender do tipo de problema. Para a conversão de imagem para imagem, o modelo Pix2Pix foi por muito tempo o estado da arte, sendo aplicada por exemplo no trabalho [[1]](#1).
Tomando como inspiração, este artigo, implementamos a primeira versão do nosso modelo generativo. A figura abaixo ilustra sua arquitetura.

> imagem

> comentários adicionais sobre nosso modelo

### Bases de Dados e Evolução
> Faça uma descrição sobre o que concluiu sobre esta base. Sugere-se que respondam perguntas ou forneçam informações indicadas a seguir:
> * Qual o formato dessa base, tamanho, tipo de anotação? **ok**
> * Quais as transformações e tratamentos feitos? Limpeza, reanotação, etc. **ok**
> * Inclua um sumário com estatísticas descritivas da(s) base(s) de estudo. **acrescentar**
> * Utilize tabelas e/ou gráficos que descrevam os aspectos principais da base que são relevantes para o projeto. **acrescentar**

Apesar de inspirar-se no artigo [[1]](#1), para o desenvolvimento deste projeto será usada a base de dados ATM'22, cuja descrição está na tabela abaixo. Tal base de dados não foi usada no desenvolvimento do projeto em [[1]](#1), mas foi escolhida no presente projeto devido a sua amplitude, a presença de dados volumétricos e em razão das imagens possuírem a delimitação das vias aéreas obtidas através de especialistas. Os volumes da base ATM'22 foram adquiridos em diferentes clínicas e considerando diferentes contextos clínicos. Construída para a realização de um desafio de segmentação automática de vias aéria utilizando IA, a base de dados está dividida em 300 volumes para treino, 50 para validação e 150 para teste.

|Base de Dados | Endereço na Web | Resumo descritivo|
|----- | ----- | -----|
|ATM'22 | https://zenodo.org/records/6590774 e https://zenodo.org/records/6590775  | Esta base contém 500 volumes CTs pulmonares, nos quais as vias aéreas estão completamente anotadas, i.e., delimitadas. Tais volumes serão fatiados em imagens 2-D, segmentados e transformados. Esta base de dados foi utilizada para um desafio de segmentação [[2]](#2).|

Os dados desta base são arquivos em padrão DICOM (Digital Imaging and Communications in Medicine), com extensão *.nii.gz, e contêm todo o volume pulmonar obtido durante um exame de tomografia. Cada arquivo com um volume pulmonar é acompanhado por um outro arquivo de mesma extensão contendo as anotações feitas por especialistas.
Dado que este trabalho centrará-se na geração de imagens sintéticas em duas dimensões de CTs pulmonares, estes volumes pulmonares serão fatiados no eixo transversal, assim como ilustrado na imagem abaixo. Como resultado, fatiaremos os 500 volumes pulmores em uma quantidade muito maior de imagens 2-D, aumentando o tamanho dos conjuntos de dados disponíveis para treinamento, validação e testes.

![Exemplo de fatia de CT pulmonar obtida a partir da base de dados ATM'22.](https://github.com/julia-cdp/dgm-2024.2/tree/main/projetos/PulmoNet/figs/dataset_exemplo_fatia.png?raw=true)

A quantia exata de dados que serão utilizados depende da configuração da fatia obtida. Isto é, não serão utilizadas todas as fatias do volume pulmonar, mas sim apenas as imagens que apresentarem o pulmão completo e cercado por tecidos. A partir desta condição, as fatias serão selecionadas e passadas pela etapa de pré-processamento. Ressalta-se que esta seleção é necessária, uma vez que é uma restrição da biblioteca em Python lungmask [[7]](#7), utilizada para segmentação automática de CTs pulmonares.
Esta segmentação é uma etapa essencial do workflow, posto que os dados de entrada da rede geradora da GAN serão máscaras pulmonares, tal como feito em [[1]](#1).

Após a limpeza e seleção do subconjunto de dados, continua-se com a etapa de pré-processamento. Os pixels de cada amostra serão normalizados para um intervalo entre 0 e 1, em escala de cinza, e as dimensões da imagem serão transformadas para (28, 28, 1).
**[---------- completar aqui caso eu tenha esquecido de mais algum passo no pré-processamento ----------------]**

> **[colocar aqui comentários da parte estatística]**

> **[talvez seja legal colocar um esquemático como a Figura 1 do artigo 1]**

### Workflow
> Use uma ferramenta que permita desenhar o workflow e salvá-lo como uma imagem (Draw.io, por exemplo). Insira a imagem nessa seção.
> Você pode optar por usar um gerenciador de workflow (Sacred, Pachyderm, etc) e nesse caso use o gerenciador para gerar uma figura para você.
> Lembre-se que o objetivo de desenhar o workflow é ajudar a quem quiser reproduzir seus experimentos. 

O fluxo de trabalho proposto por este projeto inicia-se com a obtenção da base de dados ATM'22 e seu devido tratamento, conforme detalhado na seção anterior.
Feito isso, faz-se a divisão entre os conjuntos de treinamento (80%), validação (10%) e testes (10%). 
Usando os dois primeiros conjuntos, alimenta-se a rede generativa com as imagens com as máscaras binárias. Já a rede discriminadora recebe os dados reais (antes da segmentação) e os dados sintéticos, devendo classificar cada um como "real" ou "falso".
Após o treinamento, avalia-se os dados sintéticos a partir de três perspectivas: análise qualitativa, análise quantitativa e análise de utilidade, as quais serão descritas em detalhes nas próximas seções deste relatório.

> imagem fluxo geral + imagem fluxo detalhado

Destaca-se que, em operação (após a fase treinamento), espera-se que o modelo receba máscaras binárias com o formato do pulmão e um ruído, retonando o preenchimento da área interna do pulmão.
Uma mesma máscara binária poderá gerar imagens sintéticas distintas, devido o ruído aleatório adicionado na entrada do modelo.
Os dados sintéticos deverão ser bons o suficiente para ajudarem no treinamento de modelo de segmentação das vias aéreas e potencialmente substituir o uso de dados reais, para a preservação da privacidade dos pacientes.

### Ferramentas Relevantes
A ferramenta escolhida para o desenvolvimento da arquitetura dos modelos e de treinamento é o PyTorch, em função de sua relevância na área e familiaridade por parte dos integrantes do grupo.
Ademais, para o desenvolvimento colaborativo dos modelos entre os estudantes, opta-se pela ferramenta de programação Google Collaboratory.
Já para o versionamento dos modelos e para ajustar seus hiperparâmetros, decidiu-se pela ferramenta Wandb AI dentre as opções disponíveis no mercado. E, além disso, a ferramenta do GitHub também auxiliará no versionamento dos algoritmos desenvolvidos.

### Métricas de Avaliação
Para avaliar a qualidade dos resultados obtidos com o modelo de síntese, propõe-se três tipos de avaliação: análise qualitativa, análise quantitativa e análise frente a um benchmark.

#### Análise Qualitativa
Esta estratégia será utilizada apenas nas etapas iniciais do desenvolvimento do projeto, na qual os próprios estudantes irão observar os resultados sintéticos, sejam eles imagens e/ou  volumes, e compararão com os dados reais esperados. Com isto, faz-se uma avaliação se a imagem gerada estaria muito distante de um CT pulmonar ou se o modelo já estaria se encaminhando para bons resultados. Após esta etapa, as avaliações do modelo serão feitas por meio das análises quantitativa e de utiliddade.

#### Análise Quantitativa
Já a análise quantitativa trata de uma avaliação sobre as imagens a partir dos métodos Fréchet Inception Distance (FID) e Structural Similarity Index (SSIM), os quais são utilizados para avaliação de qualidade das imagens sintéticas e de similaridade com dados reais. Ambas estratégias foram utilizadas pelos pesquisadores do artigo [[1]](#1), o que permite uma avaliação dos nossos resultados frente a esta outra pesquisa.

> obs: Vou detalhar mais essa parte!!

> FID: InceptionV3 e autoencoders + The generated and real images from the test set were passed through the encoder, where the distributions of the generated and real images were calculated and used to compute the FID distance.

> SSIM: SSIM was used in order to compare each image with its ground-truth counterpart and uses three image characteristics to compare two images: luminance, contrast distortion and loss of structural correlation

> Gerar gráfico com distribuição real e distribuição sintética, usadas no cálculo da FID. Podemos comparar nossos resultados dessa métrica com os resultados do artigo 1

#### Análise de Utilidade / Aplicabilidade
Por último, a análise de benchmark, que também pode ser considerada um estratégia quantitativa, tem como proposta a comparação das saídas de uma rede de segmentação já consolidada a partir dos dados gerados pela PulmoNet e de dados reais. Feito isso, compara-se ambas as saídas da rede, por meio do cálculo do coeficiente DICE (obtido a partir da precisão e recall da predição) e da quantidade de ramificações (métricas escolhidas com base na referência do artigo [[2]](#2)) e avalia-se se os dados sintéticos são bons o suficiente em uma aplicação real, isto é, avalia-se a utilidade do modelo generativo proposto.

> Isso aqui vamos manter?

#### Análise de Privacidade
> por serem dados na área da saúde, querem colocar essa métrica?

### Cronograma

| Nº da Tarefa | Descrição                                                                 | Data Prevista de Finalização | Semanas Entre Etapas |
|--------------|---------------------------------------------------------------------------|------------------------------|----------------------|
| 1            | Leitura de artigos, familiarização com a base de dados e GANs              | 10/09                        |                      |
| 2            | Primeira versão da GAN (inspirada no artigo de referência)                | 24/09                        | 2 semanas            |
| 3            | Estrutura de avaliação bem delimitada                                     | 07/10                        | 2 semanas            |
| 4            | E2                                                                        | 08/10                        | 1 dia                |
| 5            | Primeiros resultados com imagens segmentadas e valores para validação     | 15/10                        | 1 semana             |
| 6            | Fine-tuning e aperfeiçoamento do modelo                                   | 29/10                        | 2 semanas            |
| 7            | Evoluir para redes 3D ou continuar aperfeiçoando o modelo                 | 05/11                        | 1 semana             |
| 8            | E3                                                                        | 25/11                        | 3 semanas            |



## Experimentos, Resultados e Discussão dos Resultados

> Na entrega parcial do projeto (E2), essa seção pode conter resultados parciais, explorações de implementações realizadas e 
> discussões sobre tais experimentos, incluindo decisões de mudança de trajetória ou descrição de novos experimentos, como resultado dessas explorações.

> Na entrega final do projeto (E3), essa seção deverá elencar os **principais** resultados obtidos (não necessariamente todos), que melhor representam o cumprimento
> dos objetivos do projeto.

> A discussão dos resultados pode ser realizada em seção separada ou integrada à seção de resultados. Isso é uma questão de estilo.
> Considera-se fundamental que a apresentação de resultados não sirva como um tratado que tem como único objetivo mostrar que "se trabalhou muito".
> O que se espera da seção de resultados é que ela **apresente e discuta** somente os resultados mais **relevantes**, que mostre os **potenciais e/ou limitações** da metodologia, que destaquem aspectos
> de **performance** e que contenha conteúdo que possa ser classificado como **compartilhamento organizado, didático e reprodutível de conhecimento relevante para a comunidade**. 

## Conclusão

> A seção de Conclusão deve ser uma seção que recupera as principais informações já apresentadas no relatório e que aponta para trabalhos futuros.
> Na entrega parcial do projeto (E2) pode conter informações sobre quais etapas ou como o projeto será conduzido até a sua finalização.
> Na entrega final do projeto (E3) espera-se que a conclusão elenque, dentre outros aspectos, possibilidades de continuidade do projeto.

## Referências Bibliográficas

<a id="1">[1]</a> : José Mendes et al., Lung CT image synthesis using GANs, Expert Systems with Applications, vol. 215, 2023, pp. 119350., https://www.sciencedirect.com/science/article/pii/S0957417422023685.

<a id="2">[2]</a> : Minghui Zhang et al., Multi-site, Multi-domain Airway Tree Modeling (ATM'22): A Public Benchmark for Pulmonary Airway Segmentation, https://arxiv.org/abs/2303.05745.

<a id="3">[3]</a> :  Jacopo Lenkowicz et al., A deep learning approach to generate synthetic CT in low field MR-guided radiotherapy for lung cases, Radiotherapy and Oncology, vol. 176, 2022, pp. 31-38, https://www.sciencedirect.com/science/article/pii/S0167814022042608.

<a id="4">[4]</a> : Swati P. Pawar and Sanjay N. Talbar, LungSeg-Net: Lung field segmentation using generative adversarial network, Biomedical Signal Processing and Control, vol. 64, 2021, 102296, https://www.sciencedirect.com/science/article/pii/S1746809420304158.

<a id="5">[5]</a> : Tekatli, Hilâl et al. “Artificial intelligence-assisted quantitative CT analysis of airway changes following SABR for central lung tumors.” Radiotherapy and oncology : journal of the European Society for Therapeutic Radiology and Oncology vol. 198 (2024): 110376. doi:10.1016/j.radonc.2024.110376, https://pubmed.ncbi.nlm.nih.gov/38857700/

<a id="6">[6]</a> : Zhang, Ling et al. “Generalizing Deep Learning for Medical Image Segmentation to Unseen Domains via Deep Stacked Transformation.” IEEE transactions on medical imaging vol. 39,7 (2020): 2531-2540. doi:10.1109/TMI.2020.2973595, https://www.ncbi.nlm.nih.gov/pmc/articles/PMC7393676/

<a id="7">[7]</a> : Hofmanninger, J., Prayer, F., Pan, J. et al. Automatic lung segmentation in routine imaging is primarily a data diversity problem, not a methodology problem. Eur Radiol Exp 4, 50 (2020). https://doi.org/10.1186/s41747-020-00173-2

Documento com as referências extras identificadas: https://docs.google.com/document/d/1uatPj6byVIEVrvMuvbII6J6-5usOjf8RLrSxLHJ8u58/edit?usp=sharing
