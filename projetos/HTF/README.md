# `<Redes Adversárias Generativas Sociais: Geração de Amostras Futuras para Predição da Trajetória Humana em Espaços Populados>`



## Apresentação

O presente projeto foi originado no contexto das atividades da disciplina de pós-graduação *IA376N - IA generativa: de modelos a aplicações multimodais*, 
oferecida no segundo semestre de 2024, na Unicamp, sob supervisão da Profa. Dra. Paula Dornhofer Paro Costa, do Departamento de Engenharia de Computação e Automação (DCA) da Faculdade de Engenharia Elétrica e de Computação (FEEC).

> |Nome  | RA | Especialização|
> |--|--|--|
> | Hiuri Santana de Noronha  | 229961  | Eng. Eletricista|
> | Thiago Belina Silva Ramos  | 203975  | Eng. Eletricista|
> 

## Descrição do Problema e Motivação

A predição de trajetórias humanas em ambientes densamente populados é uma aplicação em potencial para soluções futurísticas,como sistemas autônomos em veículos e robôs sociais, possibilitando a interação com pessoas e com o meio externo social de forma segura, eficiente e socialmente aceita. A princípio, os modelos devem ter a capacidade de avaliar o ambiente e prever, com um alto grau de confiabilidade e precisão, as trajetórias futuras de pedestres e veículos de todo tipo a fim de evitar colisões ou acidentes envolvendo pessoas. Entretanto, modelar um comportamento socialmente compatível é uma tarefa complexa pois leva em conta uma série de desafios que devem ser sintetizados a fim de que modelos profundos possam compreender o processo implícito das interações sociais que existem entre os seres humanos e que são completamente desconhecidas, análogo à uma distribuição desconhecida ***p'(x)*** que modela o comportamento humano em interações sociais.

Nesse contexto, a literatura técnica tem mostrado que as Redes Adversariais Generativas Sociais (S-GANs) surgem como uma solução de modelagem do comportamento humano em predição de trajetória de pedestres com a capacidade de capturar as incertezas do movimento, gerar múltiplos cenários a partir de amostras multimodais, e gerar amostras sintéticas socialmente aceitáveis que predizem como se dará a interação entre diversas pessoas em um espaço populado. Portanto, a fim de extrair as nuançes da distribuição desconhecida das interações humanas em espaços populados, nas S-GANs, introduziram-se diversos mecanismos de redes neurais profundas que possibilitam modelar as dinâmicas do comportamento social, com a introdução de embeddings e subestruturas de *variational autoencoders* (VAEs), que resultam na estrutura conhecida por S-GAN.

Portanto, a motivação para realização deste trabalho surge do desejo de compreender tal área de pesquisa para possiveis aplicações futuras considerando a segurança das pessoas, a modelagem de comportamentos socialmente aceitáveis, a modelagem de distribuições implícitas e de modelos profundos multimodais, cujas aplicações são diversas desde robótica social (robôs humanoides), *smart cities*, e sistemas inteligentes de transporte (ITS).

 A figura 1 mostra de forma simplificada e ilustrativa como se dá o processo de predição da trajetória humana em espaços populados, cujas linhas sólidas (azul, vermelha e verde) são a representação do caminho real percorrido pelo pedestre e as linhas tracejadas são a representação de amostras sintéticas multimodais (espaço-tempo) geradas a partir do modelo profundo livres de colisão.


<p align="center">
    <img src="/projetos/HTF/images/FIG01.png" alt="Figura 01: Exemplo de predição de trajetória humana em ambientes populados. Fonte: *Safety-Compliant Generative Adversarial Networks for Human Trajectory Forecasting (Parth Kothari and Alexandre Alahi, 2023)* [1]." width="400"/>
    <br>
    <em>Figura 1:Exemplo de predição de trajetória humana em ambientes populados. Fonte: *Safety-Compliant Generative Adversarial Networks for Human Trajectory Forecasting (Parth Kothari and Alexandre Alahi, 2023)* [1]..</em>
</p>

> A seguir consta o link para a apresentação em slides do entregável 2.
> [Link da Apresentação](https://docs.google.com/presentation/d/1tbUlirsQ-t3RHzTGaomvGAdto6cgHa2D/edit?usp=sharing&ouid=101073047652792710630&rtpof=true&sd=true)

## Objetivo Geral

O projeto de pesquisa proposto é estudar e desenvolver um modelo profundo de S-GAN a fim de gerar amostras futuras de possiveis trajetórias humanas em espaços populados. Ao combinar a arquitetura tradicional da GAN com VAEs, células recorrentes Long-Short-Term Memory (LSTM) e módulos de pooling,  espera-se modelar interações multimodais de espaço e tempo de trajetórias de pedestres, conforme os acordos sociais implícitos existentes em uma distribuição real desconhecida, e ponderando-se (conforme em [1]) as interações sociais na trejatória de diversas pessoas em uma cena. O processo adversarial é capaz de gerar o que se denomina de amostragem colaborativa (CS) cujo discriminador (D) possui a tarefa primordial de disciplinar o gerador (G) e garantir que a distribuição de G seja próxima à distribuição real desconhecida. Tal processo é capaz de gerar amostras confiáveis de trajetória humana em espaços populados e cujos modelos profundos devem ser avaliados apropriadamente de acordo com métricas bem estabelecidas na literatura. O módulo de interação espacial, descrito em [1], que visa criar embeddings das informações de cada pedestre e da interação com outras pessoas, a partir dos dados de entrada e um discriminador baseado em arquitetura transformer, será considerado como um objetivo secundário e de otimização que será desenvolvido após a estruturação da S-GAN.

## Metodologia

O desenvolvimento do projeto, considerando o modelo da S-GAN e seu treinamento, terá com ponto de partida e benchmark um repositório aberto de referência disponibilizado em [2]. O primeiro passo será sua avaliação e reestruturação, a fim de compreender os blocos que o compõe. Serão avaliadas redes previamente concebidas e treinadas, assim como serão realizados novos treinamentos utilizando os datasets de referência, que tem a seguinte estrutura: número do frame, número de identificação de cada pedestre presente na cena, e suas respectivas coordenadas de posição x e y. Tais datasets proporcionam ao modelo um conjunto de dados estruturados na forma de séries temporais que contém o posicionamento dos pedestres ao longo de uma via em um passo de tempo pré-determinado. A partir destas informações, os modelos podem ser treinados para encontrar o conjunto de regras sociais implícitas existentes nas trajetórias humanas.

As avaliações qualitativas serão realizadas por observações gráficas que comparam os movimentos reais observados aos preditos e as avaliações quantitativas utilizarão as métricas do benchmark ([1]), que são amplamente utilizadas na literatura, como o Erro de Deslocamento Médio (ADE - *Average Displacement Error*), que mede a distância média entre todas as posições previstas e as trajetórias reais ao longo do tempo, fornecendo uma visão geral de quão próximas as trajetórias previstas estão das trajetórias reais dos pedestres. No entanto, o ADE não capta diretamente as interações entre pedestres, algo que modelos como o Social GAN buscam melhorar por meio de técnicas como "Social Pooling". Portanto, será empregada a métrica do Erro de Deslocamento Final (FDE - *Final Displacement Error*), que de forma semelhante ao ADE, mede a distância entre a posição final das trajetórias previstas e a posição final real dos pedestres. Essa métrica é particularmente utilizada para avaliar a precisão do modelo ao prever a posição final no horizonte de previsão. Entretanto, como ADE e o FDE isoladamente, não avaliam as interações sociais entre pedestres, é importante complementar a avaliação quantitativa a partir da métrica conhecida por taxa de colisão, que avalia a porcentagem de trajetórias previstas que resultam em colisões entre pedestres. Tal métrica é fundamental para verificar se o modelo é capaz de gerar trajetórias socialmente aceitáveis. Modelos com alta taxa de colisão indicam que as interações sociais naturais não estão corretamente modeladas. No caso da S-GAN, a modelagem das interações espaciais e temporais é essencial para minimizar a taxa de colisão e gerar melhores indicadores ADE e FDE, que a partir da aplicação do método Top-K, selecionam apenas as trajetórias potenciais cuja saída consiste nas K trajetórias mais prováveis, com base nos padrões e interações aprendidas.


## Datasets 

No projeto serão consideradas duas bases de dados principais conforme tabelas abaixo, constituidas por vídeos de trajetórias humanas em espaços populados, cujos cenários são repletos em interações. O primeiro dataset é o BIWI Walking Pedestrians e oa segundo a UCY Crowd, cuja combinação é amplamente conhecida por ETH-UCY dataset. Ambos foram convertidos para dados tabulares com coordenadas do mundo real em metros que foram interpolados para obter valores a cada 0,4 segundos, tempo este correspondente ao de um frame.

|Base de Dados | Endereço na Web | Resumo descritivo|
|----- | ----- | -----|
|BIWI Walking Pedestrians Dataset | https://data.vision.ee.ethz.ch/cvl/aem/ewap_dataset_full.tgz | Vista superior de pedestres caminhando em cenários povoados.|
|UCY Crowd Data | https://graphics.cs.ucy.ac.cy/research/downloads/crowd-data | Conjunto de dados contendo pessoas em movimento em meio a multidões.|
>Tabela 1:Datasets utilizados.

A base de dados BIWI Walking Pedestrian é composta por duas cenas denominadas ETH e a Hotel, cujas imagens exemplos podem ser observadas nas figuras 2 e 3 respectiviamente. Já a UCY Crowd é composta por seis cenas denominadas Zara01, Zara02, Zara03, Students001, Students003 e Univ. Exemplos de imagens do Zara01 e Students003 podem ser observadas nas figuras 4 e 5 respectiviamente.

<p align="center">
    <img src="/projetos/HTF/images/biwi_eth.png" alt="Figura 02: Imagem do dataset ETH" width="400"/>
    <br>
    <em>Figura 2: Imagem do dataset Biwi ETH.</em>
</p>

<p align="center">
    <img src="/projetos/HTF/images/biwi_hotel.png" alt="Figura 03: Imagem do dataset HOTEL" width="400"/>
    <br>
    <em>Figura 3: Imagem do dataset Biwi Hotel.</em>
</p>

<p align="center">
    <img src="/projetos/HTF/images/crowds_zara01.jpg" alt="Figura 4: Imagem do dataset UCY Zara 01" width="400"/>
    <br><em>Figura 4: Imagem do dataset UCY Zara 01.</em>
</p>

<p align="center">
    <img src="/projetos/HTF/images/students_003.jpg" alt="Figura 5: Imagem do dataset UCY Students 03" width="400"/>
    <br><em>Figura 5: Imagem do dataset UCY Students 03.</em>
</p>


Após tratados, o formato tabular dos datasets será conforme disposto na Figura 6, em que a primeira coluna indica o frame do vídeo, a segunda a identificação do pedestre e a terceira e quarta suas coordenadas x e y respectivamente. Cada vídeo terá o seu arquivo de dado tabular correspondente, os quais ainda necessitam de tratamento para servirem de entrada do modelo. O processo de treinamento de cada modelo utiliza todas as tabelas de dados disponíveis, com exceção da qual deseja-se prever a trajetória, ou seja, supondo que se deseja prever a trajetórias da cena ETH, esta amostra será reservada para realização de testes. As demais amostras, que são Hotel, Zara01, Zara02, Zara03, Students001, Students003 e Univ, serão dívidas em amostras de treinamento e validação. Tal estrutura pode ser observada na figura 7. 

<p align="center">
    <img src="/projetos/HTF/images/TABRAWDATA.png" alt="Figura 6: Estrutura dos dados tabulares brutos" width="400"/>
    <br><em>Figura 6: Estrutura dos dados tabulares brutos.</em>
</p>

<p align="center">
    <img src="/projetos/HTF/images/ED.png" alt="Figura 7: Estrutura dos dados para treinamento, teste e validação" width="300"/>
    <br><em>Figura 7: Estrutura dos dados para treinamento, teste e validação.</em>
</p>

Para realização desse processo, as informações do dataset são dívidas em cenas, conforme parâmetros que são o tamanho do vetor de observação e do vetor de predição, o qual é equivalente ao tamanho do vetor real, que corresponde a trajetória realizada pelo pedestre que é utilizada pelo discriminador da rede S-GAN para verificar se a gerada está de acordo com os acordos sociais implícitos durante o treinamento. As informações relevantes são num primeiro momento organizadas em tensores, conforme apresentado na figura 8 e em seguida dividas em sequencias de observação e predição, conforme exemplo da figura 9, em que se considera quatro amostras de cada uma, e tem n cenas de observação. Cada "Frame" F contém identificação dos pedestres presentes, e para cada um, a posição absoluta, a posição relativa, toma como referência a posição absoluta do frame anterior, além de informações vinculadas a validade dos dados e linearidade do movimento.
<p align="center">
    <img src="/projetos/HTF/images/DATA1.png" alt="Figura 8: Tensor estruturado de dados  width="300"/>
    <br><em>Figura 8: Tensor estruturado de dados.</em>
</p>

<p align="center">
    <img src="/projetos/HTF/images/DATA2.png" alt="Figura 9: Dataset dividido em cenas" width="600"/>
    <br><em>Figura 9: Dataset dividido em cenas.</em>
</p>

Tomando algumas cenas como exemplo, conforme disposto na figura 10, é possível compreender de forma mais clara qual a transformação realizada nos dados, em que as imagens de vídeo são transformadas em cenas com passos de observação e predição pré-definidos. O objetivo dos modelos treinados é, a partir da avaliação das informações contidas nas amostras de observação, indicadas em azul na figura 10, criar trajetórias socialmente aceitáveis e livres de colisão que se aproximem às trajetórias reais indicadas em vermelho.

<div style="text-align: center;">
    <p align="center">
    <img src="../HTF/images/trajetoriaSP6.gif" alt="Imagem 1" width="400"/>
    <img src="../HTF/images/trajetoriaSP8.gif" alt="Imagem 2" width="400"/>
    <br>
    <img src="../HTF/images/trajetoriaSP7.gif" alt="Imagem 4" width="400"/>
    <img src="../HTF/images/trajetoriaSP5.gif" alt="Imagem 5" width="400"/>
    <p align="center"><em>Figura 10: Exemplos de cenas de observação e caminho real percorrido</em></p>
</div>

## Arquitetura

A arquitetura da rede SGAN do modelo de referência, composta por um gerador e um discriminador LSTM, pode ser observada nas figuras 11 e 12. A figura 13 mostra a arquiteura do discriminador implementada com células transformer encoder-only. As posições relativas são encapsuladas em embeddings, que serão a entrada das células LSTM. Estas serão responsáveis por armazenar o histórico de movimento de cada pedestre e aprender seus estados implícitos, sendo ainda necessário um módulo capaz de combinar as informações de cada um e avaliar as interações sociais existentes. Essa é a função do módulo de pooling, que pode ser implementado de duas formas diferentes.

<p align="center">
    <img src="/projetos/HTF/images/ARQ_GE.png" alt="Figura 11: Arquitetura do gerador LSTM" width="800"/>
    <br><em>Figura 11: Arquitetura do gerador LSTM.</em>
</p>

<p align="center">
    <img src="/projetos/HTF/images/ARQ_D.png" alt="Figura 12: Arquitetura do discriminador LSTM" width="800"/>
    <br><em>Figura 12: Arquitetura do discriminador LSTM.</em>
</p>

<p align="center">
    <img src="/projetos/HTF/images/MET.png" alt="Figura 13: Arquitetura do discriminador Transformer" width="2000"/>
    <br><em>Figura 13: Arquitetura do discriminador Transformer.</em>
</p>

O pooling social considera um grid em torno de cada pedestre, para que estes ajustem suas trajetórias conforme movimento dos demais, devido a sua influência mútua. Espera-se que as camadas ocultas das LSTMs capturem as propriedades de movimento que variam ao longo do tempo. Isso é feito pelo compartilhamento dos estados entre as LSTMs vizinhas. A figura 14 mostra como tal processo é realizado para a pessoa representada pelo ponto preto. Já o pooling realtivo, considera as posições relativas dos pedestres presentes nas cenas, conforme disposto na figura 15.

<p align="center">
    <img src="/projetos/HTF/images/P_soc.png" alt="Figura 13: Representação do pooling social " width="600"/>
    <br><em>Figura 14: Representação do pooling social. Fonte: Social LSTM: Human trajectory prediction in crowded spaces. (A. Alahi, K. Goel, V. Ramanathan, A. Robicquet, L. Fei-Fei, and S. Savarese) [5]</em>
</p>

<p align="center">
    <img src="/projetos/HTF/images/P_rel.png" alt="Figura 14: Representação do pooling relativo " width="600"/>
    <br><em>Figura 15: Representação do pooling relativo. Fonte: Social GAN: Socially Acceptable Trajectories with Generative Adversarial Networks. (A. Gupta, J. Johnson, L. Fei-Fei, S. Savarese, e A. Alahi) [2]</em>
</p>


## Workflow

O workflow definido para o projeto e entregável 2, como uma visão de desenvolvimento, se dará conforme estabelecido graficamente na figura 15.

<p align="center">
    <img src="/projetos/HTF/images/Workflow.png" alt="Figura 15: Workflow do projeto" width="600"/>
    <br><em>Figura 15: Workflow do projeto.</em>
</p>


## Experimentos, Resultados e Discussão de Resultados

Os resultados do projeto e treinamento da S-GAN foram efetuados em três tipos de estudo: (i) variação de hiperparâmetros, (ii) estudo de ablação da S-GAN com o discriminador dado pela arquitetura multi-encoder transformers, e (iii) estudo de convergência dos modelos de baseline formado de células LSTM no discriminador e do modelo do discriminador com células transformers, respectivamente.

### Resultados do Estudo 1
A tabela abaixo apresenta os resultados do estudo comparativo de diferentes modelos no dataset Zara1. As métricas utilizadas foram **ADE** (Average Displacement Error) e **FDE** (Final Displacement Error), onde valores menores indicam melhor desempenho na predição de trajetórias.

### Tabela de Resultados - Estudos de Primeiro Tipo

| **Modelo**                                                                                      | **ADE** | **FDE** |
|--------------------------------------------------------------------------------------------------|---------|---------|
| **Baseline: LSTM**                                                                              | 0.22    | 0.43    |
| **LSTM: ed8, md32, hd32, 841 iterações**                                                        | 0.25    | 0.48    |
| **LSTM: ed16, md64, hd64, 841 iterações**                                                       | 0.23    | 0.45    |
| **LSTM: ed32, md64, hd64, 841 iterações**                                                       | 0.23    | 0.45    |
| **LSTM: ed64, md128, hd128, 841 iterações**                                                     | 0.21    | 0.42    |
| **Transformers: x1, ed8, md32, hd32, pe T, ap T**                                               | 0.23    | 0.46    |
| **Transformers: x1, ed16, md64, hd64, pe T, ap T**                                              | 0.23    | 0.46    |
| **Transformers: x2, ed16, md64, hd64, pe T, ap T**                                              | 0.23    | 0.45    |
| **Transformers: x4, ed16, md64, hd64, pe T, ap T**                                              | 0.21    | 0.42    |
| **Transformers: x8, ed32, md64, hd64, pe T, ap T**                                              | 0.22    | 0.45    |

> Legenda
> - **ed**: Dimensão do Embedding (Embedding Dimension)
> - **md**: Dimensão da MLP (MLP Dimension)
> - **hd**: Dimensão do Hidden State (Hidden Dimension)
> - **pe**: Codificação Posicional (Positional Encoding) (T = True)
> - **ap**: Pooling por Atenção (Attention Pooling) (T = True)
> - **xN**: Número de camadas (layers) no encoder-only.

* O **Baseline LSTM** apresentou um desempenho de referência com **ADE: 0.22** e **FDE: 0.43**.
* Entre os modelos **LSTM**, a configuração **ed64, md128, hd128** alcançou os melhores valores (ADE: 0.21, FDE: 0.42), superando o baseline.
* Para os **Transformers**, o modelo **x4 (ed16, md64, hd64)** igualou o melhor desempenho do LSTM **ed64, md128, hd128** (ADE: 0.21, FDE: 0.42), mostrando que um número moderado de camadas pode ser eficiente.
* O modelo **x8 Transformers** aumentou a complexidade sem oferecer melhorias significativas, indicando que mais camadas nem sempre resultam em ganhos de desempenho.

### Resultados do Estudo 2

A tabela abaixo apresenta os resultados do segundo estudo, que avalia os modelos Transformers no dataset Zara1 mediante estudo de ablação. As métricas utilizadas foram **ADE** (Average Displacement Error) e **FDE** (Final Displacement Error), onde valores menores indicam melhor desempenho na predição de trajetórias.

### Tabela de Resultados - Estudos de Segundo Tipo

| **Modelo**                            | **Células (xN)** | **Codificação Posicional (PE)** | **Attention Pooling (AP)** | **ADE** | **FDE** |
|---------------------------------------|------------------|----------------------------------|----------------------------|---------|---------|
| **Transformer_x4_ed16_md64_hd64**     | 4                | F                                | F                          | 0.23    | 0.45    |
| **Transformer_x4_ed16_md64_hd64**     | 4                | F                                | T                          | 0.23    | 0.47    |
| **Transformer_x4_ed16_md64_hd64**     | 4                | T                                | F                          | 0.22    | 0.44    |
| **Transformer_x4_ed16_md64_hd64**     | 4                | T                                | T                          | 0.23    | 0.45    |
| **Transformer_x8_ed16_md64_hd64**     | 8                | F                                | F                          | 0.23    | 0.46    |
| **Transformer_x8_ed16_md64_hd64**     | 8                | F                                | T                          | 0.21    | 0.42    |
| **Transformer_x8_ed16_md64_hd64**     | 8                | T                                | F                          | 0.22    | 0.44    |
| **Transformer_x8_ed16_md64_hd64**     | 8                | T                                | T                          | 0.23    | 0.45    |

> Legenda
> - **Células (xN)**: Número de camadas no encoder-only.
> - **PE (Codificação Posicional)**: T = True (ativo), F = False (inativo).
> - **AP (Attention Pooling)**: T = True (ativo), F = False (inativo).
> - **ed**: Dimensão do Embedding (Embedding Dimension).
> - **md**: Dimensão da MLP (MLP Dimension).
> - **hd**: Dimensão do Hidden State (Hidden Dimension).

* O modelo **Transformer_x8_ed16_md64_hd64** com **PE: F** e **AP: T** apresentou o melhor desempenho geral, com **ADE: 0.21** e **FDE: 0.42**, demonstrando a eficácia de pooling por atenção quando a codificação posicional está desativada.
* A codificação posicional ativa (**PE: T**) trouxe benefícios modestos em configurações específicas, mas não foi determinante para os melhores resultados.
* A configuração de 8 camadas (**x8**) mostrou-se mais eficiente em nas relações de longo alcance resultando no menor FDE de 0.42, enquanto as configurações com 4 camadas (**x4**) apresentaram resultados comparativos similares apresentando menor complexidade computacional.
* O modelo **Transformer_x4_ed16_md64_hd64** com **PE: T** e **AP: F** destacou-se como uma alternativa equilibrada, com **ADE: 0.22** e **FDE: 0.44**.


### Resultados do Estudo 3

A tabela abaixo apresenta os resultados do terceiro estudo, que compara um modelo LSTM com um modelo Transformer utilizando o dataset Zara1 após 5047 iterações. O modelo LSTM treinado é igual ao modelo baseline, porém treinado com uma fração de um quarto das iterações cujo modelo baseline foi treinado (20000+ mil iterações). As métricas utilizadas foram **ADE** (Average Displacement Error) e **FDE** (Final Displacement Error), onde valores menores indicam melhor desempenho na predição de trajetórias.

### Tabela de Resultados - Estudo de Terceiro Tipo

| **Modelo**                            | **Células (xN)** | **Codificação Posicional (PE)** | **Attention Pooling (AP)** | **ADE** | **FDE** |
|---------------------------------------|------------------|----------------------------------|----------------------------|---------|---------|
| **LSTM_ed16_md64_hd64**               | -                | -                                | -                          | 0.23    | 0.46    |
| **Transformer_x8_ed16_md64_hd64**     | 8                | T                                | T                          | 0.21    | 0.41    |

> Legenda
> - **Células (xN)**: Número de camadas no encoder-only (aplicável apenas a Transformers).
> - **PE (Codificação Posicional)**: T = True (ativo), F = False (inativo) (aplicável apenas a Transformers).
> - **AP (Attention Pooling)**: T = True (ativo), F = False (inativo) (aplicável apenas a Transformers).
> - **ed**: Dimensão do Embedding (Embedding Dimension).
> - **md**: Dimensão da MLP (MLP Dimension).
> - **hd**: Dimensão do Hidden State (Hidden Dimension).

* O modelo **Transformer_x8_ed16_md64_hd64** com **PE: T** e **AP: T** superou o **LSTM_ed16_md64_hd64**, alcançando um desempenho 10% superior com **ADE: 0.21** e **FDE: 0.41**, mesmo com o processo de treinamento reduzido (< 10 mil iterações).
* A vantagem do Transformer evidencia a eficácia dos mecanismos de atenção e codificação posicional em capturar dependências complexas em tarefas de predição de trajetórias.
* O LSTM, apesar de um desempenho inferior, mostrou resultados consistentes, reforçando sua eficiência em tarefas de menor complexidade computacional.

### Resultados Gráficos do Treinamento

<div style="text-align: center;">
    <p align="center">
    <!-- Primeira linha de 3 imagens -->
    <img src="trained_models/first_study/figure_LSTM_ed8_md32_hd32_841_zara1.png" alt="LSTM ed8" style="width: 400px;">
    <img src="trained_models/first_study/figure_LSTM_ed16_md64_hd64_841_zara1.png" alt="LSTM ed16" style="width: 400px;">
    <br>
    <!-- Segunda linha de 3 imagens -->
    <img src="trained_models/first_study/figure_Transformer_x2_ed16_md64_hd64_peT_apT_841_zara1.png" alt="Transformer x2" style="width: 400px;">
    <img src="trained_models/first_study/figure_Transformer_x4_ed16_md64_hd64_peT_apT_841_zara1.png" alt="Transformer x4" style="width: 400px;">
    <p align="center"><p><em>Figura 16: Gráficos de convergência dos treinamentos realizados no estudo de primeiro tipo para 841 iterações.</em></p>
</div>

1. **Modelo LSTM com ed8 (Figura 8.a):** observa-se maior instabilidade no treinamento, especialmente nos valores de perda (Loss). As métricas ADE e FDE apresentam flutuações maiores, indicando dificuldade na convergência.
2. **Modelo LSTM com ed16 (Figura 8.b):** com maior dimensão no embedding (ed16), o treinamento é visivelmente mais estável em comparação com ed8. Tanto as métricas ADE quanto FDE mostram menos variações, refletindo uma convergência mais consistente.
3. **Modelo Transformer x2 (Figura 8.c):** com ed16 e arquitetura Transformer, o modelo apresenta métricas ADE e FDE melhores em comparação aos modelos LSTM, embora ainda mostre leve instabilidade inicial.
4. **Modelo Transformer x4 (Figura 8.d):** este modelo é o mais estável e apresenta as menores métricas ADE e FDE ao longo do treinamento. A diferença na performance é evidente com o progresso do treinamento, indicando maior eficácia na previsão de trajetórias.
* Para mais resultados gráficos do treinamento, visite a pasta "trained_models" onde estão todos os gráficos de treinamento por modelo.

### Resultados Gráficos de Inferência


<div style="text-align: center;">
    <p align="center">
    <img src="../HTF/images/LSTM/Animação/BLG1.gif" alt="Imagem 1" width="400"/>
    <img src="../HTF/images/LSTM/Animação/BLG2.gif" alt="Imagem 2" width="400"/>
    <br>
    <img src="../HTF/images/LSTM/Animação/BLG3.gif" alt="Imagem 4" width="400"/>
    <img src="../HTF/images/LSTM/Animação/BLG4.gif" alt="Imagem 5" width="400"/>
    <br>
    <img src="../HTF/images/LSTM/Animação/BLG5.gif" alt="Imagem 4" width="400"/>
    <img src="../HTF/images/LSTM/Animação/BLG6.gif" alt="Imagem 5" width="400"/>
    <p align="center"><p align="center"><em>Figura 17: Exemplos de cenas de observação, caminho real percorrido e geração de um possivel caminho para o modelo LSTM</em></p>
</div>

<div style="text-align: center;">
     <p align="center"><!-- Primeira linha de 3 imagens -->
    <img src="../HTF/images/LSTM/KDE/BLKDE1.png" alt="Imagem 1" width="320"/>
    <img src="../HTF/images/LSTM/KDE/BLKDE2.png" alt="Imagem 2" width="320"/>
    <img src="../HTF/images/LSTM/KDE/BLKDE3.png" alt="Imagem 3" width="320"/>
    <br>
    <!-- Segunda linha de 3 imagens -->
    <img src="../HTF/images/LSTM/KDE/BLKDE4.png" alt="Imagem 4" width="320"/>
    <img src="../HTF/images/LSTM/KDE/BLKDE5.png" alt="Imagem 5" width="320"/>
    <img src="../HTF/images/LSTM/KDE/BLKDE6.png" alt="Imagem 6" width="320"/>
    <br>
    <!-- Segunda linha de 3 imagens -->
    <img src="../HTF/images/LSTM/KDE/BLKDE7.png" alt="Imagem 7" width="320"/>
    <img src="../HTF/images/LSTM/KDE/BLKDE8.png" alt="Imagem 8" width="320"/>
    <img src="../HTF/images/LSTM/KDE/BLKDE9.png" alt="Imagem 9" width="320"/>
    <br>
    <!-- Segunda linha de 3 imagens -->
    <img src="../HTF/images/LSTM/KDE/BLKDE10.png" alt="Imagem 10" width="320"/>
    <img src="../HTF/images/LSTM/KDE/BLKDE11.png" alt="Imagem 11" width="320"/>
    <img src="../HTF/images/LSTM/KDE/BLKDE12.png" alt="Imagem 12" width="320"/>
    <p align="center"><p align="center"><p><em>Figura 18: Exemplos de distribuição de probabilidade de possiveis caminhos gerados para o modelo LSTM.</em></p>
</div>

<div style="text-align: center;">
    <p align="center">
    <img src="../HTF/images/Transformer/Animação/GTR1.gif" alt="Imagem 1" width="400"/>
    <img src="../HTF/images/Transformer/Animação/GTR2.gif" alt="Imagem 2" width="400"/>
    <br>
    <img src="../HTF/images/Transformer/Animação/GTR3.gif" alt="Imagem 4" width="400"/>
    <img src="../HTF/images/Transformer/Animação/GTR4.gif" alt="Imagem 5" width="400"/>
    <br>
    <img src="../HTF/images/Transformer/Animação/GTR5.gif" alt="Imagem 4" width="400"/>
    <img src="../HTF/images/Transformer/Animação/GTR6.gif" alt="Imagem 5" width="400"/>
    <p align="center"><p align="center"><em>Figura 19: Exemplos de cenas de observação, caminho real percorrido e geração de um possivel caminho para o modelo Transformer</em></p>
</div>

<div style="text-align: center;">
     <p align="center"><!-- Primeira linha de 3 imagens -->
    <img src="../HTF/images/Transformer/KDE/TR1.png" alt="Imagem 1" width="320"/>
    <img src="../HTF/images/Transformer/KDE/TR2.png" alt="Imagem 2" width="320"/>
    <img src="../HTF/images/Transformer/KDE/TR3.png" alt="Imagem 3" width="320"/>
    <br>
    <!-- Segunda linha de 3 imagens -->
    <img src="../HTF/images/Transformer/KDE/TR4.png" alt="Imagem 4" width="320"/>
    <img src="../HTF/images/Transformer/KDE/TR5.png" alt="Imagem 5" width="320"/>
    <img src="../HTF/images/Transformer/KDE/TR6.png" alt="Imagem 6" width="320"/>
    <br>
    <!-- Segunda linha de 3 imagens -->
    <img src="../HTF/images/Transformer/KDE/TR7.png" alt="Imagem 7" width="320"/>
    <img src="../HTF/images/Transformer/KDE/TR8.png" alt="Imagem 8" width="320"/>
    <img src="../HTF/images/Transformer/KDE/TR14.png" alt="Imagem 9" width="320"/>
    <br>
    <!-- Segunda linha de 3 imagens -->
    <img src="../HTF/images/Transformer/KDE/TR10.png" alt="Imagem 10" width="320"/>
    <img src="../HTF/images/Transformer/KDE/TR11.png" alt="Imagem 11" width="320"/>
    <img src="../HTF/images/Transformer/KDE/TR12.png" alt="Imagem 12" width="320"/>
    <p align="center"><p align="center"><p><em>Figura 20: Exemplos de distribuição de probabilidade de possiveis caminhos gerados para o modelo transformer.</em></p>
</div>

## Conclusão

Para os próximos passos, conforme E3, espera-se compreender os detalhes da rede S-GAN e a modelagem multimodal, bem como criar estruturar modelos profundos de S-GAN para que a partir de novos treinamentos com os datasets de referência, ajuste de parâmetros e melhorias na rede profunda, espera-se obter resultados similares aos encontrados na literatura e que estejam de acordo com resultados socialmente aceitáveis conforme descrito nas diversas fontes sobre o tema.


## Referências Bibliográficas



> [1] P. Kothari e A. Alahi, "Safety-Compliant Generative Adversarial Networks for Human Trajectory Forecasting," IEEE Transactions on Intelligent Transportation Systems, vol. PP, pp. 1-11, abr. 2023.
>  
> [2] A. Gupta, J. Johnson, L. Fei-Fei, S. Savarese, e A. Alahi, "Social GAN: Socially Acceptable Trajectories with Generative Adversarial Networks," IEEE Conference on Computer Vision and Pattern Recognition (CVPR), 2018.
> 
> [3] Z. Lv, X. Huang, e W. Cao, "An improved GAN with transformers for pedestrian trajectory prediction models," International Journal of Intelligent Systems, vol. 37, nº 8, pp. 4417-4436, 2022.
> 
> [4] S. Eiffert, K. Li, M. Shan, S. Worrall, S. Sukkarieh and E. Nebot, "Probabilistic Crowd GAN: Multimodal Pedestrian Trajectory Prediction Using a Graph Vehicle-Pedestrian Attention Network," in IEEE Robotics and Automation Letters, vol. 5, no. 4, pp. 5026-5033, Oct. 2020.
>
> [5] A. Alahi, K. Goel, V. Ramanathan, A. Robicquet, L. Fei-Fei, and S. Savarese, "Social LSTM: Human trajectory prediction in crowded spaces," in Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR), 2016, pp. 961-971.
>
> [6] A. Mohamed, D. Zhu, W. Vu, M. Elhoseiny, and C. Claudel, "Social-Implicit: Rethinking Trajectory Prediction Evaluation and The Effectiveness of Implicit Maximum Likelihood Estimation," *2022*, pp. 1-12.
