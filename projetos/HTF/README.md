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

A predição de trajetórias humanas em ambientes densamente povoados é uma tarefa essencial para que no futuro, sistemas autônomos como carros e robôs possam interagir de forma segura e eficiente com pessoas. É necessário que estes consigam avaliar o ambiente e prever com um grau confiável de precisão as trajetórias futuras dos pedestres a fim de evitar colisões. Entretanto, modelar este comportamento é uma tarefa complexa e que impõe uma série de desafios pois é necessário entender o processo implícito de interações sociais existentes entre seres humanos.

Nesse contexto, as Redes Adversariais Generativas (GANs) surgem como uma possível solução.  As GANs, por serem modelos generativos, são capazes de capturar a incerteza inerente ao movimento humano, gerando múltiplos cenários futuros multimodais, dentre os quais os socialmente aceitáveis serão escolhidos.  A fim de tornar esse processo ainda mais eficiente, foram introduzidos mecanismos sociais neste tipo de rede a fim de permitir que o gerador compreenda melhor tais dinâmicas, criando-se assim as GANs Sociais (SGANs). A principal motivação para a realização deste trabalho é a compreensão desta nova e latente área para aplicações futuras em robótica social (robôs humanoides), cidades inteligentes e sistemas inteligentes de transporte.

 A figura 01 abaixo mostra de forma simplificada e ilustrativa como se dá o processo de predição da trajetória humana em espaços populados, cujas linhas sólidas (azul, vermelha e verde) são a representação do caminho real percorrido pelo pedestre e as linhas tracejadas são a representação de amostras sintéticas multimodais (espaço-tempo) geradas a partir do modelo profundo livres de colisão.

![objetivo](/projetos/HTF/images/FIG01.png)
>Figura 01: Exemplo de predição de trajetória humana em ambientes povoados

Fonte: *Safety-Compliant Generative Adversarial Networks for Human Trajectory Forecasting(Parth Kothari and Alexandre Alahi, 2023)*


> A seguir tem-se os links para a apresentação em slides do entregável 02.
 
[Link da Apresentação](https://docs.google.com/presentation/d/1vHEd9DeePXOXCwRnCimp_5Cx16KPEKW2/edit?usp=sharing&ouid=101073047652792710630&rtpof=true&sd=true)

## Objetivo

O projeto de pesquisa proposto irá estudar e propor modelos profundos de redes Generativas Adversariais (GANs) para geração de amostras futuras de possiveis trajetórias humanas em espaços populados. Ao combinar a arquitetura tradicional da GAN com auto encoders, células recorrentes Long-Short-Term Memory (LSTM) e módulos de pooling,  espera-se modelar interações multimodais de espaço e tempo de trajetórias de pedestres conforme os acordos sociais implícitos existentes em uma distribuição real desconhecida. O processo adversarial é capaz de gerar o que se denomina de amostragem colaborativa (CS) cujo discriminador (D) possui a tarefa primordial de disciplinar o gerador (G) e garantir que a distribuição de G seja próxima à distribuição real desconhecida. Tal processo é capaz de gerar amostras confiáveis de trajetória humana em espaços populados e cujos modelos profundos devem ser avaliados apropriadamente de acordo com métricas bem estabelecidas na literatura. Apesar da proposta inicial contemplar a implementação de novas estruturas descritas em [1], dentre as quais destacam-se o módulo de interação espacial dos dados de entrada para criar um embedding e o descriminador baseado em arquitetura transformer, este será um objetivo secundário que será trabalhado apenas se os objetivos primários forem alcançado.

## Metodologia

O desenvolvimento e treinamento do modelo terá com ponto de partida e benchmark o repositório disponibilizado em [2]. Portanto, o primeiro passo será sua reprodução, a fim de confirmar os resultados disponibilizados e compreender as propostas realizadas. O repositório permite a avaliação de redes previamente treinadas disponibilizadas, bem como novos treinamentos tanto para dos datasets ultilizados , quantos para outros que respeitem o formato proposto, que são dados tabulares obtidos a partir de videos que contem o número do frame, o número de identificação do pedestre conforme ordem de aparição no vídeo e suas coordenadas x e y. Tais datasets proporcionam ao modelo profundo um conjunto de dados estruturados na forma de séries temporais contendo o posicionamento de pedestres ao longo de uma via a um determinado tempo. Com isso, os modelos profundos podem ser treinados para modelar o conjunto de regras sociais implícitas existentes em trajetórias humanas.

As avaliações qualitativas serão realizadas por observações gráficas que comparam os movimentos reais observados, aos preditos comforme o exemplo da figura xx e as avaliações quantitaivas utilizarão as métricas do benchmark que são o Erro de Deslocamento Médio (ADE - Average Displacement Error), que mede a distância média entre todas as posições previstas e as trajetórias reais ao longo do tempo, fornecendo uma visão geral de quão próximas as trajetórias previstas estão das trajetórias reais dos pedestres. No entanto, o ADE não capta diretamente as interações entre pedestres, algo que modelos como o Social GAN busca melhorar por meio de técnicas como "Social Pooling". Tem-se também o Erro de Deslocamento Final (FDE - Final Displacement Error), que de forma semelhante ao ADE, mede a distância entre a posição final das trajetórias previstas e a posição final real dos pedestres. Essa métrica é particularmente utilizada para avaliar a precisão do modelo ao prever a posição ao final do horizonte de previsão. Assim como o ADE, o FDE, isoladamente, não avalia as interações sociais entre pedestres, sendo complementadas pela Taxa de Colisão, que avalia a porcentagem de trajetórias previstas que resultam em colisões entre pedestres. Esta métrica é fundamental para verificar se o modelo gera trajetórias socialmente aceitáveis. Modelos que geram trajetórias com altas taxas de colisão indicam que o comportamento aprendido não está alinhado com interações sociais naturais, notáveis pela ausência de colisões em geral. No caso de modelos como o Social GAN, a consideração das interações espaciais e temporais é essencial para minimizar essas colisões. Tais métricas podem ser avalidadas considerando o número total de predições geradas, ou ainda pela aplicação do método Top-K, em que são selecionadas apenas trajetórias potenciais para o futuro, e a saída do modelo consiste nas K trajetórias mais prováveis, com base nos padrões e interações aprendidos.


## Datasets

Para o projeto, serão consideradas duas bases de dados principais conforme tabelas abaixo, constituidas por vídeos de trajetórias humanas no mundo real, com cenários ricos em interações.  A primeira é a BIWI Walking Pedestrians e a segunda a UCY Crowd. Ambos foram convertidos para coordenadas do mundo real em metros, interpolados para obter valores a cada 0,4 segundos.

|Base de Dados | Endereço na Web | Resumo descritivo|
|----- | ----- | -----|
|BIWI Walking Pedestrians Dataset | https://data.vision.ee.ethz.ch/cvl/aem/ewap_dataset_full.tgz | Vista superior de pedestres caminhando em cenários povoados.|

Esta base de dados é composta por duas cenas, a ETH e a Hotel, cujas imagens podem ser observadas nas figuras 02 e 03 respectiviamente.

<div style="text-align: center;">
    <img src="/projetos/HTF/images/biwi_eth.png" alt="Figura 02: Imagem do dataset ETH" width="600"/>
    <p><em>Figura 02: Imagem do dataset Biwi ETH</em></p>
</div>

<div style="text-align: center;">
    <img src="/projetos/HTF/images/biwi_hotel.png" alt="Figura 03: Imagem do dataset HOTEL" width="600"/>
    <p><em>Figura 03: Imagem do dataset Biwi Hotel</em></p>
</div>

|Base de Dados | Endereço na Web | Resumo descritivo|
|----- | ----- | -----|
|UCY Crowd Data | https://graphics.cs.ucy.ac.cy/research/downloads/crowd-data | Conjunto de dados contendo pessoas em movimento em meio a multidões.|

Esta base de dados é composta por seis cenas que são Zara01, Zara02, Zara03, Students001, Students003 e Univ. Exemplos de imagens do Zara01 e Students003  podem ser observadas nas figuras 04 e 05 respectiviamente.

<div style="text-align: center;">
    <img src="/projetos/HTF/images/crowds_zara01.jpg" alt="Figura 04: Imagem do dataset UCY Zara 01" width="600"/>
    <p><em>Figura 04: Imagem do dataset UCY Zara 01</em></p>
</div>

<div style="text-align: center;">
    <img src="/projetos/HTF/images/students_003.jpg" alt="Figura 05:Imagem do dataset UCY Students 03" width="600"/>
    <p><em>Figura 05: Imagem do dataset UCY Students 03</em></p>
</div>

Após tratados, o formato dos datasets será como o disposto na Figura 06, em que a primeira coluna indica o frame do video, a segunda a identificação do pedestre e a terceira e quarta suas coordenadas x e y respectivamente. 

<div style="text-align: center;">
    <img src="/projetos/HTF/images/datazara.png" alt="Figura 06:Estrutura dos dados tratados" width="600"/>
    <p><em>Figura 06: Exemplo de estrutura dos dados tratados</em></p>
</div>


## Workflow

<div style="text-align: center;">
    <img src="/projetos/HTF/images/Workflow.png" alt="Figura 07: width="600"/>
    <p><em>Figura 07: </em></p>
</div>

## Experimentos, Resultados e Discussão dos Resultados

Até o momento, considerar-se que os principais avanços estão vinculados à compreesão do dataset e ao sucesso em conseguir executar o repositório de referência. Entretanto, devido a complexidade do cógido e ao seu grande número de parâmetros, será necessário realizar um estudo de sua estrutura para que a mesma possa ser melhor compreendida. Isso possibilitará verificar parametros e realizar modificações nestes a fim de testar quais impactos eles podem ter nas métricas adotadas. 

<div style="text-align: center;">
    <img src="/projetos/HTF/images/distribuicao01.jpeg" alt="Figura 08: width="600"/>
    <p><em>Figura 08: </em></p>
</div>

<div style="text-align: center;">
    <img src="/projetos/HTF/images/distribuicao02.jpeg" alt="Figura 09: width="600"/>
    <p><em>Figura 09: </em></p>
</div>

Fonte: Social-Implicit: Rethinking Trajectory Prediction Evaluation and The Effectiveness of Implicit Maximum Likelihood Estimation (Abduallah Mohamed, Deyao Zhu, Warren Vu, Mohamed Elhoseiny, Christian Claudel, 2022)*


Levando em conta os modelos pré-treinados disponibilizados no repositório, foram realizadas implementações cujos resultados podem ser observados na Figura 10 em que cada cor representa um pedestre, sendo que as linhas tracejadas representam os dados verdadeiros e as pontilhadas as previsões realizadas.


<div style="text-align: center;">
    <!-- Primeira linha de 3 imagens -->
    <img src="/projetos/HTF/images/Figure_1.png" alt="Imagem 1" width="320"/>
    <img src="/projetos/HTF/images/Figure_2.png" alt="Imagem 2" width="320"/>
    <img src="/projetos/HTF/images/Figure_3.png" alt="Imagem 3" width="320"/>
    <br>
    <!-- Segunda linha de 3 imagens -->
    <img src="/projetos/HTF/images/Figure_4.png" alt="Imagem 4" width="320"/>
    <img src="/projetos/HTF/images/Figure_5.png" alt="Imagem 5" width="320"/>
    <img src="/projetos/HTF/images/Figure_6.png" alt="Imagem 6" width="320"/>
    <p><em>Figura 10: Exemplos de predições comparadas aos dados reais </em></p>
</div>

## Conclusão

Para os próximos passos, espera-se compreender os detalhes da rede SGAN e conduzir novos treinamentos mediante variação de parâmentros a fim de obter resultados que possam ser iguais ou melhores aos fornecidos pelas referências.


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
