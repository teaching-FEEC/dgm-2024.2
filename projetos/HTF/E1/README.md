# `<Redes Adversárias Generativas Sociais para previsão de trajetória humana>`
# `<Social Generative Adversarial Networks for Human Trajectory Forecasting>`

## Apresentação

O presente projeto foi originado no contexto das atividades da disciplina de pós-graduação *IA376N - IA generativa: de modelos a aplicações multimodais*, 
oferecida no segundo semestre de 2024, na Unicamp, sob supervisão da Profa. Dra. Paula Dornhofer Paro Costa, do Departamento de Engenharia de Computação e Automação (DCA) da Faculdade de Engenharia Elétrica e de Computação (FEEC).

> Incluir nome RA e foco de especialização de cada membro do grupo. Os grupos devem ter no máximo três integrantes.
> |Nome  | RA | Especialização|
> |--|--|--|
> | Hiuri Santana de Noronha  | 229961  | Eng. Eletricista|
> | Thiago Belina Silva Ramos  | 203975  | Eng. Eletricista|


## Descrição Resumida do Projeto
> Este projeto tem como objetivo a implementação e a avaliação dos resultados da aplicação de Redes Generativas Adversárias Sociais, SGANs, para predição de trajetórias humanas em multidões. O seu principal desafio
> é modelar as interações sociais garantindo que as previsões realizadas sejam seguras e realistas.
>
> Prever trajetórias humanas é fundamental para garantir a segurança, melhorar a modelagem de interações e aprimorar a funcionalidade de vários sistemas em aplicações do mundo real, tais como interações entre seres 
> humanos e máquinas, sejam elas veiculares ou robos, segurança e monitoramento, planejamento urbano, gerenciamento de tráfego, dentre outras. Os avanços dessa área podem levar a melhorias significativas em como 
> os sistemas autônomos operam em ambientes povoados por humanos.

> Utilizando datasets como o ETH , o UCY e o Stanford Drone os quais contem vídeos com trajetórias de pedestres em meio a multidões, o objetivo é, conforme observa-se na figura abaixo, a partir de um histórico, descrito pelas linhas sólidas, avaliar o contexto das regras sociais da movimentação humana e predizer trajerias multimodais livres de colisão, descritas pelas linhas tracejadas. 
> 
![objetivo](https://github.com/thbramos/IA376_HTF/blob/main/projetos/HTF/E1/FIG01.png)
>
>Fonte: *Safety-Compliant Generative Adversarial Networks for Human Trajectory Forecasting(Parth Kothari and Alexandre Alahi, 2023)*

> 
> Incluir nessa seção link para vídeo de apresentação da proposta do projeto (máximo 5 minutos).

## Metodologia Proposta

A ideia inicial é replicar o trabalho proposto em [1], em que foi desenvolvida uma versão aprimorada da SGAN, denominada pelo autor de SGANv2, e se possivel, propor alterações na estrutura que melhorem os seus resultados.

## Base de Dados:

> Neste primeiro momento, serão consideradas as seguintes bases de dados:
> |Base de Dados  | Fonte | Descrição|
> |--|--|--|
> | [ETH Pedestrian](https://paperswithcode.com/dataset/eth) | *ETH*  |O conjunto de dados que contém 1.804 imagens em três videoclipes. Capturado de um equipamento estéreo montado no carro, com uma resolução de 640 x 480 (bayered) e uma taxa de quadros de 13-14 FPS. |
> | [UCY Crowds Data](https://graphics.cs.ucy.ac.cy/portfolio) | *UCY GRAPHICS*  | Conjunto de dados em vídeo contendo de pessoas em movimento em meio a multidões.|
> | [Stanford Drone Dataset](https://cvgl.stanford.edu/projects/uav_data/) | *STANFORD*  | Conjunto de dados em larga escala que coleta imagens e vídeos de vários tipos de agentes que navegam em um ambiente externo do mundo real, como um campus universitário.|

## Arquitetura de Rede Neural:

## Artigos de Referência:

## Ferramentas a serem utilizadas:

##  Proposta de Avaliação dos resultados:

> Para avaliação dos resultados será utilizada a métrica Top-K, método através do qual são gerados potenciais caminhos futuros e a saída do modelo são as top k trajetórias mais prováveis baseado nos padrões e
> interações aprendidos. Então serão avaliadas:

> •	O Erro de Deslocamento Médio Top-K (ADE - Average Displacement Error) mede a distância média entre as trajetórias previstas e de referência em relação às k melhores previsões.  Ela fornece uma visão de quão próximas
> as trajetórias previstas estão das trajetórias reais dos pedestres, mas não leva em consideração as interações entre pedestres.

> •	O Erro de Deslocamento Final Top-K (FDE - Final Displacement Error), de maneira semelhante ao ADE, foca na posição final das trajetórias previstas em comparação com a de referência. É útil para avaliar a precisão das
> previsões do modelo ao fim do horizonte de previsão, mas também é afetado pelas mesmas limitações que a métrica anterior quando utilizado isoladamente.

> •	A Taxa de Colisão avalia a porcentagem de trajetórias previstas que resultam em colisões com outros pedestres. Trata-se de uma medida crítica de segurança, pois taxas de colisão elevadas indicam que o modelo não está
> gerando trajetórias socialmente aceitáveis. Esta deve ser utilizada em conjunto com as anteriores para suprir as limitações especificadas.

> Além disso, é preferível a escolha de valores menores de possibilidade k a fim de se obter melhores estimativas de probabilidade em modelos generativos implícitos. Trata-se de uma abordagem que permite uma avaliação um
> pouco mais sutil da capacidade do modelo de prever trajetórias socialmente compatíveis, levando em consideração as interações entre os pedestres. Utilizando-se em conjunto as métricas apresentadas, espera-se obter
> avaliações abrangentes de eficácia e segurança do modelo.


## Cronograma
> Proposta de cronograma. Procure estimar quantas semanas serão gastas para cada etapa do projeto.

## Referências Bibliográficas

> [1] P. Kothari e A. Alahi, "Safety-Compliant Generative Adversarial Networks for Human Trajectory Forecasting," IEEE Transactions on Intelligent Transportation Systems, vol. PP, pp. 1-11, abr. 2023.
>  
> [2] A. Gupta, J. Johnson, L. Fei-Fei, S. Savarese, e A. Alahi, "Social GAN: Socially Acceptable Trajectories with Generative Adversarial Networks," IEEE Conference on Computer Vision and Pattern Recognition (CVPR), 2018.
> 
> [3] Z. Lv, X. Huang, e W. Cao, "An improved GAN with transformers for pedestrian trajectory prediction models," International Journal of Intelligent Systems, vol. 37, nº 8, pp. 4417-4436, 2022.
> 
> [4] S. Eiffert, K. Li, M. Shan, S. Worrall, S. Sukkarieh and E. Nebot, "Probabilistic Crowd GAN: Multimodal Pedestrian Trajectory Prediction Using a Graph Vehicle-Pedestrian Attention Network," in IEEE Robotics and Automation Letters, vol. 5, no. 4, pp. 5026-5033, Oct. 2020.
