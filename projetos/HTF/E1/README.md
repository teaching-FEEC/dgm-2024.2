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

> Utilizando datasets como o ETH e o UCY, os quais contem vídeos com trajetórias de pedestres em meio a multidões, o objetivo é, conforme observa-se na figura abaixo, a partir de um histórico, descrito pelas  linhas sólidas, avaliar o contexto das regras sociais da movimentação humana e predizer trajerias multimodais livres de colisão, descritas pelas linhas tracejadas. 
> 
![objetivo](https://github.com/thbramos/IA376_HTF/blob/main/projetos/HTF/E1/FIG01.png)
>
>Fonte: *Safety-Compliant Generative Adversarial Networks for Human Trajectory Forecasting(Parth Kothari and Alexandre Alahi, 2023)*

> 
> Incluir nessa seção link para vídeo de apresentação da proposta do projeto (máximo 5 minutos).

## Metodologia Proposta

### Base de Dados:

> Neste primeiro momento, para o projeto serão utilizadas as seguintes bases de dados:
> |Base de Dados  | Fonte | Descrição|
> |--|--|--|
> | [ETH Pedestrian](https://paperswithcode.com/dataset/eth) | *ETH*  |O conjunto de dados que contém 1.804 imagens em três videoclipes. Capturado de um equipamento estéreo montado no carro, com uma resolução de 640 x 480 (bayered) e uma taxa de quadros de 13-14 FPS |
> | [UCY Crowds Data](https://graphics.cs.ucy.ac.cy/portfolio) | *UCY GRAPHICS*  | Conjunto de dados em vídeo contendo de pessoas em movimento em meio a multidões|
> | [Stanford Drone Dataset](https://cvgl.stanford.edu/projects/uav_data/) | *STANFORD*  | Conjunto de dados em larga escala que coleta imagens e vídeos de vários tipos de agentes que navegam em um ambiente externo do mundo real, como um campus universitário.|
>


### Arquitetura da Rede Neural:
> Para a primeira entrega, a metodologia proposta deve esclarecer:
> * Qual(is) base(s) de dado(s) o projeto pretende utilizar, justificando a(s) escolha(s) realizadas.
> * Quais abordagens de modelagem generativa o grupo já enxerga como interessantes de serem estudadas.
> * Artigos de referência já identificados e que serão estudados ou usados como parte do planejamento do projeto
> * Ferramentas a serem utilizadas (com base na visão atual do grupo sobre o projeto).
> * Resultados esperados
> * Proposta de avaliação dos resultados de síntese

## Cronograma
> Proposta de cronograma. Procure estimar quantas semanas serão gastas para cada etapa do projeto.

## Referências Bibliográficas
> Apontar nesta seção as referências bibliográficas adotadas no projeto.
