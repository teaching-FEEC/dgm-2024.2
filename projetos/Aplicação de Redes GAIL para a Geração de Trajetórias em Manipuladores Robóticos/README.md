# `Aplicação de Redes GAIL para a Geração de Trajetórias em Manipuladores Robóticos`
# `Application of GAIL Networks for Trajectory Generation in Robotic Manipulators`

## Apresentação

O presente projeto foi originado no contexto das atividades da disciplina de pós-graduação *IA376N - IA generativa: de modelos a aplicações multimodais*, 
oferecida no segundo semestre de 2024, na Unicamp, sob supervisão da Profa. Dra. Paula Dornhofer Paro Costa, do Departamento de Engenharia de Computação e Automação (DCA) da Faculdade de Engenharia Elétrica e de Computação (FEEC).

> |Nome  | RA | Especialização|
> |--|--|--|
> | Maria Fernanda Paulino Gomes  | 206745  | Eng. de Computação|
> | Raisson Leal Silva  | 186273  | Eng. Eletricista|



## Descrição Resumida do Projeto

# Tema e Contexto:
  Um dos desafios no campo da robótica é a geração de trajetórias eficientes e seguras, tanto para robôs móveis, quanto para robôs de base fixa (manipuladores robóticos). Este projeto tem como objetivo utilizar técnicas de Aprendizado por Imitação (IL) e Aprendizado Generativo Adversarial (GAN)
  para abordar o problema de geração de trajetórias válidas para manipuladores robóticos. Inicialmente será utilizado um manipulador de 3DoF (graus de liberdade), e os dados espercialistas serão gerados com base em planejadores de caminhos clássicos, que fornecem trajetórias otimizadas de referência.

# Motivação:
  A capacidade de gerar automaticamente trajetórias válidas e realistas é essencial para melhorar a autonomia de sistemas robóticos em ambientes complexos. Com a evolução dos métodos de machine learning, as Redes Adversárias Generativas (GANs) têm demonstrado grande potencial em imitar comportamentos complexos,
  como a geração de imagens, vídeos e, neste caso, trajetórias de sistemas robóticos. O uso de uma combinação de duas técnicas de machine learning (Imitation Learning e GANs), chamada GAIL (Generative Adversarial Imitation Learning), permite que o modelo aprenda a replicar trajetórias de alta qualidade a partir de exemplos fornecidos por especialistas.

# Objetivo Principal
  O objetivo principal deste projeto é desenvolver um modelo baseado em redes GAIL capaz de gerar trajetórias válidas para um manipulador de 3DoF. O modelo será treinado utilizando trajetórias geradas por planejadores clássicos, que fornecerão dados especialistas, como as configurações de juntas, coordenadas cartesianas (x, y, z, roll, pitch, yaw), quaternions e o rastro da caminho percorrido.

# Saída do Modelo Generativo
  A saída do modelo generativo será um conjunto de trajetórias no formato de configurações de juntas e coordenadas cartesianas (incluindo orientação). Essas trajetórias serão comparadas com as trajetórias fornecidas pelos dados especialistas, e a rede discriminadora verificará se a trajetória gerada é válida e realista. O objetivo final é que o modelo seja capaz de gerar trajetórias válidas.
> Incluir nessa seção link para vídeo de apresentação da proposta do projeto (máximo 5 minutos).

## Metodologia Proposta

# Base de Dados:
  Os dados para este projeto serão gerados por meio de robótica clássica, utilizando planejadores de caminhos para manipuladores robóticos (como o algoritmo de campos potenciais). A escolha de gerar os próprios dados se justifica pelo controle completo sobre as configurrações de trajetória, o que permite garantir a precisão e qualidade dos dados especialistas.
  Esses dados incluirão as configurações de juntas, coordenadas cartesianas, quaternions e o rastro do caminho percorrido pelo manipulador.

# Abordagens de Modelagem Generativa:
  O projeto utilizará uma abordagem baseada em **GAIL (Generative Adversarial Imitation Learning)**, uma variante das GANs adaptada para o aprendizado por imitação. Além da GAIL, outras técnicas de aprendizado por reforço (como PPO) podem ser exploradas para complementar a geração de trajetórias otimizadas.

# Artigos de Referência:

* HO, Jonathan; ERMON, Stefano. Generative adversarial imitation learning. Advances in neural information processing systems, v. 29, 2016. <https://proceedings.neurips.cc/paper_files/paper/2016/file/cc7e2b878868cbae992d1fb743995d8f-Paper.pdf>
* WANG, Haoxu; MEGER, David. Robotic object manipulation with full-trajectory gan-based imitation learning. In: 2021 18th Conference on Robots and Vision (CRV). IEEE, 2021. p. 57-63. <https://ieeexplore.ieee.org/abstract/document/9469449>
* SYLAJA, Midhun Muraleedharan; KAMAL, Suraj; KURIAN, James. Example-driven trajectory learner for robots under structured static environment. International Journal of Intelligent Robotics and Applications, p. 1-18, 2024. <https://link.springer.com/content/pdf/10.1007/s41315-024-00353-y.pdf>
* TSURUMINE, Yoshihisa; MATSUBARA, Takamitsu. Goal-aware generative adversarial imitation learning from imperfect demonstration for robotic cloth manipulation. Robotics and Autonomous Systems, v. 158, p. 104264, 2022. <https://www.sciencedirect.com/science/article/pii/S0921889022001543>
* REN, Hailin; BEN-TZVI, Pinhas. Learning inverse kinematics and dynamics of a robotic manipulator using generative adversarial networks. Robotics and Autonomous Systems, v. 124, p. 103386, 2020. <https://www.sciencedirect.com/science/article/pii/S0921889019303501>

# Ferramentas a serem Utilizadas:

* Python (Jupyter Notebook / Google Colab): para o desenvolvimento de scripts e integração entre os componentes do sistema;
* CoppeliaSim: para a simulação das trajetórias desenvolvidas pelo manipulador robótico;
* PyTorch/TensorFlow: para a implementação das redes GAIL e treinamento dos modelos.

# Resultados Esperados:
  É esperado que o modelo adotado durante o projeto seja capaz de:
  * Imitar trajetórias dentro do workspace do manipulador robótico, respeitando suas restrições cinemáticas, sigando o mesmo padrão de movimentos produzidos pelos planejadores de caminhos clássicos;
  * Gerar caminhos inéditos, oferecendo alternativas de trajetórias viáveis, que não foram apresentadas nos dados especialistas;
  * Reduzir o tempo de planejamento, a geração de trajetórias utilizando o modelo treinado deve ser mais eficiente em termos computacionais do que o uso de planejadores tradicionais, após o processo de aprendizado.

# Proposta de Avaliação dos Resultados:
  As trajetórias obtidas pelo modelo serão avaliadas por meio de verificação da precisão e viabilidade das trajetórias realizadas, quando comparadas com os dados especialistas. Serão utilizadas métricas como distância e diferença de orientação entre as trajetórias geradas.
  Para a visualização das trajetórias geradas, serão feitas simulações, onde por meio da simulação será possível avaliar os resultados obtidos.
    

## Cronograma
> Proposta de cronograma. Procure estimar quantas semanas serão gastas para cada etapa do projeto.

## Referências Bibliográficas

# Artigos de Referência:

* HO, Jonathan; ERMON, Stefano. Generative adversarial imitation learning. Advances in neural information processing systems, v. 29, 2016. <https://proceedings.neurips.cc/paper_files/paper/2016/file/cc7e2b878868cbae992d1fb743995d8f-Paper.pdf>
* WANG, Haoxu; MEGER, David. Robotic object manipulation with full-trajectory gan-based imitation learning. In: 2021 18th Conference on Robots and Vision (CRV). IEEE, 2021. p. 57-63. <https://ieeexplore.ieee.org/abstract/document/9469449>
* SYLAJA, Midhun Muraleedharan; KAMAL, Suraj; KURIAN, James. Example-driven trajectory learner for robots under structured static environment. International Journal of Intelligent Robotics and Applications, p. 1-18, 2024. <https://link.springer.com/content/pdf/10.1007/s41315-024-00353-y.pdf>
* TSURUMINE, Yoshihisa; MATSUBARA, Takamitsu. Goal-aware generative adversarial imitation learning from imperfect demonstration for robotic cloth manipulation. Robotics and Autonomous Systems, v. 158, p. 104264, 2022. <https://www.sciencedirect.com/science/article/pii/S0921889022001543>
* REN, Hailin; BEN-TZVI, Pinhas. Learning inverse kinematics and dynamics of a robotic manipulator using generative adversarial networks. Robotics and Autonomous Systems, v. 124, p. 103386, 2020. <https://www.sciencedirect.com/science/article/pii/S0921889019303501>

# API de Referência:
* Gleave, Adam, Taufeeque, Mohammad, Rocamonde, Juan, Jenner, Erik, Wang, Steven H., Toyer, Sam, Ernestus, Maximilian, Belrose, Nora, Emmons, Scott, Russell, Stuart. (2022). Imitation: Clean Imitation Learning Implementations. [arXiv:2211.11972v1 [cs.LG]](https://arxiv.org/abs/2211.11972). <https://imitation.readthedocs.io/en/latest/index.html>

# Simulador que será Utilizado (CoppeliaSim):
* <https://manual.coppeliarobotics.com/>


