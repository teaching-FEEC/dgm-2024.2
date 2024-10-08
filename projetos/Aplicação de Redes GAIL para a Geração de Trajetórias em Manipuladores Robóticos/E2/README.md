
# `Aplicação de Redes GAIL para a Geração de Trajetórias em Manipuladores Robóticos`
# `Application of GAIL Networks for Trajectory Generation in Robotic Manipulators`

## Apresentação

O presente projeto foi originado no contexto das atividades da disciplina de pós-graduação *IA376N - IA generativa: de modelos a aplicações multimodais*, 
oferecida no segundo semestre de 2024, na Unicamp, sob supervisão da Profa. Dra. Paula Dornhofer Paro Costa, do Departamento de Engenharia de Computação e Automação (DCA) da Faculdade de Engenharia Elétrica e de Computação (FEEC).

> |Nome  | RA | Especialização|
> |--|--|--|
> | Maria Fernanda Paulino Gomes  | 206745  | Eng. de Computação|
> | Raisson Leal Silva  | 186273  | Eng. Eletricista|

## Resumo (Abstract)
Este projeto desenvolve uma abordagem de aprendizado por imitação para o manipulador robótico Kinova Gen3, com foco na geração de trajetórias válidas para tarefas assistivas. A metodologia envolve a teleoperação do robô com joystick, coletando dados sobre posições angulares das juntas, posição cartesiana da garra e sua orientação, que são armazenados em arquivos JSON. Na entrega E2, foram organizados os dados e implementada com sucesso a teleoperação, permitindo a coleta eficiente de dados para o treinamento de uma rede GAIL, preparando o robô para realizar tarefas complexas de forma autônoma.


## Descrição do Problema/Motivação

A proposta inicial era utilizar redes GAIL (Generative Adversarial Imitation Learning) para gerar trajetórias válidas para um manipulador robótico de 3 DoF. Contudo, como um integrante do grupo já trabalhava em paralelo na simulação de um manipulador robótico de 7 DoF, a aplicação foi adaptada para o Kinova Gen3, um manipulador de 7 DoF. O objetivo é fazer com que o robô consiga vestir um paciente, colocando um jaleco cirúrgico. 

A motivação para utilizar redes generativas surgiu da insatisfação com os resultados obtidos no projeto em paralelo, visando aprimorar a eficácia na geração de trajetórias e alcançar resultados satisfatórios. Para a disciplina, o foco é alinhar o jaleco cirúrgico com o braço esquerdo da paciente, conforme ilustrado na imagem.


![trajetória a ser gerada](https://github.com/user-attachments/assets/b19c3f4a-8d2c-442d-92bf-3d4dcd39ee76)


## Objetivo

Este projeto tem como objetivo desenvolver um sistema baseado em redes GAIL (Generative Adversarial Imitation Learning) para gerar trajetórias válidas para o manipulador robótico Kinova Gen3 (7 DoF), automatizando o processo de vestir um paciente com um jaleco cirúrgico de forma eficiente e segura.

### Objetivo Geral
- Gerar trajetórias eficientes e seguras para o manipulador robótico, permitindo a realização autônoma da tarefa de vestir um paciente.

### Objetivos Específicos
- **Coleta de Dados de Teleoperação:** Capturar dados das posições angulares das juntas, posição cartesiana e orientação da garra durante a teleoperação.
- **Treinamento da Rede GAIL:** Utilizar os dados coletados para treinar uma rede GAIL, capaz de replicar as trajetórias observadas.
- **Validação das Trajetórias Geradas:** Avaliar a eficácia e segurança das trajetórias geradas em simulações, garantindo que sejam realizáveis e seguras.


## Metodologia

A metodologia proposta para alcançar os objetivos do projeto envolve as seguintes etapas:

### 1. Coleta de Dados
Neste projeto, será necessário criar um dataset específico para o manipulador robótico Kinova Gen3 de 7 DoF, visto que não serão utilizados datasets prontos. A coleta de dados será realizada por meio de teleoperação no ambiente de simulação, capturando:
- **Posições angulares das juntas** em tempo real durante a teleoperação.
- **Posição cartesiana da garra** (x, y, z).
- **Orientação da garra** representada por [roll, pitch, yaw].

Os dados serão armazenados em arquivos JSON, conforme descrito na seção [**Bases de Dados e Evolução**](#bases-de-dados-e-evolução), e utilizados como referência para o treinamento do modelo.

### 2. Treinamento da Rede GAIL
A abordagem GAIL foi escolhida por sua capacidade de imitar comportamentos complexos, utilizando aprendizado por reforço e aprendizado por imitação. O modelo será treinado para replicar as trajetórias capturadas, seguindo os seguintes passos:
- **Arquitetura da Rede:** Implementação de uma rede neural para o gerador e o discriminador, utilizando frameworks como TensorFlow ou PyTorch.
- **Função de Perda:** A função de perda adversarial será aplicada para medir o desempenho do gerador em relação ao discriminador.

A GAIL se destaca por unir as vantagens do aprendizado por imitação e do aprendizado por reforço, permitindo que o modelo aprenda a replicar trajetórias a partir de demonstrações fornecidas.

### 3. Validação das Trajetórias
Após o treinamento, as trajetórias geradas serão validadas em simulações. As seguintes métricas serão usadas para avaliar a eficácia:
- **Proximidade do Ponto de Contato**: Medição da distância entre a posição prevista da garra e a posição objetivo.
- **Segurança**: Avaliação da trajetória para evitar colisões e garantir a segurança do paciente simulado.

### 4. Metodologia de Avaliação
Os objetivos do projeto serão avaliados através de:
- **Testes de Simulação**: Execução de simulações para verificar se o manipulador consegue realizar a tarefa proposta.
- **Análise Quantitativa**: Avaliação estatística das trajetórias geradas em comparação com as trajetórias desejadas.

### Slides da Apresentação da Segunda Etapa do Projeto

- [Link para os slides](https://docs.google.com/presentation/d/1d-bhS5pN8eW80D_pL8XNPtMwcds4ifCcIb9pqzJnMhk/edit#slide=id.p1)


### Bases de Dados e Evolução

Para este projeto, a base de dados foi montada manualmente, com dados coletados através da teleoperação de um manipulador robótico (usando um joystick) no ambiente de simulação. Inicialmente, as trajetórias eram salvas em múltiplos arquivos JSON, mas esses arquivos foram unificados em um único arquivo JSON para simplificação e uso em treinamentos de Redes Adversárias Generativas de Imitação (GAIL).

| Base de Dados  | Endereço na Web | Resumo Descritivo |
|----------------|-----------------|------------------|
| GAIL_Dataset   | N/A             | Base de dados unificada contendo trajetórias de teleoperação do manipulador robótico, com informações sobre as posições angulares das juntas, a posição cartesiana e a orientação da garra. |


#### Descrição e Análise da Base de Dados

* **Formato:** A base de dados foi consolidada em um único arquivo JSON, contendo múltiplas trajetórias simuladas. Cada registro no dataset é composto por observações e ações associadas ao manipulador robótico.

* **Estrutura:**
  - **Observações:** Cada observação inclui:
    - **Posições angulares das juntas**: 7 valores que representam o estado de cada junta do manipulador.
    - **Posição cartesiana da garra**: 3 valores ([x, y, z]) que indicam a posição da garra no espaço.
    - **Orientação da garra**: 3 valores de rotação (pitch, yaw, roll) que definem a orientação da garra.
  
  - **Ações:** As ações são calculadas como a diferença entre dois estados consecutivos e incluem:
    - **Movimentação do gripper:** Diferença na posição e orientação da garra entre dois estados consecutivos, refletindo as mudanças ocorridas ao longo da trajetória.

* **Tamanho:** O arquivo consolidado contém diversas trajetórias, que anteriormente estavam separadas em vários arquivos JSON, mas foram unificadas para otimização do processamento e treinamento.

* **Tipo de Anotação:** A base de dados contém anotações referentes às trajetórias do manipulador. As observações descrevem as posições e orientações tanto das juntas quanto da garra, enquanto as ações descrevem as mudanças de posição e orientação entre os estados.

* **Transformações e Tratamentos:** As trajetórias foram coletadas diretamente do ambiente de simulação. Para a formatação final, os dados de todas as simulações foram unificados em um único JSON, com observações e ações organizadas para serem processadas por redes adversárias generativas (GAIL). Como a coleta de dados ocorreu em um ambiente controlado, não foi necessário aplicar técnicas de limpeza ou filtragem.

#### Sumário com Estatísticas Descritivas da Base de Estudo

##### Estatísticas das Posições Angulares das Juntas

| Estatística | joint_1    | joint_2    | joint_3    | joint_4    | joint_5    | joint_6    | joint_7    |
|-------------|------------|------------|------------|------------|------------|------------|------------|
| count       | 16051.0000 | 16051.0000 | 16051.0000 | 16051.0000 | 16051.0000 | 16051.0000 | 16051.0000 |
| mean        | 216.9260   | 244.5008   | 155.2496   | 160.6793   | 153.4203   | 163.3758   | 141.0073   |
| std         | 112.8747   | 133.0374   | 123.9732   | 132.8272   | 145.7203   | 80.5145    | 111.4131   |
| min         | 0.0974     | 0.0061     | 0.0048     | 0.0091     | 0.0030     | 33.6910    | 0.0283     |
| 25%         | 99.8671    | 286.0129   | 53.0035    | 46.4387    | 15.2208    | 77.8979    | 57.9813    |
| 50%         | 290.8084   | 297.0499   | 96.9260    | 68.6351    | 78.9634    | 127.7249   | 72.2764    |
| 75%         | 308.5260   | 330.3035   | 312.3682   | 301.6214   | 343.1981   | 243.2315   | 246.2258   |
| max         | 359.7806   | 359.9976   | 359.9915   | 359.9679   | 359.9981   | 348.0733   | 359.9999   |

![distribuicoes_angulares](https://github.com/user-attachments/assets/41d1405f-1809-40f6-b323-0c4290bb5e42)

##### Estatísticas das Posições da Garra

| Estatística | gripper_x   | gripper_y   | gripper_z   |
|-------------|-------------|-------------|-------------|
| count       | 16051.0000  | 16051.0000  | 16051.0000  |
| mean        | 1.3223      | 1.8376      | 0.6307      |
| std         | 0.4224      | 0.0823      | 0.1922      |
| min         | 0.7617      | 1.5202      | 0.1754      |
| 25%         | 0.9824      | 1.7740      | 0.5554      |
| 50%         | 1.0688      | 1.8232      | 0.6301      |
| 75%         | 1.8875      | 1.8843      | 0.7168      |
| max         | 1.9766      | 2.2050      | 1.1005      |

##### Estatísticas das Rotações da Garra

| Estatística | gripper_pitch | gripper_yaw | gripper_roll |
|-------------|---------------|--------------|--------------|
| count       | 16051.0000    | 16051.0000   | 16051.0000   |
| mean        | 107.9481      | 201.8668     | 161.1637     |
| std         | 144.1107      | 50.9297      | 35.6482      |
| min         | 0.0032        | 0.0454       | 0.0585       |
| 25%         | 5.0705        | 174.3852     | 130.1401     |
| 50%         | 37.0476       | 207.8905     | 174.4622     |
| 75%         | 294.0989      | 225.9177     | 180.7627     |
| max         | 359.9729      | 359.0083     | 359.9722     |

##### Estatísticas das Ações da Garra

| Estatística | delta_gripper_x | delta_gripper_y | delta_gripper_z | delta_gripper_pitch | delta_gripper_yaw | delta_gripper_roll |
|-------------|------------------|------------------|------------------|----------------------|--------------------|---------------------|
| count       | 16051.0000       | 16051.0000       | 16051.0000       | 16051.0000           | 16051.0000         | 16051.0000          |
| mean        | -0.0006          | 0.0000           | -0.0000          | 0.1051               | 0.0251             | 0.0385              |
| std         | 0.0066           | 0.0066           | 0.0089           | 19.9207              | 6.5096             | 6.4767              |
| min         | -0.1018          | -0.1503          | -0.2055          | -359.8785            | -358.9630          | -359.8317           |
| 25%         | -0.0002          | -0.0001          | -0.0003          | -0.0124              | -0.0372            | -0.0130             |
| 50%         | 0.0000           | 0.0000           | 0.0000           | -0.0000              | 0.0000             | 0.0000              |
| 75%         | 0.0001           | 0.0001           | 0.0003           | 0.0121               | 0.0470             | 0.0123              |
| max         | 0.0980           | 0.1940           | 0.1729           | 359.8557             | 355.2020           | 359.8180            |

#### Conclusão sobre a Base de Dados

A base de dados construída oferece uma visão completa das trajetórias de teleoperação do manipulador robótico, com dados bem estruturados para treinamento do modelo GAIL. A unificação dos arquivos JSON em um único arquivo facilitou o processo de análise e treinamento, garantindo consistência nos dados. As análises estatísticas descritivas fornecem insights importantes sobre as posições angulares das juntas, a posição cartesiana e a rotação da garra, além das mudanças ocorridas entre os estados, o que será crucial para o sucesso do treinamento da rede GAIL.


### Workflow

Este projeto adota um workflow bem definido para alcançar o objetivo de gerar trajetórias válidas para manipuladores robóticos de 7 graus de liberdade (DoF), utilizando redes GAIL (Generative Adversarial Imitation Learning). O processo foi dividido em seis etapas principais, que cobrem desde a definição do escopo até a validação e testes, conforme ilustrado nas imagens a seguir.


### Imagens de Fluxograma:

#### Figura 1:
![Workflow_Etapas_1-Página-1 drawio (2)](https://github.com/user-attachments/assets/023dc0b2-a584-4356-b72e-9d308da9be10)



#### Figura 2:


![Workflow_Etapas_1-Página-2 drawio (1)](https://github.com/user-attachments/assets/deddb135-14a6-481c-bef8-89004084487d)



## Experimentos, Resultados e Discussão dos Resultados

Até o momento, foram realizados avanços significativos no processamento e organização dos dados para o treinamento do modelo. Utilizando um conjunto de dados coletado e consolidado manualmente, foi realizada uma análise para entender melhor as características das trajetórias de movimentação e ação da garra robótica. As estatísticas descritivas geradas forneceram uma visão das posições angulares das juntas, posições e rotações da garra, bem como as variações (ações) realizadas ao longo das trajetórias.

### Análise dos Dados
A análise inicial revelou padrões importantes, como a variação dos ângulos das juntas e a dispersão das posições e rotações da garra. Essas informações são essenciais para definir a abordagem do modelo de aprendizado por imitação (no caso, GAIL). O processo de normalização de ângulos garantiu que todos os dados estivessem adequadamente preparados para o treinamento, corrigindo valores fora do intervalo esperado (por exemplo, ângulos maiores que 360 graus).

### Próximos Passos: Definição da Arquitetura da Rede Neural
Com a base de dados já estruturada e explorada, o foco atual está na definição da arquitetura da rede neural que será utilizada para treinar o modelo de GAIL. O modelo precisará capturar tanto a dinâmica das posições e rotações da garra quanto as ações que representam as mudanças realizadas em cada iteração.

A definição da arquitetura envolverá decisões sobre:

* Tipo de rede neural: A princípio, planeja-se utilizar redes neurais recorrentes (RNNs), como LSTM ou GRU, que são adequadas para capturar dependências temporais nas sequências de observações e ações.
* Entrada e saída do modelo: As observações das posições angulares, posições e rotações da garra serão usadas como entrada, enquanto as ações serão a saída que o modelo tentará replicar.
* Número de camadas e neurônios: Experimentos serão conduzidos para identificar a quantidade ideal de camadas e neurônios, buscando um equilíbrio entre a capacidade do modelo de generalizar e sua eficiência computacional.

Esses elementos serão ajustados com base em experimentos subsequentes, onde o desempenho do modelo será avaliado em termos de sua capacidade de imitar as trajetórias de forma realista. A partir desses testes, ajustes na arquitetura poderão ser feitos para otimizar o desempenho.

### Discussão
Apesar de ainda estarmos na fase de exploração da arquitetura da rede, os dados já indicam que o modelo terá de lidar com uma alta variabilidade, especialmente nas rotações da garra e nas ações associadas. A diversidade observada nas posições e rotações reforça a necessidade de uma arquitetura robusta, capaz de lidar com essas variações.

Os próximos experimentos irão se concentrar em testar diferentes arquiteturas e parâmetros, e ajustes poderão ser feitos com base nos resultados preliminares, buscando uma melhora no desempenho do modelo.

## Conclusão

Até o momento, o projeto avançou na preparação e análise dos dados de demonstração, gerando estatísticas importantes sobre o comportamento do robô. Também foi realizada uma análise preliminar das arquiteturas de rede para o GAIL, definindo os próximos passos para implementação e testes no ambiente de simulação.

Para a etapa final do projeto, serão realizados os seguintes passos:

* Implementação completa da arquitetura de GAIL.
* Treinamento do agente (robô) no ambiente de simulação, ajustando a política de ações para que o comportamento imite o especialista de forma eficaz.
* Avaliação do desempenho do robô após o treinamento, tanto quantitativamente (medidas de erro ou similaridade com o especialista) quanto qualitativamente (análise visual das execuções do robô).
* Refinamentos adicionais com base nos resultados obtidos e considerações para possíveis melhorias futuras, como a expansão do dataset ou ajustes na estrutura da rede.

Com essas próximas etapas, o projeto avançará para a fase final, onde o foco estará na implementação, treinamento e avaliação do modelo de aprendizado por imitação.


## Referências Bibliográficas
#### Artigos de Referência:

* **GAIL**: Ho, J. & Ermon, S. (2016). Generative Adversarial Imitation Learning. [arXiv:1606.03476](https://arxiv.org/abs/1606.03476).
* WANG, Haoxu; MEGER, David. Robotic object manipulation with full-trajectory gan-based imitation learning. In: 2021 18th Conference on Robots and Vision (CRV). IEEE, 2021. p. 57-63. <https://ieeexplore.ieee.org/abstract/document/9469449>
* SYLAJA, Midhun Muraleedharan; KAMAL, Suraj; KURIAN, James. Example-driven trajectory learner for robots under structured static environment. International Journal of Intelligent Robotics and Applications, p. 1-18, 2024. <https://link.springer.com/content/pdf/10.1007/s41315-024-00353-y.pdf>
* TSURUMINE, Yoshihisa; MATSUBARA, Takamitsu. Goal-aware generative adversarial imitation learning from imperfect demonstration for robotic cloth manipulation. Robotics and Autonomous Systems, v. 158, p. 104264, 2022. <https://www.sciencedirect.com/science/article/pii/S0921889022001543>
* REN, Hailin; BEN-TZVI, Pinhas. Learning inverse kinematics and dynamics of a robotic manipulator using generative adversarial networks. Robotics and Autonomous Systems, v. 124, p. 103386, 2020. <https://www.sciencedirect.com/science/article/pii/S0921889019303501>

#### API de Referência:
* Gleave, Adam, Taufeeque, Mohammad, Rocamonde, Juan, Jenner, Erik, Wang, Steven H., Toyer, Sam, Ernestus, Maximilian, Belrose, Nora, Emmons, Scott, Russell, Stuart. (2022). Imitation: Clean Imitation Learning Implementations. [arXiv:2211.11972v1 [cs.LG]](https://arxiv.org/abs/2211.11972). <https://imitation.readthedocs.io/en/latest/index.html>

#### Simulador que será Utilizado (RCareWorld - Unity):
* **RCareWorld**: <https://github.com/empriselab/RCareWorld>

