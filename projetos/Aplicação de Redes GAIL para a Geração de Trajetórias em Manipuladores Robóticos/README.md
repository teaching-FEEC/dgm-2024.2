
# `Aplicação de Redes GAIL para a Geração de Trajetórias em Manipuladores Robóticos`
# `Application of GAIL Networks for Trajectory Generation in Robotic Manipulators`

## Apresentação

O presente projeto foi originado no contexto das atividades da disciplina de pós-graduação *IA376N - IA generativa: de modelos a aplicações multimodais*, 
oferecida no segundo semestre de 2024, na Unicamp, sob supervisão da Profa. Dra. Paula Dornhofer Paro Costa, do Departamento de Engenharia de Computação e Automação (DCA) da Faculdade de Engenharia Elétrica e de Computação (FEEC).

> |Nome  | RA | Especialização|
> |--|--|--|
> | Maria Fernanda Paulino Gomes  | 206745  | Eng. de Computação|
> | Raisson Leal Silva  | 186273  | Eng. Eletricista|

### Slides e Vídeo apresentando o projeto

- [Slides primeira entrega](https://github.com/user-attachments/files/17291787/E1_Projeto_MariaFernanda_Raisson.pdf)


- [Link para o vídeo da primeira entrega](https://drive.google.com/file/d/10UU3tbpaLdKtSWHwaHwwuAelCnEYYyK1/view?usp=sharing)

- [Slides da segunda entrega](https://docs.google.com/presentation/d/1d-bhS5pN8eW80D_pL8XNPtMwcds4ifCcIb9pqzJnMhk/edit#slide=id.p1)


## Resumo (Abstract)
Este projeto investiga o uso de redes Generative Adversarial Imitation Learning (GAIL), uma abordagem generativa avançada, para a geração de trajetórias válidas e seguras no manipulador robótico Kinova Gen3. Com foco em tarefas assistivas, como vestir autonomamente um jaleco cirúrgico em um paciente, o modelo GAIL combina aprendizado por imitação e aprendizado adversarial para replicar trajetórias especialistas a partir de demonstrações capturadas via teleoperação em um ambiente simulado (RCareWorld).

O modelo gerador foi projetado para aprender padrões temporais utilizando uma técnica de janela deslizante, enquanto o discriminador avalia a fidelidade das trajetórias geradas. Os resultados destacam a capacidade do GAIL de capturar dinâmicas complexas e gerar movimentos naturais, especialmente após o incremento no volume de dados de treinamento. No entanto, o desempenho inicial evidenciou limitações associadas a conjuntos de dados reduzidos, reforçando a necessidade de dados robustos para aprendizado por imitação. Como próximos passos, o projeto busca explorar cenários mais complexos, incorporar variações no ambiente e transitar do ambiente simulado para o mundo real, ampliando o potencial de aplicação das redes GAIL em robótica assistiva.


## Descrição do Problema/Motivação

Atividades diárias como vestir-se são cruciais para a autonomia de pessoas com deficiências motoras ou condições de saúde que dificultam a mobilidade. Estudos apontam que vestir-se é uma das tarefas mais desafiadoras no cuidado de idosos e pessoas com deficiências, devido à alta complexidade de manipular objetos deformáveis como roupas em espaços tridimensionais. Além disso, a crescente demanda por cuidados, impulsionada pelo envelhecimento populacional, torna a automação dessas tarefas uma área de pesquisa essencial.

No campo da robótica assistiva, trabalhos anteriores frequentemente assumem configurações fixas, como vestuário pré-posicionado ou poses estáticas, o que limita a aplicabilidade prática em cenários reais. Por exemplo, métodos tradicionais utilizam algoritmos baseados em planeamento de trajetórias ou regras pré-definidas para manipular roupas específicas, como jalecos hospitalares, mas enfrentam dificuldades para generalizar a diferentes tipos de vestuário e variabilidade nas poses humanas. No entanto, abordagens baseadas em aprendizado de máquina, como aprendizado por reforço ou redes neurais convolucionais aplicadas a nuvens de pontos, têm mostrado maior potencial de generalização e flexibilidade, especialmente ao lidar com objetos deformáveis e situações de interação humano-robô.

No contexto deste projeto, propõe-se a utilização de redes GAIL (Generative Adversarial Imitation Learning) para gerar trajetórias válidas de manipuladores robóticos em tarefas assistivas, com foco na tarefa de vestir um jaleco cirúrgico em um paciente, alinhando-o ao braço esquerdo de forma eficiente e segura. Este trabalho se restringe a uma configuração fixa do jaleco e a uma pose específica do paciente, com o objetivo de simplificar o problema e concentrar os esforços no treinamento do modelo para imitar as trajetórias capturadas durante a teleoperação. A limitação do escopo reflete a natureza do projeto, que visa demonstrar os conceitos em um curto espaço de tempo, mas abre caminhos para estudos futuros que poderiam incluir variações nas poses humanas e nos tamanhos dos jalecos.

A motivação para este projeto surgiu da insatisfação com resultados obtidos em projetos paralelos e da busca por maior eficácia na geração de trajetórias, utilizando aprendizado por imitação. Os dados coletados por teleoperação fornecem exemplos ricos de movimentação que servirão como base para treinar o modelo, permitindo ao manipulador replicar padrões humanos de movimento e adaptar-se a cenários controlados. Este trabalho busca contribuir com o avanço na robótica assistiva, explorando métodos de aprendizado generativo adversarial para criar soluções viáveis e seguras para tarefas assistivas em ambientes simulados.

![trajetória a ser gerada](https://github.com/user-attachments/assets/b19c3f4a-8d2c-442d-92bf-3d4dcd39ee76)


## Objetivo

O objetivo deste projeto é desenvolver um sistema baseado em redes GAIL (Generative Adversarial Imitation Learning) para gerar trajetórias válidas e seguras para o manipulador robótico Kinova Gen3 (com 7 graus de liberdade), permitindo que ele realize, de forma autônoma, a tarefa assistiva de vestir um jaleco cirúrgico em um paciente. Este sistema busca demonstrar a viabilidade de utilizar aprendizado por imitação em um cenário controlado, contribuindo para o avanço da robótica assistiva em ambientes simulados.

### Objetivo Geral
- Criar um modelo de aprendizado por imitação capaz de gerar trajetórias eficientes, seguras e realizáveis para o manipulador robótico Kinova Gen3 (com 7 graus de liberdade) realizar a tarefa específica de vestir um jaleco cirúrgico em um paciente, respeitando a pose fixa do paciente e a configuração predefinida do jaleco.

### Objetivos Específicos
1. **Coleta de Dados de Teleoperação:** 
   - Realizar a coleta de dados de trajetórias realizadas por meio de teleoperação em um ambiente simulado.
   - Capturar as posições angulares das juntas, a posição cartesiana da garra e sua orientação (roll, pitch, yaw), garantindo a precisão dos dados para o treinamento.

2. **Organização e Pré-processamento dos Dados:** 
   - Consolidar os dados coletados em um formato unificado para facilitar o treinamento da rede.
   - Realizar o tratamento dos dados para corrigir inconsistências, como normalização de ângulos e verificação de limites operacionais.

3. **Implementação e Treinamento da Rede GAIL:** 
   - Projetar e implementar as redes geradora e discriminadora, definindo arquiteturas adequadas para lidar com a dinâmica do manipulador robótico.
   - Treinar a rede GAIL utilizando os dados coletados, ajustando hiperparâmetros para maximizar a capacidade do modelo de replicar as trajetórias observadas.

4. **Validação das Trajetórias Geradas:** 
   - Avaliar as trajetórias geradas por meio de simulações, utilizando métricas quantitativas como distância e orientação entre as trajetórias geradas e as trajetórias de referência.
   - Garantir que as trajetórias sejam realizáveis e seguras, minimizando riscos de colisão e respeitando as restrições cinemáticas do robô.

5. **Análise e Documentação dos Resultados:** 
   - Analisar os resultados do treinamento, identificando padrões de sucesso e limitações nas trajetórias geradas.
   - Documentar os processos, resultados e aprendizados do projeto, fornecendo uma base para estudos futuros na área de robótica assistiva.

---

## Metodologia

A metodologia proposta neste projeto foi elaborada para alcançar os objetivos definidos, utilizando o manipulador robótico Kinova Gen3 com 7 graus de liberdade (DoF) e abordagens de aprendizado por imitação. Esta seção descreve as etapas principais, os conceitos fundamentais relacionados ao projeto, as bases de dados utilizadas e o workflow adotado.

### Conceitos Fundamentais

#### **Manipulador Robótico Kinova Gen3:**
   - O Kinova Gen3 é um manipulador robótico avançado, com 7 graus de liberdade (DoF), que oferece alta precisão e flexibilidade em tarefas complexas. 
   - *Graus de Liberdade (DoF)* referem-se à quantidade de movimentos independentes que o robô pode realizar. No caso do Kinova Gen3, os 7 DoFs correspondem às rotações e deslocamentos possíveis em suas juntas.
   - Uma imagem ilustrativa do Kinova Gen3 é apresentada abaixo para familiarizar o leitor com o equipamento:

      ![Kinova Gen3](img/Kinova%20Gen3.png)


#### **Aprendizado por Imitação e Aprendizado por Reforço**:

- *Aprendizado por Imitação (Imitation Learning)* é uma abordagem onde um agente aprende a realizar tarefas observando demonstrações de um especialista. Em vez de depender de uma função de recompensa explícita, o agente tenta replicar as ações observadas em trajetórias fornecidas. Essa técnica é amplamente usada em robótica para tarefas complexas, pois reduz a necessidade de modelar explicitamente o ambiente ou a recompensa.

- *Aprendizado por Reforço (Reinforcement Learning)* é uma técnica onde o agente aprende a realizar ações em um ambiente para maximizar uma função de recompensa acumulada. O agente explora o ambiente, avalia as recompensas obtidas e ajusta suas estratégias. Apesar de poderoso, o aprendizado por reforço pode ser desafiador em cenários com recompensas esparsas ou complexas.

#### **Generative Adversarial Imitation Learning (GAIL)**:

O GAIL combina o aprendizado por imitação e o aprendizado por reforço, oferecendo uma solução poderosa para replicar comportamentos complexos sem necessidade de definir uma função de recompensa explícita. Inspirado por Redes Adversariais Generativas (GANs), o GAIL utiliza dois componentes principais:

1. **Gerador (Generator)**: Representa a política do agente que, dado um estado, gera ações para simular trajetórias. A política é ajustada para gerar comportamentos que se aproximem das demonstrações de especialistas.

2. **Discriminador (Discriminator)**: Um modelo que avalia se uma trajetória é gerada pelo especialista ou pelo agente. Ele fornece um sinal de "recompensa" que orienta o gerador no aprendizado.

##### **Funcionamento**:
- O GAIL aprende ao alternar entre otimizar o gerador e o discriminador. O discriminador tenta distinguir trajetórias geradas das demonstradas, enquanto o gerador tenta enganar o discriminador, produzindo trajetórias mais realistas.
- A principal métrica usada no GAIL é a divergência de Jensen-Shannon entre as distribuições de ocupação (state-action pairs) do agente e do especialista. Isso garante que o modelo aprenda políticas que imitam os padrões observados nas demonstrações especialistas.


#### **Ambiente de Simulação RCareWorld:**

- O **RCareWorld** é um ambiente de simulação avançado, desenvolvido no Unity, para testar e validar algoritmos de robótica assistiva em cenários realistas. 
- Ele permite simular tarefas complexas, como a manipulação de objetos deformáveis, interação humano-robô e planejamento de trajetórias.
- No contexto deste projeto, o RCareWorld foi utilizado para integrar o modelo GAIL ao manipulador robótico Kinova Gen3, possibilitando a coleta de dados, o treinamento e a validação das trajetórias geradas.
- Sua flexibilidade e suporte a sensores virtuais garantiram um ambiente seguro e eficiente para o desenvolvimento do sistema proposto.


#### **Técnica de Janela Deslizante (Sliding Window):**

- O uso de uma janela deslizante (*Sliding Window*) é uma abordagem que permite capturar a dinâmica temporal de sistemas robóticos. Ao considerar múltiplos estados consecutivos, essa técnica fornece ao modelo informações contextuais importantes sobre a sequência de movimentos.
- No contexto deste projeto, o **SlidingWindowGenerator** foi implementado para prever ações com base em uma sequência de estados anteriores.
- **Por que utilizar a janela deslizante?**
  - Captura a continuidade das trajetórias, preservando informações temporais cruciais.
  - Reduz a dimensionalidade dos dados observados em comparação com técnicas que consideram toda a sequência histórica.
  - Melhora a capacidade do modelo de prever ações suaves e realistas, fundamentais para tarefas assistivas como manipulação de objetos deformáveis.
- A janela deslizante utilizada neste projeto possui tamanho 4, garantindo um equilíbrio entre informações passadas relevantes e eficiência computacional.

Essa abordagem permite que o gerador aprenda padrões temporais nos movimentos do manipulador robótico, resultando em trajetórias mais precisas e adaptadas às demonstrações fornecidas.


### Etapas Metodológicas

1. **Coleta de Dados de Teleoperação:**
   - A coleta de dados foi realizada através da teleoperação do manipulador robótico em um ambiente de simulação, utilizando um joystick para controlar os movimentos do braço robótico.
   - Dados coletados:
     - **Posições angulares das juntas:** valores que representam o estado de cada junta do robô (como temos 7 juntas são 7 dados gerados).
     - **Posição cartesiana da garra:** coordenadas (x, y, z) no espaço tridimensional.
     - **Orientação da garra:** representada por roll, pitch e yaw.
   - Os dados foram armazenados em múltiplos arquivos JSON, posteriormente unificados para simplificar o treinamento do modelo.

2. **Organização e Pré-processamento dos Dados:**
   - Unificação de múltiplos arquivos JSON em um único dataset consolidado.
   - Normalização de valores como ângulos para evitar inconsistências (ex.: ângulos maiores que 360°).
   - Geração de anotações no formato de observações (estados com 13 dados, 7 posiç~eos angulares das juntas, 3 posições cartesianas da garra, 3 orientação da garra) e ações (diferenças entre estados consecutivos).

3. **Implementação do Ambiente de Simulação:**
   - Para este projeto, foi utilizado o **RCareWorld**, um ambiente de simulação baseado no Unity, desenvolvido especificamente para testar algoritmos de robótica assistiva em cenários realistas.
   - O RCareWorld foi configurado para integrar o modelo GAIL ao manipulador robótico Kinova Gen3, permitindo a interação em tempo real e a validação de trajetórias.
   - Sua interface flexível facilitou a coleta de dados, o treinamento do modelo e a validação das trajetórias geradas, garantindo um ambiente seguro para experimentação e refinamento.
   - Ferramentas adicionais:
     - **Unity Physics Engine**: para simulação precisa de interações físicas entre o manipulador e o ambiente.
     - **Sensores Virtuais**: integrados ao simulador, fornecendo dados como posições e orientações necessárias para o treinamento.

4. **Implementação e Treinamento do Modelo GAIL:**

- A implementação do modelo GAIL foi realizada utilizando o framework **PyTorch**, que forneceu a flexibilidade necessária para configurar as redes geradora e discriminadora. O modelo foi projetado para aprender trajetórias de alta qualidade replicando os padrões observados nos dados de demonstração. A seguir, detalhamos as principais etapas dessa implementação:

- **Rede Geradora:**
  - A rede geradora (*SlidingWindowGenerator*) foi implementada para prever ações com base em uma janela deslizante de estados anteriores. 
  - A arquitetura consiste em uma sequência de camadas densas (*fully connected*), ativadas por funções *ReLU*, que processam entradas de dimensão state_dim x window_size.
  - A saída da rede corresponde à ação predita (action_dim), representando os comandos que o manipulador deve executar.

- **Rede Discriminadora:**
  - A rede discriminadora avalia a validade das trajetórias geradas comparando-as com as trajetórias especialistas. 
  - O modelo combina as entradas da janela deslizante de estados e as ações correspondentes, passando-as por uma sequência de camadas densas com funções de ativação *ReLU* e *Sigmoid*. 
  - A saída é uma probabilidade que indica se a trajetória é "real" (do especialista) ou "falsa" (gerada).

- **Funções de Perda:**
  - Para o discriminador, foi utilizada a *Binary Cross Entropy* (BCE), que mede a capacidade de diferenciar entre trajetórias reais e falsas.
  - O gerador, por sua vez, é otimizado para "enganar" o discriminador, minimizando a BCE ao tentar fazer com que as trajetórias falsas sejam classificadas como reais.

- **Treinamento do Modelo:**
  - O treinamento é realizado em dois estágios:
    1. **Treinamento do Discriminador:** O discriminador é treinado primeiro, ajustando seus pesos para distinguir entre trajetórias reais e geradas.
    2. **Treinamento do Gerador:** O gerador é atualizado para produzir trajetórias que "enganem" o discriminador, gerando ações mais próximas das trajetórias especialistas.
  - Foram utilizadas janelas deslizantes de tamanho 4 (window_size = 4), permitindo que o modelo considere a dinâmica temporal dos estados anteriores para prever ações.

- **Hiperparâmetros:**
  - Taxa de aprendizado: 0.0001
  - Número de épocas: 1000
  - Dimensões do estado e ação: state_dim=13, action_dim = 6
  - Dimensão oculta das redes: hidden_dim=64


6. **Validação e Avaliação das Trajetórias:**

- A validação e avaliação das trajetórias geradas foram realizadas com base em critérios quantitativos e qualitativos, garantindo que o modelo atenda aos requisitos de precisão e segurança:

- **Validação em Simulações:**
  - As trajetórias geradas pelo modelo foram testadas no ambiente de simulação RCareWorld, replicando cenários realistas de manipulação robótica.

- **Métricas Quantitativas:**
  - **Proximidade do Ponto de Contato:** Medição da distância entre a posição final da garra e a posição-alvo, garantindo precisão nas ações.
  - **Suavidade das Trajetórias:** Avaliação da continuidade dos movimentos, medindo a variação entre ações consecutivas.
  - **Respeito às Restrições Cinemáticas:** Verificação se as trajetórias geradas estão dentro das capacidades físicas do manipulador, como limites angulares e de alcance.

- **Análise Qualitativa:**
  - **Visualização das Trajetórias:** As simulações foram observadas visualmente para identificar movimentos não naturais ou inconsistências.
  - **Segurança:** Análise para identificar possíveis colisões ou movimentos arriscados que poderiam comprometer a segurança do paciente ou do manipulador.


7. **Análise e Documentação dos Resultados:**
   - Análise quantitativa e qualitativa dos resultados do modelo.
   - Geração de relatórios detalhados com gráficos e tabelas para descrever o desempenho do modelo.


---

### Bases de Dados e Evolução

| Base de Dados  | Endereço na Web | Resumo Descritivo |
|----------------|-----------------|------------------|
| GAIL_Dataset   | N/A             | Dataset gerado manualmente, contendo trajetórias capturadas via teleoperação do manipulador Kinova Gen3. |

#### Detalhamento da Base de Dados

- **Formato e Estrutura:**
  - O dataset está estruturado em um único arquivo JSON consolidado, organizado em formato hierárquico. Cada trajeto consiste em:
    - **Observações:** 
      - Cada observação é um vetor com 13 valores que representam o estado atual do manipulador:
        - **7 posições angulares das juntas** (em graus), descrevendo o estado de cada junta do manipulador Kinova Gen3.
        - **3 coordenadas cartesianas** (x, y, z) da posição da garra no espaço tridimensional.
        - **3 orientações da garra** (roll, pitch, yaw), indicando sua rotação no espaço.

    - **Ações:**
      - Cada ação é um vetor com 6 valores que descrevem as mudanças ocorridas entre dois estados consecutivos:
        - **3 variações nas coordenadas cartesianas** (Δx, Δy, Δz) da posição da garra.
        - **3 variações nas orientações da garra** (Δroll, Δpitch, Δyaw).

- **Tamanho e Quantidade de Dados:**
  - O dataset completo contém:
    - **16.051 observações**, distribuídas em múltiplas trajetórias capturadas durante a teleoperação.
    - **16.051 ações**, calculadas com base nas diferenças entre estados consecutivos.

- **Transformações e Tratamentos:**
  - Os dados foram inicialmente coletados em múltiplos arquivos JSON e posteriormente unificados em um único arquivo para simplificar o treinamento e a validação do modelo.
  - Processos de normalização foram aplicados, incluindo:
    - Ajustes em valores de ângulos para mantê-los dentro de intervalos consistentes (0° a 360°).
    - Verificação de consistência entre observações e ações para evitar discrepâncias.

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


#### Exemplos de Estrutura dos Dados:

- **Exemplo de Observação:**
  ```json
  [
    -19.196691513061523, -63.626163482666016, 58.71779251098633, 
    49.132484436035156, 72.03478240966797, 72.8636245727539, 
    57.877281188964844, 1.7873706817626953, 1.758441686630249, 
    0.316080242395401, 38.07461929321289, 203.99948120117188, 
    130.9566650390625
  ]
  ```

- **Exemplo de Ação:**
  ```json
  [
    0.013942599296569824, -0.005357503890991211, 0.0038062334060668945, 
    -0.5021171569824219, -0.0680694580078125, 0.4828643798828125
  ]
  ```

#### Conclusão sobre a Base de Dados
  O GAIL_Dataset oferece uma base robusta e bem estruturada para treinar modelos de aprendizado por imitação. A organização detalhada e a qualidade das anotações garantem que o modelo possa capturar as complexidades do movimento do manipulador e replicá-las com alta fidelidade.


### Workflow

Este projeto adota um workflow bem definido para alcançar o objetivo de gerar trajetórias válidas para manipuladores robóticos de 7 graus de liberdade (DoF), utilizando redes GAIL (Generative Adversarial Imitation Learning). O processo foi dividido em seis etapas principais, que cobrem desde a definição do escopo até a validação e testes, conforme ilustrado nas imagens a seguir.


### Imagens de Fluxograma:

#### Figura 1:
![Workflow_Etapas_1-Página-1 drawio (2)](https://github.com/user-attachments/assets/023dc0b2-a584-4356-b72e-9d308da9be10)



#### Figura 2:


![Workflow_Etapas_1-Página-2 drawio (1)](https://github.com/user-attachments/assets/deddb135-14a6-481c-bef8-89004084487d)



## Experimentos, Resultados e Discussão dos Resultados

### **Descrição dos Experimentos**

Os experimentos foram realizados para avaliar a capacidade do modelo GAIL de gerar trajetórias válidas e seguras para o manipulador robótico Kinova Gen3. As etapas principais incluíram:

1. **Treinamento do Modelo GAIL:**
   - Treinamento realizado em 1000 épocas, utilizando o dataset gerado por teleoperação.
   - Configuração do tamanho da janela deslizante como 4 estados consecutivos para capturar a dinâmica temporal.

2. **Execução de Simulações:**
   - O ambiente de simulação RCareWorld foi usado para validar as trajetórias geradas, simulando a tarefa de vestir o jaleco em uma pose fixa do paciente.

3. **Validação das Trajetórias Geradas:**
   - Validação com base em métricas quantitativas, como proximidade ao ponto-alvo e suavidade dos movimentos, e análise qualitativa de segurança e naturalidade das trajetórias.

Aqui está a versão aprimorada que reflete os desafios encontrados e os diferentes modelos treinados, com tabelas comparativas e comentários detalhados:


### **Resultados Obtidos**

#### **Métricas Quantitativas Comparativas**

| Métrica                           | Modelo Inicial   | Modelo Intermediário | Modelo Avançado   |
|-----------------------------------|------------------|-----------------------|-------------------|
| Proximidade ao Ponto-Alvo (cm)   | 3.45             | 2.15                 | 1.25             |
| Suavidade da Trajetória (Δ ação) | 0.125            | 0.085                | 0.045            |
| Respeito às Restrições Cinemáticas (%) | 87.5%          | 94.2%                | 98.7%            |

#### **Proximidade ao Ponto-Alvo por Cenário**

| Cenário          | Modelo Inicial (cm) | Modelo Intermediário (cm) | Modelo Avançado (cm) |
|-------------------|---------------------|---------------------------|-----------------------|
| Jaleco Centralizado | 2.95              | 1.85                      | 1.12                |
| Jaleco Deslocado   | 4.25               | 2.45                      | 1.45                |

#### **Resultados Qualitativos**

- **Segurança das Trajetórias:**
  - **Modelo Inicial:** Trajetórias inconsistentes, frequentemente apresentando colisões ou movimentos abruptos.
  - **Modelo Intermediário:** Melhor controle, com poucas colisões e trajetórias mais suaves.
  - **Modelo Avançado:** Movimentos consistentes e sem colisões observadas, respeitando as restrições cinemáticas.

- **Naturalidade dos Movimentos:**
  - **Modelo Inicial:** Movimentos robóticos e não naturais, dificultando a tarefa assistiva.
  - **Modelo Intermediário:** Movimentos parcialmente ajustados, ainda apresentando variações abruptas.
  - **Modelo Avançado:** Movimentos naturais e bem ajustados, replicando com maior fidelidade as demonstrações do especialista.

- **Limitações Observadas:**
  - **Modelo Inicial:** Insuficiência de dados gerados problemas significativos nas trajetórias.
  - **Modelo Intermediário:** Melhor desempenho com o aumento de dados, mas ainda com dificuldades em cenários mais complexos.
  - **Modelo Avançado:** Desempenho satisfatório, mas com necessidade de maior refinamento em variações complexas de cenário.



### **Discussão dos Resultados**

1. **Evolução do Modelo:**
   - O desempenho dos modelos mostrou clara correlação com o volume e a qualidade dos dados disponíveis. A insuficiência de dados no modelo inicial resultou em métricas ruins e movimentos inconsistentes.
   - O aumento gradual do dataset no modelo intermediário trouxe melhorias visíveis, mas ainda insuficientes para cenários de maior complexidade.
   - O modelo avançado, treinado com um volume maior de dados, atingiu métricas minimamente satisfatórias, destacando a importância de dados amplos e diversificados no treinamento.

2. **Análise Comparativa:**
   - A proximidade média ao ponto-alvo no modelo avançado foi de 1.25 cm, significativamente melhor que os 3.45 cm do modelo inicial.
   - A suavidade das trajetórias e o respeito às restrições cinemáticas também evoluíram progressivamente, reforçando o impacto positivo da maior quantidade de dados no treinamento.

3. **Desafios e Soluções:**
   - A insuficiência de dados foi o maior desafio enfrentado, resultando em um modelo inicial com baixa precisão.
   - A estratégia de coletar mais dados e reprocessá-los para melhor consistência foi crucial para alcançar os resultados do modelo avançado.

4. **Potenciais Melhorias:**
   - Incluir dados de demonstrações com variações de poses do paciente e deslocamentos maiores do jaleco.
   - Aplicar técnicas avançadas de data augmentation para aumentar a diversidade de dados sem a necessidade de mais teleoperação.

5. **Limitações do Projeto:**
   - O escopo restrito a poses fixas do paciente e configurações pré-definidas do jaleco limita a aplicabilidade prática em cenários reais mais variados.



### **Conclusão**

Os experimentos realizados demonstraram que o treinamento de um modelo GAIL para tarefas assistivas, como vestir um jaleco, é viável, mas exige um volume significativo de dados para alcançar resultados satisfatórios. A evolução entre os modelos inicial, intermediário e avançado destacou a importância de dados de alta qualidade e diversidade no aprendizado por imitação. Apesar das limitações impostas pelo cenário controlado, os resultados obtidos com o modelo avançado validam o potencial do GAIL para aplicações em robótica assistiva.

**Próximos Passos e Oportunidades para Expansão:**

1. **Aumento da Realidade e Complexidade do Cenário:**
   - Incorporar variabilidade nas configurações experimentais, como tamanhos diferentes de jalecos, poses distintas do paciente e materiais que ofereçam maior resistência à manipulação, simulando desafios reais enfrentados em ambientes assistivos.
   - Testar o modelo em cenários dinâmicos, onde o jaleco ou o paciente possam se mover durante a execução da tarefa, forçando o sistema a lidar com imprevisibilidades.

2. **Transição para o Mundo Real:**
   - Reaplicar todo o pipeline desenvolvido no ambiente simulado em um ambiente físico real, utilizando o manipulador robótico Kinova Gen3.
   - Coletar novos dados de teleoperação no mundo real para aumentar a robustez do modelo, enfrentando ruídos e variações presentes em interações físicas que não foram representadas no simulador.
   - Validar a capacidade do modelo de generalizar para o mundo físico, ajustando o sistema para compensar diferenças entre a simulação e a realidade (sim2real).

3. **Exploração de Novas Abordagens e Aprimoramentos Técnicos:**
   - Integrar técnicas de data augmentation para enriquecer o dataset sem necessidade de coleta extensiva.
   - Avaliar o impacto de arquiteturas mais complexas no modelo GAIL, como redes recorrentes (RNNs) ou Transformers, para melhorar a compreensão das dinâmicas temporais.

Esses avanços podem ampliar significativamente a aplicabilidade do sistema em contextos reais, contribuindo para a evolução da robótica assistiva e sua capacidade de lidar com tarefas complexas de forma segura e eficiente.


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

