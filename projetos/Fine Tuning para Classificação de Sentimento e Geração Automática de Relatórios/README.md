# `Fine-Tuning para Classificação de Sentimentos e Geração Automática de Relatórios em Redes Sociais`
# `Fine-Tuning for Sentiment Classification and Automatic Report Generation on Social Media`

## Apresentação

O presente projeto foi originado no contexto das atividades da disciplina de pós-graduação **IA376N - IA generativa: de modelos a aplicações multimodais**, 
oferecida no segundo semestre de 2024, na Unicamp, sob supervisão da Profa. Dra. Paula Dornhofer Paro Costa, do Departamento de Engenharia de Computação e Automação (DCA) da Faculdade de Engenharia Elétrica e de Computação (FEEC).


> |Nome  | RA | Especialização|
> |--|--|--|
> | Maria Fernanda Paulino Gomes | 206745  | Eng. de Computação|
> | Raisson Leal Silva  | 186273  | Eng. Eletricista|



## Descrição Resumida do Projeto

### Tema:
O projeto propõe a Fine-Tuning para Classificação de Sentimentos e Geração Automática de Relatórios em Redes Sociais. Dado o volume de dados gerados diariamente nas redes sociais, compreender e sintetizar as emoções expressas tornou-se um desafio crucial para empresas e instituições. 

Este projeto visa criar uma solução automatizada para classificar sentimentos em posts e gerar relatórios personalizados para guiar decisões estratégicas.

### Contexto e Motivação:

Com o crescente uso das redes sociais, empresas e instituições educacionais buscam maneiras de analisar a opinião pública e o bem-estar emocional dos usuários. Monitorar esses sentimentos pode orientar ações estratégicas, campanhas de marketing, ou intervenções de saúde mental em escolas.

### Objetivo Principal:

Desenvolver um sistema que realize o fine-tuning de um modelo de NLP para classificar sentimentos em posts de redes sociais e utilizar um modelo generativo para sintetizar os resultados em relatórios textuais.

### Saída do Modelo Generativo:

A saída será um relatório textual automatizado, que sintetizará as análises sentimentais, destacando padrões emocionais e oferecendo recomendações práticas.

**Segue link da [apresentação da proposta](https://drive.google.com/file/d/1zRdguOu8w2gxaxyDLmr0tO2RyHJllme3/view?usp=sharing)**

## Metodologia Proposta

### Base de dados:

Inicialmente, tem-se com pretensão utilizar datasets de redes sociais contendo posts anotados com sentimentos, como o **Sentiment140** (tweets anotados) e o **SentiStrength**. Essas bases são adequadas devido à sua linguagem informal e textos curtos, característicos de redes sociais.

### Abordagens de Modelagem Generativa:

Utilizaremos modelos de fine-tuning como BERT ou RoBERTa para a classificação de sentimentos. E para a geração de relatórios, serão explorados modelos Transformers como o **GPT-3** ou **GPT-4**, devido à sua habilidade de gerar textos coerentes e resumir dados com base em padrões identificados.

### Artigos de Referência:

> * Prottasha, N.J.; Sami, A.A.; Kowsher, M.; Murad, S.A.; Bairagi, A.K.; Masud, M.; Baz, M. Transfer Learning for Sentiment Analysis Using BERT Based Supervised Fine-Tuning. Sensors 2022, 22, 4157. https://doi.org/10.3390/s22114157.
> * Cantini, R., Cosentino, C., Marozzo, F. (2024). Multi-dimensional Classification on Social Media Data for Detailed Reporting with Large Language Models. In: Maglogiannis, I., Iliadis, L., Macintyre, J., Avlonitis, M., Papaleonidas, A. (eds) Artificial Intelligence Applications and Innovations. AIAI 2024. IFIP Advances in Information and Communication Technology, vol 712. Springer, Cham. https://doi.org/10.1007/978-3-031-63215-0_8.
> * Devlin, J. et al. (2019): BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding. https://arxiv.org/abs/1810.04805?amp=1
> * Radford, A. et al. (2019): Language Models are Few-Shot Learners. https://proceedings.mlr.press/v139/radford21a
> * Pang, B., et al. (2002): Thumbs up? Sentiment Classification using Machine Learning Techniques. https://arxiv.org/abs/cs/0205070

### Ferramentas:

As principais ferramentas a serem utilizadas serão:
- **Python** com bibliotecas:
    - **Hugging Face Transformers**, **spaCy**, **TensorFlow** e **PyTorch**: Para o treinamento e fine-tuning dos modelos.
    - **Pandas** e **NumPy**: Para manipulação e pré-processamento dos dados.
    - **Matplotlib** e **Seaborn**: Serão utilizados para gerar gráficos que ajudem a interpretar e analisar os resultados obtidos pelo modelo, como curvas de precisão e recall, além de visualizações dos relatórios gerados.
- **Google Colab** ou **Jupyter Notebooks** para implementação e testes.
- **OpenAI API** e/ou outros modelos de geração de linguagem para os relatórios.



### Resultados Esperados:

Espera-se um sistema que classsifique sentimentos com alta precisão e gere relatórios automáticos que identifiquem padrões de emoções, facilitando decisões informadas.

### Avaliação dos Resultados de Síntese:

1. **Avaliação da Classificação de Sentimentos:**

    Serão utilizadas métricas padrão de avaliação de modelos de classificação, como **precisão** e **recall**. Essas métricas permitirão uma análise objetiva da performance do modelo de classificação de sentimentos, comparando os resultados previstos pelo modelo com as classificações reais disponíveis na base de dados.

2. **Avaliação dos Relatórios Gerados:**
    - **Análise Comparativa**: Os relatórios gerados automaticamente pelo modelo serão comparados qualitativamente com relatórios manuais criados pelos participantes do grupo. Essa comparação se concentrará em 4 pilares: **coerência**, **clareza**, e **relevância das informações apresentadas**, além da **adequação das recomendações fornecidas**. Um grupo de voluntarios (alunos da nossa turma IA376N) vão receber um questionário estruturado que incluirá questões sobre os 4 pilares pontuando cada versão de 0 a 10.


## Cronograma

### **Fase 1: Revisão de Literatura e Planejamento**
**08/09/2024 a 13/09/2024** (5 dias)
- **Atividades**: 
  - Revisão de artigos e materiais de referência.
  - Definição detalhada do pipeline do projeto.
  - Escolha e preparação das bases de dados (Sentiment140 e SentiStrength).
  - Configuração do ambiente de desenvolvimento (Google Colab, Jupyter Notebooks).
- **Entrega Esperada**:
    - Fluxograma do processo com detalhamento das entradas e saídas de cada etapa.

### **Fase 2: Coleta e Pré-processamento dos Dados**
**14/09/2024 a 18/09/2024** (5 dias)
- **Atividades**:
  - Carregamento dos datasets.
  - Limpeza e normalização dos dados.
  - Análise exploratória dos dados.
  - Divisão dos dados em conjuntos de treinamento e teste.
- **Entrega Esperada**:
    - Conjunto de dados de treinamento e teste tratados


### **Fase 3: Fine-Tuning do Modelo de Classificação de Sentimentos**
**19/09/2024 a 06/10/2024** (2 semanas e meia)
- **Atividades**:
  - Implementação do fine-tuning usando BERT ou RoBERTa.
  - Treinamento do modelo com validação cruzada.
  - Avaliação preliminar do modelo usando métricas como precisão, recall.
  - Ajustes e otimização do modelo com base nos resultados obtidos.
- **Entrega Esperada**:
    - Modelo de classificação de sentimentos e suas métricas de avaliação

### **Entrega E2: Apresentação do status atual do projeto**
**06/10/2024 a 08/10/2024**
- **Atividade**:
    - Criação da apresentação 2, aonde vamos trazer os avanços do projeto.
    - Atualizar a todos se ouve alguma mudança no projeto inicial, se sim trazer novo cronograma
    - Afirmar sobre oque esperamos entregar no final do semestre
- **Entrega Esperada**:
    - Apresentação com status do projeto
    

### **Fase 4: Implementação do Modelo Generativo e Geração de Relatórios**
**09/10/2024 a 22/10/2024** (2 semanas)
- **Atividades**:
  - Configuração da API do OpenAI para uso do GPT-3 ou GPT-4.
  - Desenvolvimento de scripts para geração automática de relatórios baseados nos resultados da classificação de sentimentos.
  - Teste da qualidade e coerência dos relatórios gerados.
- **Entrega Esperada**:
    - Modelo de geração de relatorios especificos para avaliação de sentimentos das redes sociais


### **Fase 5: Avaliação dos Resultados e Análise Comparativa**
**23/10/2024 a 01/11/2024** (1 semana e meia)
- **Atividades**:
  - Realização da análise comparativa entre relatórios automáticos e manuais.
  - Reunião com voluntários para aplicação dos questionários e coleta de feedback.
  - Consolidação dos resultados da avaliação.
- **Entrega Esperada**:
    - Coleta das métricas de avaliação definidas


### **Fase 6: Ajustes Finais e Documentação**
**02/11/2024 a 15/11/2024** (2 semanas)
- **Atividades**:
  - Ajustes finais no modelo de classificação e na geração de relatórios com base na avaliação.
  - Elaboração da documentação final do projeto, incluindo detalhes técnicos e resultados.
- **Entrega Esperada**:
    - Documentação tecnica do processo de geração dos relatórios e modelos classificatório e gerador


### **Fase 7: Preparação para a Entrega Final**
**16/11/2024 a 25/11/2024** (1 semana e meia)
- **Atividades**:
  - Criação de slides e preparação para a apresentação oral.
  - Treinamento para a apresentação da entrega final, ensaios e ajustes.
- **Entrega Esperada**:
    - Apresentação final

## Referências Bibliográficas
> * Prottasha, N.J.; Sami, A.A.; Kowsher, M.; Murad, S.A.; Bairagi, A.K.; Masud, M.; Baz, M. Transfer Learning for Sentiment Analysis Using BERT Based Supervised Fine-Tuning. Sensors 2022, 22, 4157. https://doi.org/10.3390/s22114157.
> * Cantini, R., Cosentino, C., Marozzo, F. (2024). Multi-dimensional Classification on Social Media Data for Detailed Reporting with Large Language Models. In: Maglogiannis, I., Iliadis, L., Macintyre, J., Avlonitis, M., Papaleonidas, A. (eds) Artificial Intelligence Applications and Innovations. AIAI 2024. IFIP Advances in Information and Communication Technology, vol 712. Springer, Cham. https://doi.org/10.1007/978-3-031-63215-0_8.
> * Devlin, J. et al. (2019): BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding. https://arxiv.org/abs/1810.04805?amp=1
> * Radford, A. et al. (2019): Language Models are Few-Shot Learners. https://proceedings.mlr.press/v139/radford21a
> * Pang, B., et al. (2002): Thumbs up? Sentiment Classification using Machine Learning Techniques. https://arxiv.org/abs/cs/0205070
