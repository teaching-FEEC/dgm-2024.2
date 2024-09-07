# `<Fine-Tuning para Classificação de Sentimentos e Geração Automática de Relatórios em Redes Sociais>`
# `<Fine-Tuning for Sentiment Classification and Automatic Report Generation on Social Media>`

## Apresentação

O presente projeto foi originado no contexto das atividades da disciplina de pós-graduação *IA376N - IA generativa: de modelos a aplicações multimodais*, 
oferecida no segundo semestre de 2024, na Unicamp, sob supervisão da Profa. Dra. Paula Dornhofer Paro Costa, do Departamento de Engenharia de Computação e Automação (DCA) da Faculdade de Engenharia Elétrica e de Computação (FEEC).


> |Nome  | RA | Especialização|
> |--|--|--|
> | Maria Fernanda Paulino Gomes | 206745  | Eng. de Computação|
> | Raisson Leal Silva  | 123456  | Eng. Eletricista|



## Descrição Resumida do Projeto

**Tema:**

O projeto propõe a Fine-Tuning para Classificação de Sentimentos e Geração Automática de Relatórios em Redes Sociais. 
Dado o volume de dados gerados diariamente nas redes sociais, compreender e sintetizar as emoções expressas tornou-se um desafio crucial para empresas e instituições. 
Este projeto visa criar uma solução automatizada para classificar sentimentos em posts e gerar relatórios personalizados para guiar decisões estratégicas.

**Contexto e Motivação:**

Com o crescente uso das redes sociais, empresas e instituições educacionais buscam maneiras de analisar a opinião pública e o bem-estar emocional dos usuários. Monitorar esses sentimentos pode orientar ações estratégicas, campanhas de marketing, ou intervenções de saúde mental em escolas.

**Objetivo Principal:**

Desenvolver um sistema que realize o fine-tuning de um modelo de NLP para classificar sentimentos em posts de redes sociais e utilizar um modelo generativo para sintetizar os resultados em relatórios textuais.

**Saída do Modelo Generativo:**

A saída será um relatório textual automatizado, que sintetizará as análises sentimentais, destacando padrões emocionais e oferecendo recomendações práticas.

> Incluir nessa seção link para vídeo de apresentação da proposta do projeto (máximo 5 minutos).

## Metodologia Proposta

**Base de dados:**

Inicialmente, tem-se com pretensão utilizar datasets de redes sociais contendo posts anotados com sentimentos, como o **Sentiment140** (tweets anotados) e o **SentiStrength**. Essas bases são adequadas devido à sua linguagem informal e textos curtos, característicos de redes sociais.

**Abordagens de Modelagem Generativa:**

Para a geração de relatórios, serão explorados modelos como o **GPT-3** ou **GPT-4**, devido à sua habilidade de gerar textos coerentes e resumir dados com base em padrões identificados.

**Artigos de Referência:**

> Prottasha, N.J.; Sami, A.A.; Kowsher, M.; Murad, S.A.; Bairagi, A.K.; Masud, M.; Baz, M. Transfer Learning for Sentiment Analysis Using BERT Based Supervised Fine-Tuning. Sensors 2022, 22, 4157. https://doi.org/10.3390/s22114157.
> > Cantini, R., Cosentino, C., Marozzo, F. (2024). Multi-dimensional Classification on Social Media Data for Detailed Reporting with Large Language Models. In: Maglogiannis, I., Iliadis, L., Macintyre, J., Avlonitis, M., Papaleonidas, A. (eds) Artificial Intelligence Applications and Innovations. AIAI 2024. IFIP Advances in Information and Communication Technology, vol 712. Springer, Cham. https://doi.org/10.1007/978-3-031-63215-0_8.
> Devlin, J. et al. (2019): BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding.
> Radford, A. et al. (2019): Language Models are Few-Shot Learners.
> Pang, B., et al. (2002): Thumbs up? Sentiment Classification using Machine Learning Techniques.

**Ferramentas:**

As principais ferramentas a serem utilizadas serão:
> **Python** com bibliotecas de NLP como **Hugging Face Transformers** e **spaCy** para o fine-tuning de modelos.
> **OpenAI API** e/ou outros modelos de geração de linguagem para os relatórios.
> **Google Colab** ou **Jupyter Notebooks** para implementação e testes.


**Resultados Esperados:**

Espera-se um sistema que classsifique sentimentos com alta precisão e gere relatórios automáticos que identifiquem padrões de emoções, facilitando decisões informadas.

**Avaliação dos Resultados de Síntese:**

A avaliação será baseada na **precisão** e no **recall** da classificação de sentimentos, além de testes qualitativos e quantitativos para verificar a coerência e relevância dos relatórios.


## Cronograma
> Proposta de cronograma. Procure estimar quantas semanas serão gastas para cada etapa do projeto.

## Referências Bibliográficas
> Apontar nesta seção as referências bibliográficas adotadas no projeto.
