# `Geração Sintética de Dados Aplicado a Reconhecimento de Atividades Humanas (HAR)`
# `Synthetic Data Generation for Human Activity Recognition (HAR)`

## Apresentação

O presente projeto foi originado no contexto das atividades da disciplina de pós-graduação *IA376N - IA generativa: de modelos a aplicações multimodais*,
oferecida no segundo semestre de 2024, na Unicamp, sob supervisão da Profa. Dra. Paula Dornhofer Paro Costa, do Departamento de Engenharia de Computação e Automação (DCA) da Faculdade de Engenharia Elétrica e de Computação (FEEC).

|Nome  | RA | Especialização|
|--|--|--|
| Bruno Guedes da Silva  | 203657  | Eng. de Computação|
| Amparo Díaz  | 152301  | Aluna especial|

## Resumo (Abstract)

> Resumo do objetivo, metodologia **e resultados** obtidos (na entrega E2 é possível relatar resultados parciais). Sugere-se máximo de 100 palavras.

## Descrição do Problema/Motivação

- **Falta de Dados:** A escassez de dados relevantes e diversos é um desafio significativo para o treinamento e avaliação de modelos de HAR. A coleta desse tipo de dados requer a participação de diversas pessoas em diferentes cenários e atividades. Embora a janela de tempo de cada captura de dados seja relativamente pequena (cerca de 1 a 15 minutos) o tempo de preparo do participante e deslocamento entre os locais em que as atividades são realizadas pode ser grande. Além disso, deve-se garantir que todos os sensores funcionem corretamente durante o experimento e que os dados sejam coretamente sincronizados e anonimizados. Diferentemente de dados como imagens, áudios e textos que são abundantemente presentes na internet, dados de sensores são mais escassos.
- **Heterogeneidade:** A variabilidade nas classes de atividade, na posição dos sensores e nas características das pessoas cria dificuldades para criar um dataset representativo e generalizável. A quantidade de atividades que uma pessoa pode realizar é imensa (subir escadas, pular, nadar, andar, correr) e pode ser modulada por diferentes fatores externos (clima, elevação, angulação do chão). Além disso, as características físicas do participante (altura, idade, peso, etc.) influenciam o comportamento dos dados. Esses fatores tornam difícil a construção de um dataset com classes bem definidas e variedade de participantes de forma a ser representativo o suficiente para generalização de modelos de aprendizado.

## Objetivo 

Diante do contexto e motivação apresentados, temos como objetivo a implementação e avaliação de um modelo que gere dados de sensores de acelerômetro e giroscópio (e possivelmente expandir para outras modalidades) correspondentes a diferentes atividades humanas.

> - Descrição do que o projeto se propõe a fazer.
> - É possível explicitar um objetivo geral e objetivos específicos do projeto.

## Metodologia

> - Descrever de maneira clara e objetiva, citando referências, a metodologia proposta para se alcançar os objetivos do projeto.
> - Descrever bases de dados utilizadas.

>### Bases de Dados a Serem Utilizadas
>
>Neste projeto, pretende-se utilizar datasets de ambientes controlados e não controlados para realizar uma comparação entre a performance do modelo generativo em cada cenário.

> - Citar algoritmos de referência.

>#### Redes Adversariais Generativas (GANs)
>
> Essa abordagem tem sido amplamente utilizada para gerar dados de sensores sintéticos, oferecendo uma boa capacidade de criar amostras realistas.
>- **Vantagens:** Capacidade de gerar dados complexos e variados, o que pode ser útil para criar um dataset diversificado de HAR.
>- **Desvantagens:** Treinamento pode ser instável e requer um grande número de exemplos para atingir uma boa performance.
>
>#### Modelos de Difusão
>
>- Recentemente, esses modelos têm mostrado resultados promissores em várias tarefas de geração de dados.
>- **Vantagens:** Capacidade de gerar amostras de alta qualidade e realismo. Eles podem ser particularmente úteis para criar dados sintéticos de HAR que mantêm as características dos dados reais.
>- **Desvantagens:** Modelos geralmente são mais complexos e podem exigir mais recursos computacionais para treinamento.

> - Justificar os porquês dos métodos escolhidos.
> - Apontar ferramentas relevantes.
> - Descrever metodologia de avaliação (como se avalia se os objetivos foram cumpridos ou não?).

### Bases de Dados e Evolução
> - Elencar bases de dados utilizadas no projeto.
> - Para cada base, coloque uma mini-tabela no modelo a seguir e depois detalhamento sobre como ela foi analisada/usada, conforme exemplo a seguir.

|Base de Dados | Endereço na Web | Resumo descritivo|
|----- | ----- | -----|
|Título da Base | http://base1.org/ | Breve resumo (duas ou três linhas) sobre a base.|

> Faça uma descrição sobre o que concluiu sobre esta base. Sugere-se que respondam perguntas ou forneçam informações indicadas a seguir:
> * Qual o formato dessa base, tamanho, tipo de anotação?
> * Quais as transformações e tratamentos feitos? Limpeza, reanotação, etc.
> * Inclua um sumário com estatísticas descritivas da(s) base(s) de estudo.
> * Utilize tabelas e/ou gráficos que descrevam os aspectos principais da base que são relevantes para o projeto.


### Workflow
> Use uma ferramenta que permita desenhar o workflow e salvá-lo como uma imagem (Draw.io, por exemplo). Insira a imagem nessa seção.
> Você pode optar por usar um gerenciador de workflow (Sacred, Pachyderm, etc) e nesse caso use o gerenciador para gerar uma figura para você.
> Lembre-se que o objetivo de desenhar o workflow é ajudar a quem quiser reproduzir seus experimentos. 



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

HUANG, S.; CHEN, P.-Y.; MCCANN, J. DiffAR: Adaptive Conditional Diffusion Model for Temporal-augmented Human Activity Recognition. Proceedings of the Thirty-Second International Joint Conference on Artificial Intelligence. Anais... Em: THIRTY-SECOND INTERNATIONAL JOINT CONFERENCE ON ARTIFICIAL INTELLIGENCE {IJCAI-23}. Macau, SAR China: International Joint Conferences on Artificial Intelligence Organization, ago. 2023. Disponível em: <https://www.ijcai.org/proceedings/2023/424>

MALEKZADEH, M. et al. Protecting Sensory Data against Sensitive Inferences. Proceedings of the 1st Workshop on Privacy by Design in Distributed Systems. Anais... Em: EUROSYS ’18: THIRTEENTH EUROSYS CONFERENCE 2018. Porto Portugal: ACM, 23 abr. 2018. Disponível em: <https://dl.acm.org/doi/10.1145/3195258.3195260>.

NORGAARD, S. et al. Synthetic Sensor Data Generation for Health Applications: A Supervised Deep Learning Approach. 2018 40th Annual International Conference of the IEEE Engineering in Medicine and Biology Society (EMBC). Anais... Em: 2018 40TH ANNUAL INTERNATIONAL CONFERENCE OF THE IEEE ENGINEERING IN MEDICINE AND BIOLOGY SOCIETY (EMBC). Honolulu, HI: IEEE, jul. 2018. Disponível em: <https://ieeexplore.ieee.org/document/8512470/>.

RAVURU, C.; SAKHINANA, S. S.; RUNKANA, V. Agentic Retrieval-Augmented Generation for Time Series Analysis. arXiv, , 18 ago. 2024. Disponível em: <http://arxiv.org/abs/2408.14484>.

VAIZMAN, Y.; ELLIS, K.; LANCKRIET, G. Recognizing Detailed Human Context in the Wild from Smartphones and Smartwatches. IEEE Pervasive Computing, v. 16, n. 4, p. 62–74, out. 2017. 

ydataai/ydata-profiling. YData, , 9 set. 2024. Disponível em: <https://github.com/ydataai/ydata-profiling>.
