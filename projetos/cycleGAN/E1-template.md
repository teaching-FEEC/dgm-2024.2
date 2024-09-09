# `Implementação de CycleGAN`
# `CycleGAN Implementation`

## Apresentação

O presente projeto foi originado no contexto das atividades da disciplina de pós-graduação *IA376N - IA generativa: de modelos a aplicações multimodais*, 
oferecida no segundo semestre de 2024, na Unicamp, sob supervisão da Profa. Dra. Paula Dornhofer Paro Costa, do Departamento de Engenharia de Computação e Automação (DCA) da Faculdade de Engenharia Elétrica e de Computação (FEEC).

> |Nome  | RA | Especialização|
> |--|--|--|
> | Vinicius Ventura Andreossi  | 195125  | Eng. de Computação|
> | Cosme Rodolfo Roque dos Santos  | 042687  | Doutorado Eng. Elétrica|


## Descrição Resumida do Projeto
A IA gerativa possui diversas aplicações além da sua mais comum, síntese direta de dados. Uma dessas aplicações é a translação imagem-para-imagem (I2I), que procura transferir uma imagem de um domínio para outro enquanto preserva o conteúdo da imagem original. 

A motivação pelo estudo de translação I2I é estudar aplicações menos convencionais de modelos gerativos, visto que a síntese de dados já é estudada com frequência. Além disso, imagens são sinais que não exigem conhecimentos aprofundados na área para analisar qualitativamente os resultados obtidos. 

Dentro do contexto de translação I2I, a CycleGAN se destaca por dois principais motivos: 

- É um modelo conhecido e consolidado com diversas implementações, conteúdo e artigos disponíveis na internet.
- Utiliza em seu treinamento duas distribuições não pareadas (*unsupervised I2I translation*), isto é, não são necessários pares de uma mesma imagem nos dois domínios de interesse.

Com a finalidade de fazer um estudo inicial do modelo, o problema escolhido para esse momento será a translação de cavalo para zebra. Futuramente, o modelo obtido anteriormente será adaptado para a solução de um problema mais relevante, sendo um potencial candidato a extração de mapas a partir de imagens de satélite.

## Metodologia Proposta
Inicialmente, pretende-se dividir o projeto em duas etapas:
- Etapa 1: Estudos iniciais da CycleGAN com um modelo conversor de imagens de cavalo para imagens de zebra. A implementação e o conjunto de dados utilizados serão reproduzidos de um blog na internet. O objetivo dessa etapa será estudar os aspectos práticos da implementação, buscando enxergar a aplicação dos conceitos teóricos e isolar possíveis pontos de dificuldade na criação do modelo, isolando os problemas relacionados ao processamento dos dados. Base de dados a ser utilizada: [Horse2zebra](https://www.kaggle.com/datasets/balraj98/horse2zebra-dataset?resource=download).
- Etapa 2: Adaptação da implementação inicial para a extração de mapas a partir de imagens de satélite. Esta etapa ainda não está definida pois apenas encontramos um artigo abordando o problema. No entanto, a ideia seria utilizar conjuntos de dados com imagens do próprio Google Maps e testar a solução final com imagens arbitrárias, extraídas "ao vivo". Possível base de dados: [Pix2pix-UCBerkerley](http://efrosgans.eecs.berkeley.edu/pix2pix/datasets/).

Cada uma das etapas seguirá a seguinte metodologia:
1. Análise inicial dos dados busca por datasets adequados, visualização de amostras, definição do pipeline de pré-processamento atentando-se a limitações computacionais (por exemplo, pode ser necessário reduzir a resolução das imagens).
2. Implementação de um código funcional, sem necessariamente conseguir fazer o modelo convergir. As implementações serão todas realizadas em PyTorch e o monitoramento do treinamento será provavelmente realizado via WandB.
3. Ajuste de hiperparâmetros buscando uma convergência satisfatória do modelo a ser analisada qualitativamente. Nesse ponto, o objetivo é obter resultados que demonstrem que o mapeamento desejado foi obtido, ainda que com imperfeições e/ou distorções.
4. Análise quantitativa dos resultados reproduzindo métricas utilizadas nos trabalhos de referência.

No momento atual, existe uma grande confiança na obtenção do modelo de conversão de cavalos para zebras por se tratar da reprodução de uma solução conhecida. O modelo de extração de mapas de imagens de satélite é ambicioso e mais arriscado. Seu sucesso dependerá muito da disponibilidade de um conjunto de dados de boa qualidade, poder computacional suficiente e, até mesmo, sorte de encontrar um bom conjunto de hiperparâmetros a tempo. 

O resultado esperado ideal seria um modelo capaz de extrair um mapa a partir de uma foto de satélite arbitrária. Um marco interessante seria conseguir extrair um mapa da Unicamp a partir de sua imagem de satélite e, como meta adicional, obter um mapa verossímil a partir de uma vista aérea de drone (e não de satélite). Se o modelo for robusto, é possível que consiga extrair um mapa com deformações mas não tão distante do mapa real.

> * Qual(is) base(s) de dado(s) o projeto pretende utilizar, justificando a(s) escolha(s) realizadas.
> * Quais abordagens de modelagem generativa o grupo já enxerga como interessantes de serem estudadas.
> * Artigos de referência já identificados e que serão estudados ou usados como parte do planejamento do projeto
> * Ferramentas a serem utilizadas (com base na visão atual do grupo sobre o projeto).
> * Resultados esperados
> * Proposta de avaliação dos resultados de síntese

## Cronograma
| Data    | Planejamento |
| --------| ------- |
| 10/09   | Definição do tema e definição da bibliografia. |
| 24/09   | Implementação inicial da CycleGAN + treinamento com conjunto arbitrário de hiperparâmetros. |
| 08/10   | Etapa 1 concluída com CycleGAN otimizada e produzindo resultados satisfatórios. Início da busca por datasets para a Etapa 2.|
| 22/10   | Datasets escolhidos e visualizados. Leitura de trabalhos na área e definição de modelo. Início da implementação. |
| 05/11   | Data prevista para finalização da Etapa 2. O resto do prazo será utilizado para resolver problemas e imprevistos, ou para otimizar os resultados obtidos. |
| 26/11   | Etapa 2 concluída e funcionando adequadamente. README.md escrito, aplicativo Streamlit / Gradio mostrando o projeto em funcionamento (*). |

## Referências Bibliográficas
<a id="1">[1]</a>
Jun-Yan Zhu*, Taesung Park*, Phillip Isola, and Alexei A. Efros, (2020).
"Unpaired Image-to-Image Translation using Cycle-Consistent Adversarial Networks", in
*IEEE International Conference on Computer Vision (ICCV)*, 2017.

<a id="2">[2]</a>
Hoyez, H.; Schockaert, C.; Rambach, J.; Mirbach, B.; Stricker, D. Unsupervised Image-to-Image Translation: a Review. *Sensors*. **2022**, 22, 8540

<a id="2">[3]</a>
Song, J.; Li, J.; Chen, H.; Wu J. RSMT: A Remote Sensing Image-to-Map Translation Model via Adversarial Deep Transfer Learning. *Remote Sensing*. **2022**, 14, 919.