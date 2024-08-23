# `<PulmoNet: Rede Neuronal Generativa para Imagens Tomográficas Pulmonares>`
# `<PulmoNet: Generative Neuronal Network for Pulmonary Tomographic Images>`

## Apresentação

O presente projeto foi originado no contexto das atividades da disciplina de pós-graduação *IA376N - IA generativa: de modelos a aplicações multimodais*, 
oferecida no segundo semestre de 2024, na Unicamp, sob supervisão da Profa. Dra. Paula Dornhofer Paro Costa, do Departamento de Engenharia de Computação e Automação (DCA) da Faculdade de Engenharia Elétrica e de Computação (FEEC).

> Incluir nome RA e foco de especialização de cada membro do grupo. Os grupos devem ter no máximo três integrantes.
> |Nome  | RA | Especialização|
> |--|--|--|
> | Arthur Matheus Do Nascimento | 290906 | Eng. Elétrica |
> | Júlia Castro de Paula | 219193 | Eng. Eletrica |
> | Letícia Levin Diniz | 201428  | Eng. Elétrica |


## Descrição Resumida do Projeto
> Descrição do tema do projeto, incluindo contexto gerador, motivação. 
> Descrição do objetivo principal do projeto.
> Esclarecer qual será a saída do modelo generativo.
Imagens tomográficas pulmonares são muito relevantes no contexto diagnóstico de enfermidades pulmonares e para mapeamento para a realização de processos operatórios na região. A seguimentação das vias-aéreas tem se mostrado um facilitador nesses processos por automatizar a identificação de tais estruturas. No entanto, as arquiteturas atuais dependem de um grande volume de dados para serem eficientes. Devido...
- porque redes generativas são úteis nesse problema
- objetivo: gerar imagens sintéticas de CT pulmonar comparáveis as reais
- saídas: imagem CT e a segmentação das vias-aereas na imagem
> Incluir nessa seção link para vídeo de apresentação da proposta do projeto (máximo 5 minutos).

## Metodologia Proposta
> Para a primeira entrega, a metodologia proposta deve esclarecer:
> * Qual(is) base(s) de dado(s) o projeto pretende utilizar, justificando a(s) escolha(s) realizadas.
- Base de dados principal: ATM'22 -> volumes CT pulmonares, volume com a segmentação das vias aéreas 
- 500 CT scans (300 for training, 50 for validation, and 150 for testing)
- multi-site
> * Quais abordagens de modelagem generativa o grupo já enxerga como interessantes de serem estudadas.
- GAN (devido as referências principais)
> * Artigos de referência já identificados e que serão estudados ou usados como parte do planejamento do projeto
- https://www.sciencedirect.com/science/article/abs/pii/S0957417422023685
> * Ferramentas a serem utilizadas (com base na visão atual do grupo sobre o projeto).
- Pytorch
- MLFlow
- Wand AI
- Google Colab
> * Resultados esperados
- Gerar amostras de imagens onde seja possível realizar a segmentação das vias aéreas
- Se tudo der certo... evoluir de imagens para volumes
> * Proposta de avaliação dos resultados de síntese
- Análise qualitativa: observação das imagens/volumes - GT/sintética
- Análise quantitativa: das imagens: DICE, SSIM; 
- Benchmark: segmentação das imagens reais e sintéticas
    - análise quantitativa da segmentação: DICE, precisão, qntdd ramificações

## Cronograma
> Proposta de cronograma. Procure estimar quantas semanas serão gastas para cada etapa do projeto.
- Cronograma:
- 10/09: leitura de artigo + familiarização com a DB/visualizar/ver se tem como identificar imagens com covid/sem
- 24/09: primeira versão da GAN (inspirada em https://www.sciencedirect.com/science/article/abs/pii/S0957417422023685)
- 07/10: estrutura de avaliação delimitada
- 08/10: E2
- 15/10: ter os primeiros resultados com imagens segmentadas e valores para validação 
- 29/10: fine-tuning, aperfeiçoamento do modelo
- 05/11: para frente: se os resultados estiverem minimamente decentes-> 3D
         senão: melhorar até ficar minimanete decente

## Referências Bibliográficas
> Apontar nesta seção as referências bibliográficas adotadas no projeto.
- Doc ref: https://docs.google.com/document/d/1uatPj6byVIEVrvMuvbII6J6-5usOjf8RLrSxLHJ8u58/edit