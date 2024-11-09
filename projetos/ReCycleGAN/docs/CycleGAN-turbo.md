# CycleGAN-turbo

A rede geradora da CycleGAN-turbo tem diversos elementos, e alguns são compartilhados entre as redes geradoras A→B e B→A. Além disso, muitos dos pesos são pré-treinados. A tabela abaixo lista o total de parâmetros associados a cada elemento. Para ter uma melhor comparação com os demais modelos, na coluna de parâmetros da rede geradora na tabela de modelos foi reportada a metade da soma dos pesos treináveis das quatro VAE e da U-net. Foi assumido que o encoder de texto é uma rede acessória, e assim não foi adicionado a este total.

| Elemento     |Total (MM)|Treináveis (MM) |
|-|:-:|:-:|
| VAE Encoder        |  170,077 |   1,786 |
| VAE Decoder        |  170,077 |   1,786 |
| U-net              |  947,025 |  81,114 |
| Text Encoder       |  340,388 | 340,388 |
| VAE A→B            |   85,039 |   0,893 |
| VAE B→A            |   85,039 |   0,893 |
| Discriminadora A→B |    3,672 |   3,672 |
| Discriminadora B→A |    3,672 |   3,672 |
| Total              | 1464.834 | 430.633 |
