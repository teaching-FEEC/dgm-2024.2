
# Nexet 2017

A base de dados **Nexet 2017** contém 50.000 imagens, e 99,8% tem resolução 1280x720. Todas as imagens tem dados de condição de luz (dia, noite, ocaso) e local (Nova York, São Francisco, Tel Aviv, Resto do mundo). Também existem dados anotados da posição (*box*) dos veículos que aparecem em cada imagem. Para o treinamento e teste das redes propostas foram utilizadas apenas as imagens 1280x720 de Nova York, nas condições de luz **dia** (4885 imagens) e **noite** (4406 imagens).

<div>
    <p align="center">
        <img src='assets/nexet_imgs.png' align="center" alt="Imagens Nexet" width=600px>
    </p>
</div>

<p align="center">
    <strong>Exemplos de imagens da base Nexet 2017 (dia acima e noite abaixo).</strong>
</p>

## Imagens com Problemas

Algumas das imagens da base de dados parecem ter tido problemas na sua captura. Em diversas imagens o conteúdo da mesma se encontrava em um dos cantos da imagem. Para tratar estas imagens é feita uma busca pela linhas e colunas da imagem buscando *informação*. Uma linha ou coluna é considerada *sem informação* quando a imagem equivalente em escala de cinza não tinha nenhum pixel com valor maior que 10 (em uma escala até 255). A imagem original é então cortada na região *com informação* antes de escalar e cortar as imagens para 256x256. Imagens recortadas com menos de 256 pixeis de altura ou largura foram ignoradas (imagem à esquerda abaixo).

<div>
    <p align="center">
        <img src='assets/nexet/bad_image01.jpg' align="center" alt="Imagem ruim" width=250px>
        <img src='assets/nexet/bad_image02.jpg' align="center" alt="Imagem ruim" width=250px>
    </p>
</div>

<p align="center">
    <strong>Exemplos de imagens com problemas.</strong>
</p>

## Filtro de Imagens

Observou-se que algumas imagens da base de dados Nexet apresentavam características que poderiam comprometer a qualidade do treinamento. Foi feito um trabalho *semi*-manual de filtragem destas imagens. Muitas das análises foram feitas com base nas *distâncias* entre as imagens de cada grupo. Estas distâncias foram calculadas a partir da saída da penúltima camada de uma rede classificadora de imagens pré-treinada ResNet18 [[13]](https://doi.org/10.1109/CVPR.2016.90), disponibilizada diretamente no [PyTorch](https://pytorch.org/vision/main/models/generated/torchvision.models.resnet18.html). Esta extração de características foi realizada com as imagens já escaladas e recortadas para o formato de treinamento.

### Imagens muito parecidas

* Foram listados os pares de imagens que apresentavam menores distâncias entre si.
* Foi definido por inspeção visual, para a classe **dia**, que os 93 pares mais próximos eram de imagens muito semelhantes. Para cada par uma das imagens é excluída da base de dados.
* Para a classe **noite** esta abordagem não se mostrou muito eficiente. Imagens com pequena distância entre si não eram consideradas parecidas em uma inspeção visual. Para esta classe nenhuma imagem foi retirada.

<div>
    <p align="center">
        <img src='assets/nexet/close_pair_day_01.png' align="center" alt="Imagens próximas dia 1" width=350px>
        <img src='assets/nexet/close_pair_day_02.png' align="center" alt="Imagens próximas dia 2" width=350px>
    </p>
</div>

<p align="center">
  <strong>Exemplos de pares de imagens muito parecidas na classe dia.</strong>
</p>

<div>
    <p align="center">
        <img src='assets/nexet/close_pair_night_01.png' align="center" alt="Imagens próximas noite 1" width=350px>
        <img src='assets/nexet/close_pair_night_02.png' align="center" alt="Imagens próximas noite 2" width=350px>
    </p>
</div>

<p align="center">
    <strong>Exemplos de pares de imagens muito parecidas na classe noite.</strong>
</p>

### Imagens *Difíceis*

* Para *facilitar* o treinamento da rede, foram excluídas imagens com características consideradas *difíceis* ou que não ajudam no treinamento: chuva *forte*, túneis, desfoque, objetos bloqueando a visão.
* Para esta análise as imagens de cada classe foram agrupadas em 20 classes, com **k-Means**. Para cada classe foram sorteadas 36 imagens e foi feita uma análise visual de cada grupo.
* A partir da análise visual, os grupos que foram considerados *problemáticos* são novamente divididos com k-means. A análise visual dos subgrupos é que define que conjuntos de imagens são excluídos do treinamento.

<div>
    <p align="center">
        <img src="assets/nexet/bad_cluster_day.jpg" align="center" alt="Imagens difíceis dia" width=350px>
        <img src="assets/nexet/bad_cluster_night.jpg" align="center" alt="Imagens difíceis noite" width=350px>
    </p>
</div>

<p align="center">
    <strong>Exemplos de grupos de imagens consideradas difíceis para o treinamento.</strong>
</p>

Os filtros aplicados retiraram 146 (3%) das imagens da classe **Dia** e 216 (5%) das imagens da classe **Noite**. Os totais de imagens para cada classe são apresentados abaixo.

| Classe       | Treino | Teste | Total |
|--------------|--------|-------|-------|
|**Dia** (A)   | 3788   | 949   | 4737  |
|**Noite** (B) | 3316   | 842   | 4158  |

Todo o procedimento de filtro das imagens está codificado em um único [Notebook](../src/notebooks/Filter_DayNight.ipynb).

A base de dados utilizada pode ser encontrada neste [link](https://github.com/TiagoCAAmorim/dgm-2024.2/releases/download/v0.1.1-nexet/Nexet.zip). Foram utilizadas as imagens listadas nos arquivos com *\_filtered.csv* no final do nome.
