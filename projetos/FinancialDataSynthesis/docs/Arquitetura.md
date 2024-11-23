# Detalhes da Arquitetura ##
==============================
<div align="center">
    <img src="../Arquitetura_Blocos.png" alt="Arquitetura em Blocos" title="Arquitetura em Blocos" />
    <p><em>Figura 1: Arquitetura da Rede Neural em Blocos.</em></p>
</div>


1.**Input:**

A entrada são as séries temporais do preço:

$$ X_{1:N} = [x(1), x(2), ..., x(N)] $$

E as séries temporais dos features, que por simplicidade, consideramos apenas uma série temporal, dada por:

$$ F_{1:N} = [f(1), f(2), ..., f(N)] $$

Essas séries são agrupadas em um mesmo dataframe.
