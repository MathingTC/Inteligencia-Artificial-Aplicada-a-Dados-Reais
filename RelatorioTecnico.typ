#import "@preview/abntly:0.1.1": *
#show: trabalho-academico.with(
  dados: config-dados(
    titulo: [Relatorio Tecnico - Inteligencia Artificial Aplicada],
    autor: "Matheus Camilo e Osmar Filho",
    orientador: "Malgarete Costa",
    instituicao: [Universidade Positivo],
    local: [Curitiba],
    ano: 2026,
  ),
)

#capa()
#folha-de-rosto[
  #lorem(50)
]

#outline()

= Introducao e contexto
O reconhecimento automático de caracteres manuscritos é um problema clássico de visão computacional e aprendizado de máquina, como por exemplo em aplicações para leitura de cheques bancários, triagem de correspondências por código postal e digitalização de documentos físicos. Por ter classes bem definidas e imagens pequenas, o reconhecimento de dígitos é muito usado para avaliar e comparar técnicas de aprendizado supervisionado.

Neste trabalho, uma Rede Neural de Multicamadas(MLP) foi aplicada à classificação de  dígitos manuscritos de 0 a 9 a partir de imagens em tom de cinza. O objetivo é analisar como a arquitetura da rede e a escolha da função de ativação influenciam o desempenho do modelo.

= Descricao do dataset
*Fonte:* foi utilizado o conjunto de dados "MNIST 60000 Hand Written Number Images", disponibilizado na plataforma Kaggle(https://www.kaggle.com/datasets/rakuraku678/mnist-60000-hand-written-number-images). Ele reproduz em formato CSV o MNIST.

*Período:* não se aplica, um período de coleta, pois se trata de uma basa estática de imagens, sem dimensão temporal.

*Quantidade de amostras:* VERIFICAR DEPOIS QUAL A QUANTIDADE EXATA como ficará o texto -> o arquivo _mnist_test.csv_ OU _mnist_train.csv_ contém X amostras e 785 colunas.

*Atributos:* A primeira coluna é o rótulo(label), um inteiro de 0 a 9. As 784 colunas seguintes são os pixels de uma imagem 28X28 _1X28X28?_, em ordem de linha a linha, com intensidade de 0(fundo preto) e 255(traço branco). //Verificar se coloco o seguinte texto: Cerca de 81% dos pixels têm valor 0, porque os dígitos ocupam uma pequena parte da imagem.

*Distribuição das classes:* //Verificar o que exatamente colocar arquivo


= Metodologia
== Preparação dos dados
*Seleção de atributos:* foram utilizados os 784 atributos de entrada, sem seleção prévia. Cada pixel pode carregar informação sobre o traço, e a própria rede aprende quais combinações são relevantes.

*Váriavel-alvo:* o rótulo do dígito(0 a 9), o que caracteriza uma classificação multiclasse com dez classes.

*Normalização:* os valores dos pixels foram divididos por 255, ficando entre 0 e 1. Esse procedimento evita que valores grandes dominem o treino e tende a acelerar a convergência.

*Divisão dos dados:* //Verificar exatamente o que colocar aqui, quantos % para treino e quantos % para testes.

== Implementação
O modelo foi implementado em Python com a biblioteca PyTorch. A função de perda foi a entropia cruzada(_CrossEntropyLoss_), adequada à classificação multiclasse. Como a entropia cruzada aplica o _softmax_ internamente, a última camada retorna valores brutos. O treinamento foi feito em GPU quando disponível.

+ Arquitetura: //verificar corretamente
+ Funções de ativação comparadas: //verificar tbm
+ Hiperparâmetros testados: //tbm olhar melhor

= Resultados obtido
metricas, graficos e comparacao entre funcoes de ativacao

= Discussao
analise critica dos resultados e limitacoes do modelo

= Conclusao
licoes aprendidas e potenciais melhorias




















