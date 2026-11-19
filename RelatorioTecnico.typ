#import "@preview/abntly:0.1.1": *
#show: trabalho-academico.with(
  dados: config-dados(
    titulo: [Relatório Técnico - Inteligência Artificial Aplicada],
    autor: "Matheus Teclak Camilo e Osmar Cardoso Lopes Filho",
    orientador: "Malgarete Costa",
    instituicao: [Universidade Positivo],
    local: [Curitiba],
    ano: 2026,
  ),
)

#capa()
#folha-de-rosto[
Relatório técnico apresentado como atividade avaliativa da disciplina de Sistemas Inteligentes, da Universidade Positivo.
]

#outline()

= Introdução e contexto
O reconhecimento automático de caracteres manuscritos é um problema clássico de visão computacional e aprendizado de máquina, com aplicações como a leitura de cheques bancários, a triagem de correspondências por código postal e a digitalização de documentos físicos. Por ter classes bem definidas e imagens pequenas, o reconhecimento de dígitos é muito usado para avaliar e comparar técnicas de aprendizado supervisionado.

Neste trabalho, uma Rede Neural Convolucional (CNN) foi aplicada à classificação de dígitos manuscritos de 0 a 9 a partir de imagens em tons de cinza. Diferentemente de uma rede densa, a CNN utiliza camadas convolucionais que extraem padrões locais da imagem, como bordas e traços, preservando a estrutura espacial dos pixels. O objetivo é avaliar o desempenho dessa arquitetura na tarefa de reconhecimento de dígitos e analisar suas limitações.

= Descrição do dataset
*Fonte:* foi utilizado o conjunto de dados MNIST (_Modified National Institute of Standards and Technology_), base de dígitos manuscritos amplamente difundida na literatura (LECUN et al., 1998). Os dados foram obtidos diretamente pela função _datasets.MNIST_ da biblioteca _torchvision_, que baixa os arquivos originais da base, sem o uso de arquivos CSV. // TODO: acrescentar a data de acesso e a referência LECUN et al. (1998) na lista de referências.

*Período:* não se aplica um período de coleta, pois se trata de uma base estática de imagens, sem dimensão temporal.

*Quantidade de amostras:* o conjunto possui 70.000 imagens, divididas em 60.000 amostras de treino e 10.000 amostras de teste.

*Atributos:* cada amostra é uma imagem de 28 × 28 pixels em tons de cinza (um único canal), com intensidade de 0 (fundo preto) a 255 (traço branco), acompanhada de um rótulo (_label_), um inteiro de 0 a 9 que indica o dígito representado. Cada imagem possui, portanto, 784 valores de entrada.

*Distribuição das classes:* as classes são razoavelmente balanceadas. No conjunto de treino, a menor classe é o dígito 5 (5.421 amostras) e a maior é o dígito 1 (6.742 amostras); no conjunto de teste, a menor é novamente o dígito 5 (892 amostras) e a maior é o dígito 1 (1.135 amostras), conforme a @tab-classes. Por isso a acurácia é uma métrica adequada para a avaliação.

#figure(
  table(
    columns: 11,
    align: center,
    [*Dígito*], [0], [1], [2], [3], [4], [5], [6], [7], [8], [9],
    [*Treino*], [5.923], [6.742], [5.958], [6.131], [5.842], [5.421], [5.918], [6.265], [5.851], [5.949],
    [*Teste*], [980], [1.135], [1.032], [1.010], [982], [892], [958], [1.028], [974], [1.009],
  ),
  caption: [Distribuição das classes nos conjuntos de treino e de teste do MNIST],
) <tab-classes>

= Metodologia
== Preparação dos dados
*Seleção de atributos:* a imagem inteira, como um tensor de dimensão 1 × 28 × 28, é fornecida à rede, sem seleção prévia de pixels. Como a CNN aprende sozinha quais padrões locais são relevantes, não foi necessário descartar atributos.

*Variável-alvo:* o rótulo do dígito (0 a 9), o que caracteriza uma classificação multiclasse com dez classes.

*Normalização:* os valores dos pixels foram convertidos para a escala de 0 a 1 e, em seguida, padronizados com média 0,1307 e desvio padrão 0,3081, valores calculados sobre o conjunto de treino do MNIST. Esse procedimento evita que valores grandes dominem o treino e tende a acelerar e estabilizar a convergência.

*Divisão dos dados:* foi usada a divisão original do MNIST: 60.000 imagens para treino (85,7%) e 10.000 para teste (14,3%). Não foi separado um conjunto de validação. Durante o treino, os dados são embaralhados a cada época e processados em lotes de 64 imagens. O conjunto de teste é avaliado ao final de cada época, em lotes de 1.000 imagens.

== Implementação
O modelo foi implementado em Python com a biblioteca PyTorch. A função de perda foi a entropia cruzada (_CrossEntropyLoss_), adequada à classificação multiclasse. Como a entropia cruzada aplica o _softmax_ internamente, a última camada retorna valores brutos (_logits_). O treinamento foi feito em GPU quando disponível.

+ *Arquitetura:* a rede recebe uma imagem de 1 × 28 × 28 e possui duas etapas convolucionais seguidas de duas camadas totalmente conectadas. A primeira convolução aplica 32 filtros 3 × 3 com preenchimento de 1 pixel e é seguida de _max pooling_ 2 × 2, gerando 32 mapas de 14 × 14. A segunda aplica 64 filtros 3 × 3, também seguida de _max pooling_ 2 × 2, gerando 64 mapas de 7 × 7. Esses mapas são achatados em um vetor de 3.136 elementos, que passa por uma camada linear de 128 neurônios, por _dropout_ de 25% e por uma camada linear final de 10 neurônios, uma por dígito. O modelo possui 421.642 parâmetros treináveis.

+ *Funções de ativação:* a função ReLU foi aplicada após as duas convoluções e após a primeira camada linear. A camada de saída não usa ativação, pois fornece os _logits_ à função de perda. A ReLU foi escolhida por ser simples, computacionalmente barata e por mitigar o problema de desaparecimento do gradiente. Não houve comparação com outras funções de ativação.

+ *Hiperparâmetros:* otimizador Adam com taxa de aprendizado de 0,001, lotes de 64 imagens, _dropout_ de 25% e no máximo 5 épocas. O treino é interrompido antes se a acurácia no conjunto de teste ultrapassar 95%. Esses são valores usuais para o MNIST e não foram obtidos por busca sistemática. // TODO: informar em qual época o treino parou (o código não registra isso; o critério de 95% costuma ser atingido já na 1ª época).

= Resultados obtidos
O modelo final foi avaliado nas 10.000 imagens do conjunto de teste. Ele acertou 9.889 imagens e errou 111, o que corresponde a uma acurácia de *98,89%* e a uma perda (entropia cruzada) de 0,0405.

A @tab-metricas mostra o desempenho por dígito. A revocação (_recall_) indica, entre as imagens de cada dígito, quantas foram reconhecidas corretamente, e a precisão indica, entre as previsões de cada dígito, quantas estavam corretas. Os dígitos 3 e 6 tiveram as maiores revocações (99,70% e 99,69%), e os dígitos 9 e 8 tiveram as menores (97,32% e 97,84%).

#figure(
  table(
    columns: 4,
    align: center,
    [*Dígito*], [*Precisão (%)*], [*Revocação (%)*], [*F1 (%)*],
    [0], [99,28], [98,98], [99,13],
    [1], [99,47], [99,38], [99,43],
    [2], [98,74], [98,93], [98,84],
    [3], [98,24], [99,70], [98,97],
    [4], [98,98], [99,19], [99,08],
    [5], [99,10], [98,99], [99,05],
    [6], [98,56], [99,69], [99,12],
    [7], [98,64], [98,83], [98,74],
    [8], [99,06], [97,84], [98,45],
    [9], [98,79], [97,32], [98,05],
  ),
  caption: [Métricas por dígito no conjunto de teste],
) <tab-metricas>

A @fig-matriz apresenta a matriz de confusão, em que as linhas são o dígito real e as colunas o dígito previsto. A diagonal concentra quase todas as amostras, e os erros são poucos e dispersos. As confusões mais frequentes foram o dígito 7 previsto como 2 (7 casos), o 0 previsto como 6 (7 casos), o 9 previsto como 7 (6 casos), o 9 como 4 (6 casos) e o 8 como 9 (6 casos).

#figure(
  image("matriz_confusao_contagem.png", width: 85%),
  caption: [Matriz de confusão no conjunto de teste (contagem de imagens)],
) <fig-matriz>

// TODO (opcional): o descritivo pede o vetor de erros por iteração. O código imprime a perda média a cada 200 lotes;
// guardar esses valores em uma lista e gerar um gráfico de perda por iteração.
// TODO: não há comparação entre funções de ativação, pois só a ReLU foi usada. Ver Discussão.

= Discussão
O desempenho de 98,89% mostra que a CNN extrai bem os padrões que distinguem os dígitos, mesmo com poucas épocas de treino. Os erros se concentram em formas que podem ser visualmente parecidas, como 7 e 2, 0 e 6, ou 9 com 4, 7 e 8, o que sugere que a rede erra principalmente em escritas ambíguas ou pouco convencionais. Esta é uma interpretação a partir da matriz de confusão, pois as imagens erradas não foram inspecionadas individualmente.

O trabalho tem limitações que devem ser consideradas ao interpretar os resultados:

- *Ausência de validação:* o conjunto de teste foi usado a cada época e para decidir a parada do treino. Por isso, a acurácia de teste pode ser levemente otimista, e o ideal seria separar um conjunto de validação a partir dos dados de treino.
- *Execução única:* o modelo foi treinado uma vez, sem repetir com sementes diferentes, então não há estimativa da variação do resultado.
- *Sem comparação de alternativas:* foi usada apenas a ReLU, e os hiperparâmetros não foram variados de forma sistemática, de modo que não é possível afirmar que esta configuração é a melhor.
- *Dataset simples:* o MNIST tem dígitos centralizados, com fundo uniforme e de tamanho padronizado. O desempenho pode ser menor em escritas reais, com ruído, inclinação ou fundos variados.

= Conclusão
A CNN desenvolvida alcançou 98,89% de acurácia na classificação de dígitos manuscritos do MNIST, com erros concentrados em poucos pares de dígitos parecidos. O trabalho mostrou o fluxo completo de um problema de classificação com redes neurais: preparação dos dados, definição da arquitetura, treinamento e avaliação com métricas por classe e matriz de confusão.

Como melhorias futuras, destacam-se: separar um conjunto de validação para decidir a parada do treino; comparar a ReLU com outras funções de ativação, como Tanh e LeakyReLU; ajustar a taxa de aprendizado, o tamanho do lote e o número de épocas de forma sistemática; repetir o treino com diferentes sementes; usar aumento de dados (rotações e deslocamentos pequenos); registrar a curva de perda por iteração; e testar o modelo em imagens de escrita real, fora do MNIST.
