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
O reconhecimento automático de caracteres manuscritos é um problema clássico de visão computacional e aprendizado de máquina, como por exemplo em aplicações de leitura de cheques bancários, a triagem de correspondências por código postal e a digitalização de documentos físicos. Por ter classes bem definidas e imagens pequenas, o reconhecimento de dígitos é muito usado para avaliar e comparar técnicas de aprendizado supervisionado.

Neste trabalho, uma Rede Neural Convolucional(CNN) foi aplicada à classificação de dígitos manuscritos de 0 a 9 a partir de imagens em tons de cinza Diferentemente de uma rede densa, a CNN utiliza camadas convolucionais que extraem padrões locais da imagem, como bordas e traços, preservando a estrutura espacial dos pixels. O objetivo é avaliar o desempenho dessa arquitetura na tarefa de reconhecimento de dígitos e analisar a influência dos hiperparâmetros de treinamento no resultado.

= Descrição do dataset
*Fonte:* foi utilizado o conjunto de dados "MNIST 60000 Hand Written Number Images", disponibilizado na plataforma Kaggle(https://www.kaggle.com/datasets/rakuraku678/mnist-60000-hand-written-number-images). Ele reproduz em formato CSV o MNIST, fonte acessada em 11/11/2026.

*Período:* não se aplica, um período de coleta, pois se trata de uma basa estática de imagens, sem dimensão temporal.

*Quantidade de amostras:* VERIFICAR DEPOIS QUAL A QUANTIDADE EXATA como ficará o texto -> o arquivo _mnist_test.csv_ OU _mnist_train.csv_ contém X amostras e 785 colunas.

*Atributos:* A primeira coluna é o rótulo(label), um inteiro de 0 a 9. As 784 colunas seguintes são os pixels de uma imagem 28X28 _1X28X28?_, em ordem de linha a linha, com intensidade de 0(fundo preto) e 255(traço branco). Cerca de 81% dos pixels têm valor 0, porque os dígitos ocupam uma pequena parte da imagem.

*Distribuição das classes:* A distribuição das classes são razoavelmente balanceadas. No arquivo _mnist_test.csv_, a menor classe é o dígito 5 (892 amostras) e a maior é o dígito 1 (1.135 amostras), conforme a @tab-classes. Por isso a acurácia é uma métrica adequada para a avaliação.

#figure(
  table(
    columns: 11,
    align: center,
    [*Dígito*], [0], [1], [2], [3], [4], [5], [6], [7], [8], [9],
    [*Amostras*], [980], [1.135], [1.032], [1.010], [982], [892], [958], [1.028], [974], [1.009],
  ),
  caption: [Distribuição das classes no arquivo _mnist_test.csv_],
) <tab-classes>

= Metodologia
== Preparação dos dados
*Seleção de atributos:* foram utilizados os 784 atributos de entrada, sem seleção prévia. Cada pixel pode carregar informação sobre o traço, e a própria rede aprende quais combinações são relevantes.

*Váriavel-alvo:* o rótulo do dígito(0 a 9), o que caracteriza uma classificação multiclasse com dez classes.

*Normalização:* os valores dos pixels foram divididos por 255, ficando entre 0 e 1, e em seguida padronizados com média 0,1307 e desvio padrão 0,3081, valores usuais do MNIST. Esse procedimento evita que valores grandes dominem o treino e tende a acelerar e estabilizar a convergência.

*Divisão dos dados:* 
// TODO: definir depois do código final. O main.py atual usa 60.000 amostras de treino e 10.000 de teste,
// sem conjunto de validação, e usa o teste a cada época para decidir a parada (MIN_ACC).
// Isso deve ser corrigido no código ou citado como limitação na Discussão.

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

