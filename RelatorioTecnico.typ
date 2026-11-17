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

#resumo[
  #lorem(100)
  #palavras-chave[#lorem(1) #lorem(1) Pimba]
]

#resumo(idioma: "en")[
  Same as before, but in english... not really
  #palavras-chave[#lorem(1) #lorem(1) AmericanPimba]
]

#outline()

= Introducao e contexto
descricao do tema e importancia da analise

= Descricao do dataset
fonte, variaveis, periodo, quantidade de amostras

= Metodologia
arquitetura da rede, funcoes de ativacao, parametros ajustados

= Resultados obtido
metricas, graficos e comparacao entre funcoes de ativacao

= Discussao
analise critica dos resultados e limitacoes do modelo

= Conclusao
licoes aprendidas e potenciais melhorias




