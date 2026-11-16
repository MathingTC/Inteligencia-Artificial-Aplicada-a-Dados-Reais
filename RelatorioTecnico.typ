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
#ficha-catalografica()//talvez seja melhor tirar isso aqui
#folha-de-aprovacao(
  //provavelmente nao colocar isso tbm
)

#resumo[
  #lorem(100)
  #palavras-chave[#lorem(1) #lorem(1) Pimba]
]

#resumo(idioma: "en")[
  Same as before, but in english... not really
  #palavras-chave[#lorem(1) #lorem(1) AmericanPimba]
]

#outline()

= Introducao
Texto de introducao com algo 

= Desenvolvimento
Toda a explicacao do pq escolhemos esse dataset, no desenvolvimento da aplicacao, treinamento, etc., etc., etc..

= Desenvolvimento2 
Verificar se sera necessario

= Conclusao
Blah blah blah

//#bibliography("refs.bib")
//referencia para esse template: https://typst.app/universe/package/abntly
//nao entendi como usar essa funcao de referencias 



