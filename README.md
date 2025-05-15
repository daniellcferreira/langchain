# Assistente de Análise de Dados com IA


![Python](https://img.shields.io/badge/Python-Language-3776AB?style=flat-square&logo=python&logoColor=FFD43B)
![Streamlit](https://img.shields.io/badge/Streamlit-WebApp-FF4B4B?style=flat-square&logo=streamlit&logoColor=FFFFFF)
![LangChain](https://img.shields.io/badge/LangChain-LLM_Framework-00A6ED?style=flat-square&logo=OpenAI&logoColor=10A37F)
![Pandas](https://img.shields.io/badge/Pandas-Library-150458?style=flat-square&logo=pandas&logoColor=FFFFFF)
![Groq](https://img.shields.io/badge/Groq-API-FF6D00?style=flat-square&logo=groq&logoColor=000000)




## Descrição

Este projeto é um assistente inteligente para análise exploratória de dados via interface web. Desenvolvido com Streamlit e LangChain, ele permite que usuários façam o upload de arquivos CSV e obtenham insights automatizados através de perguntas em linguagem natural.

A aplicação usa um agente ReAct com base no modelo LLaMA 3 da Groq, oferecendo respostas contextuais e interativas. Com isso, é possível obter análises estatísticas, relatórios descritivos, criação de gráficos e respostas a perguntas específicas, tudo isso sem a necessidade de escrever uma linha de código.

É ideal para analistas, cientistas de dados, profissionais de BI e times que precisam de respostas rápidas a partir de um dataset. A interface é simples, acessível e funcional, pensada para melhorar a produtividade e a tomada de decisão baseada em dados.

## Funcionalidades

- Interface web interativa para uso direto no navegador.
- Upload de arquivos CSV para análise automatizada.
- Geração de relatórios explicativos com base nos dados carregados:
  - Relatório de informações gerais (shape, tipos de dados, valores nulos, duplicados, sugestões).
  - Relatório de estatísticas descritivas (média, mediana, moda, desvio padrão, valores extremos).
- Capacidade de responder perguntas em linguagem natural, como:
  - "Qual a média da coluna X?"
  - "Quantas categorias existem na coluna Y?"
  - "Existe correlação entre as colunas A e B?"
- Geração automática de gráficos com base em comandos simples, como:
  - "Crie um gráfico de barras com a média de vendas por região."
  - "Mostre a distribuição dos dados da coluna tempo_de_entrega."
- Interface com resposta visual clara e opção de baixar os relatórios em formato Markdown.
- Processamento de linguagem natural para interpretar comandos textuais.

## Tecnologias utilizadas

- **Python**: Linguagem de programação utilizada para o backend.
- **Streamlit**: Ferramenta de desenvolvimento para aplicações web com foco em ciência de dados.
- **LangChain**: Framework utilizado para estruturar e orquestrar o agente LLM.
- **Groq + LLaMA 3**: Modelo de linguagem utilizado para interpretação e resposta das perguntas dos usuários.
- **Pandas**: Biblioteca de análise e manipulação de dados em estruturas tabulares.

Este repositório é uma base sólida para sistemas de análise de dados assistida por IA, voltados a acelerar o ciclo de exploração, diagnóstico e visualização de datasets tabulares.
