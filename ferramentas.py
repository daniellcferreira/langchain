from langchain_groq import ChatGroq
from langchain.tools import tool
from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain.agents import Tool
from langchain_experimental.tools import PythonAstREPLTool

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
import os
from dotenv import load_dotenv

# obtenção da chave da api
load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

# configurações do LLM
llm = ChatGroq(
  api_key=GROQ_API_KEY,
  model="llama3-70b-8192",
  temperature=0
)

# relatorio informações
@tool
def informacoes_dataframe(pergunta: str, df: pd.DataFrame) -> str:
  """Utilize esta ferramenta sempre que o usuário solicitar informações gerais sobre o dataframe,
        incluindo número de colunas e linhas, nomes das colunas e seus tipos de dados, contagem de dados
        nulos e duplicados para dar um panomara geral sobre o arquivo."""

  # coleta de informações
  shape = df.shape
  columns = df.dtypes.to_string()
  nulos = df.isnull().sum().to_string()
  nans_str = df.apply(lambda col: col[~col.isna()].astype(str).str.strip().str.lower().eq('nan').sum()).to_string()
  duplicados = df.duplicated().sum()

  # prompt de resposta
  template_resposta = PromptTemplate(
    template="""
    Você é um analista de dados encarregado de apresentar um resumo informativo sobre um DataFrame
    a partir de uma {pergunta} feita pelo usuário.

    A seguir, você encontrará as informações gerais da base de dados:

    ================ INFORMAÇÕES DO DATAFRAME ================

    Dimensões: {shape}
    Colunas e tipos de dados: {columns}
    Valores nulos por colunas: {nulos}
    Strings 'nan' (qualquer capitalização) por coluna: {nans_str}
    Linhas duplicadas: {duplicados}

    ==========================================================

    Com base nessas informações, escreva um resuma claro e organizado contendo:

    1. Um titulo: ### Relatório de informações gerais sobre o dataset
    2. A dimensão total do DataFrame;
    3. A descrição de cada coluna (incluindo nome, tipo de dado e o que aquela coluna é)
    4. As colunas que contém dados nulos, com a respectiva quantidade.
    5. As colunas que contém string 'nan', com a respectiva quantidade.
    6. E a existência (ou não) de dados duplicados.
    7. Escreva um parágrafo sobre análises que podem ser feitas com esses dados.
    8. Escreva um parágrafo sobre tratamentos que podem se feitos nos dados.
   """,
   input_variables=["pergunta", "shape", "columns", "nulos", "nans_str", "duplicados"]
   )
  
  cadeia = template_resposta | llm | StrOutputParser()

  resposta = cadeia.invoke({
    "pergunta": pergunta,
    "shape": shape,
    "columns": columns,
    "nulos": nulos,
    "nans_str": nans_str,
    "duplicados": duplicados
  })

  return resposta

# relatorio estatístico
@tool
def resumo_estatistico(pergunta: str, df: pd.DataFrame) -> str:
  """
    Utilize esta ferramenta sempre que o usuário solicitar um resumo estatístico completo e descritivo da base de dados,
    incluindo várias estatísticas (média, desvio padrão, mínimo, máximo etc.).
    Não utilize esta ferramenta para calcular uma única métrica como 'qual é a média de X' ou 'qual a correlação das variáveis'.
    """

  # coleta de estatísticas descritivas
  estatisticas_descritivas = df.describe(include='number').transpose().to_string()

  # prompt de resposta
  template_resposta = PromptTemplate(
    template="""
      Você é um analista de dados encarregado de interpretar resultados estatísticos de uma base de dados
      a partir de uma {pergunta} feita pelo usuário.

      A seguir, você encontrará as estastísticas descritivas de base de dados:

      ================ ESTATÍSTICAS DESCRITIVAS ================

      {resumo}

      ==========================================================

      Com base nesses dados, elabore um resumo explicativo com linguagem clara, acessível e fluida, destacando
      os principais pontos dos resultados. Inclua:

      1. Um titulo: ### Relatório de estatísticas descritivas
      2. Um visão geral das estatísticas das colunas numéricas
      3. Um parágrafo sobre casa uma das colunas, comentando informações sobre seus valores.
      4. Indentificação de possíveis outliers com base nos valores mínimo e máximo
      5. Recomendações de próximos passos na análise com base nos padrões indentificados
    """,
    input_variables=["pergunta", "resumo"]
    )
  
  cadeia = template_resposta | llm | StrOutputParser()

  resposta = cadeia.invoke({
    "pergunta": pergunta,
    "resumo": estatisticas_descritivas
  })

  return resposta

# gerador de gráficos
@tool
def gerar_grafico(pergunta: str, df: pd.DataFrame) -> str:
  """
  Utilize esta ferramenta sempre que o usuário solicitar um gráfico a partir de um DataFrame pandas (`df`) com base em uma instrução do usuário.
  A instrução pode conter pedidos como: 'Crie um gráfico da média de tempo de entrega por clima','Plote a distribuição do tempo de entrega'"
  ou "Plote a relação entre a classifição dos agentes e o tempo de entrega. Palavras-chave comuns que indicam o uso desta ferramenta incluem:
  'crie um gráfico', 'plote', 'visualize', 'faça um gráfico de', 'mostre a distribuição', 'represente graficamente', entre outros."""

  # captura informações sobre os dataframe
  colunas_info = "\n".join([f"- {col} ({dtype})" for col, dtype in df.dtypes.items()])
  amostra_dados = df.head(3).to_dict(orient='records')

  # template otimizado para geração do codigo de graficos
  template_resposta = PromptTemplate(
    template="""
    Você é uma especialista em vizualização de dados. Sua tarefa é gerar **apenas código Python** para plotar uma grafico com base na solicitação do usuário.

    ### Solicitação do usuário:
    "{pergunta}"

    ### Metadados do DataFrame:
    {colunas}

    ### Amostra dos dados (3 primeiras linhas):
    {amostra}

    ### Instruções obrigatórias:
    1. Use as bibliotecas `matplotlib.pyplot` (como `plt`) e `seaborn` (como `sns`)
    2. Defina o tema com `sns.set_theme()`
    3. Certifique-se de que todas as colunas mencionadas na solicitação existem no DataFrame chamado `df`
    4. Escolha o tipo de gráfico adequado conforme a análise solicitada:
    - **Distribuição de variaveis numericas**: `histplot`, `kdeplot`, `boxplot` ou `violinplot`
    - **Distrubuição de variaveis categoricas**: `countplot`
    - **Comparação entre categorias**: `barplot`
    - **Relação entre variaveis**: `scatterplot` ou `lineplot`
    - **Séries temporais**: `lineplot`, com o eixo X formatado como datas
    5. Configure o tamanho do grafico com `figsize=(8, 4)`
    6. Adicione titulo e rotulos (`labels`) apropriados aos eixos
    7. Posicione o titulo à esquerda com `loc='left'`, deixe o `pad=20` e use `fontsize=14`
    8. Mantenha os ticks eixo X sem rotação com `plt.xticks(rotation=0)`
    9. Remova as bordas superior e direita do grafico com `sns.despine()`
    10. Finalize o codigo com `plt.show()`

    Retorne APENAS o codigo Python, sem nenhum texto adicional ou explicação.

    Código Python: 
    """,
    input_variables=["pergunta", "colunas", "amostra"]
    )
  
  # gera o codigo
  cadeia = template_resposta | llm | StrOutputParser()
  codigo_bruto = cadeia.invoke({
    "pergunta": pergunta,
    "colunas": colunas_info,
    "amostra": amostra_dados
  })

  # limpa o codigo gerado
  codigo_limpo = codigo_bruto.replace("```python", "").replace("```", "").strip()

  # tenta executar o codigo para validação
  exec_globals = {'df': df, 'plt': plt, 'sns': sns}
  exec_locals = {}
  exec(codigo_limpo, exec_globals, exec_locals)

  # mostra o grafico
  fig = plt.gcf()
  st.pyplot(fig)

  return ""

# função para criar ferramentas
def criar_ferramentas(df):
  ferramenta_informacoes_dataframe = Tool(
    name="Informações DataFrame",
    func=lambda pergunta:informacoes_dataframe.run({"pergunta": pergunta, "df": df}),
    description="""Utilize esta ferramenta sempre que o usuario solicitar infomações gerais sobre o dataframe,
    incluindo numero de colunas e linhas, nomes das colunas e seus tipos de dados, contagem de dados
    nulos e duplicados para dar um panorama geral sobre o arquivo.""",
    return_direct=True
  )

  ferramenta_resumo_estatistico = Tool(
    name="Resumo Estatístico",
    func=lambda pergunta:resumo_estatistico.run({"pergunta": pergunta, "df": df}),
    description="""Utilize esta ferramenta sempre que o usuario solicitar um resumo estatistico completo e descritivo da base de dados,
    incluindo varias estatisticas (media, desvio padrão, minimo, maximo etc.) e/ou multiplas colunas numericas.
    Não ultilize esta ferramenta para calcular uma unica metrica como 'qual é a media de X' ou 'qual a correlação das variaveis'.
    Para isso, use a ferramenta_python.""",
    return_direct=True
  )

  ferramenta_gerar_grafico = Tool(
    name="Gerar Grafico",
    func=lambda pergunta:gerar_grafico.run({"pergunta": pergunta, "df": df}),
    description="""Utilize esta ferramenta sempre que o usuario solicitar um grafico a partir de um DataFrame pandas (`df`) com base em uma instrução do usuario.
    A instrução pode conter pedidos como: 'Crie um grafico da media de tempo de entrega por clima', 'Plote a distriuição do tempo de entrega'
    ou Plote a relação entre classificação dos agente e o tempo de entrega. Palavras-chave comuns que indicam o uso desta ferramenta incluem:
    'crie um grafico', 'plote', 'vizualize', 'faça um grafico de', 'mostre a distribuição', 'represente graficamente' entre outros.""",
    return_direct=True
  )

  ferramenta_codigos_python = Tool(
    name="Codigos Python",
    func=PythonAstREPLTool(locals={"df": df}),
    description="""Utilize esta ferramenta sempre que o usuario solicitar calculos, consultas ou transformações especificas usando Python diretamente sobre o DataFrame `df`.
    Exemplos de uso incluem: "Qual é a media da colunaX?", "Quais são os valores unicos da coluna Y?", "Qual o correlação entre A e B?".
    Evite utilizar esta ferramenta para solicitaç~eos mais amplas ou descritivas, como infomações gerais sobre o dataframe, resumo estatistico completos ou geração de graficos - nesses casos, use as ferramentas apropriadas."""
  )

  return [
    ferramenta_informacoes_dataframe,
    ferramenta_resumo_estatistico,
    ferramenta_gerar_grafico,
    ferramenta_codigos_python
  ]
