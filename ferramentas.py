from langchain_groq import ChatGroq
from langchain.tools import tool
from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain.agents import Tool
from langchain_experimental.tools import PythonAstREPLTool

import pandas as pd
import matplotlib as plt
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

  # coleta de informações
  shape = df.shape
  columns = df.dtypes
  nulos = df.isnull().sum()
  nans_str = df.apply(lambda col: col[~col.isna()].astype(str).str.strip().str.lower().eq('nan').sum())
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