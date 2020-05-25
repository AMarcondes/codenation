#!/usr/bin/env python
# coding: utf-8

# # Desafio 6
# 
# Neste desafio, vamos praticar _feature engineering_, um dos processos mais importantes e trabalhosos de ML. Utilizaremos o _data set_ [Countries of the world](https://www.kaggle.com/fernandol/countries-of-the-world), que contém dados sobre os 227 países do mundo com informações sobre tamanho da população, área, imigração e setores de produção.
# 
# > Obs.: Por favor, não modifique o nome das funções de resposta.

# ## _Setup_ geral

# In[3]:


import pandas as pd
pd.set_option('display.max_columns', 200)
import numpy as np
import seaborn as sns
import sklearn as sk
from sklearn.preprocessing import KBinsDiscretizer
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.datasets import load_digits, fetch_20newsgroups
from sklearn.feature_extraction.text import CountVectorizer

from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import TfidfVectorizer


# In[4]:


# Algumas configurações para o matplotlib.
# %matplotlib inline

# from IPython.core.pylabtools import figsize


# figsize(12, 8)

sns.set()


# In[5]:


countries = pd.read_csv("countries.csv")


# In[6]:


new_column_names = [
    "Country", "Region", "Population", "Area", "Pop_density", "Coastline_ratio",
    "Net_migration", "Infant_mortality", "GDP", "Literacy", "Phones_per_1000",
    "Arable", "Crops", "Other", "Climate", "Birthrate", "Deathrate", "Agriculture",
    "Industry", "Service"
]

countries.columns = new_column_names

countries.head(5)


# ## Observações
# 
# Esse _data set_ ainda precisa de alguns ajustes iniciais. Primeiro, note que as variáveis numéricas estão usando vírgula como separador decimal e estão codificadas como strings. Corrija isso antes de continuar: transforme essas variáveis em numéricas adequadamente.
# 
# Além disso, as variáveis `Country` e `Region` possuem espaços a mais no começo e no final da string. Você pode utilizar o método `str.strip()` para remover esses espaços.

# ## Inicia sua análise a partir daqui
# 
# ### Características principais do dataset

# In[7]:


print('O dataset contém {} linhas e {} colunas.'.format(countries.shape[0], countries.shape[1]))


# In[8]:


countries.columns


# In[9]:


countries.head()


# In[10]:


countries.Infant_mortality


# In[11]:


countries.dtypes


# In[12]:


countries.isnull().sum()


# ### Correção das colunas numéricas

# Algumas das colunas numéricas estão com vírgulas como separador decimal e são do tipo string. Devemos transfomar essas variáveis para float.
# 
# As colunas são as seguintes:
# 
# * Birthrate, Deathrate, Agriculture, Industry, Service, Literacy, Phones_per_1000, Arable, Crops, Other, Pop_density, Coastline_ratio, Net_migration, Infant_mortality.

# In[13]:


def to_float(string):
    if string is not np.nan:
        string = string.replace(',', '.')
        string = float(string)
    return string


# In[14]:


cols = ['Climate',
        'Birthrate',
        'Deathrate',
        'Agriculture',
        'Industry',
        'Service',
        'Literacy',
        'Phones_per_1000',
        'Arable',
        'Crops',
        'Other',
        'Pop_density',
        'Coastline_ratio',
        'Net_migration',
        'Infant_mortality']
countries[cols] = countries[cols].applymap(to_float)


# In[15]:


countries.dtypes


# ### Correção das colunas de strings
# 
# Como mencionado, algumas strings possuem espaços extras e estes espaços devem ser removidos.

# In[16]:


countries.Country


# In[44]:


countries[['Country','Region']] = countries[['Country','Region']].applymap(lambda x: x.strip()) 


# ## Questão 1
# 
# Quais são as regiões (variável `Region`) presentes no _data set_? Retorne uma lista com as regiões únicas do _data set_ com os espaços à frente e atrás da string removidos (mas mantenha pontuação: ponto, hífen etc) e ordenadas em ordem alfabética.

# In[45]:


def q1():
    result = sorted(countries.Region.unique().tolist())
    return result


# ## Questão 2
# 
# Discretizando a variável `Pop_density` em 10 intervalos com `KBinsDiscretizer`, seguindo o encode `ordinal` e estratégia `quantile`, quantos países se encontram acima do 90º percentil? Responda como um único escalar inteiro.

# In[46]:


def q2():
    discretizer = KBinsDiscretizer(n_bins=10, encode="ordinal", strategy="quantile")
    pop_bins = discretizer.fit_transform(countries[['Pop_density']])
    return pop_bins[pop_bins==9].shape[0]


# In[47]:


q2()


# # Questão 3
# 
# Se codificarmos as variáveis `Region` e `Climate` usando _one-hot encoding_, quantos novos atributos seriam criados? Responda como um único escalar.

# In[54]:


def q3():
    # Cada valor único de cada coluna gera um novo atributo
    # Soma-se 1 devido a presença de nulos em Climate
    result = countries.Region.nunique() + countries.Climate.nunique() + 1
    return result


# In[55]:


q3()


# ## Questão 4
# 
# Aplique o seguinte _pipeline_:
# 
# 1. Preencha as variáveis do tipo `int64` e `float64` com suas respectivas medianas.
# 2. Padronize essas variáveis.
# 
# Após aplicado o _pipeline_ descrito acima aos dados (somente nas variáveis dos tipos especificados), aplique o mesmo _pipeline_ (ou `ColumnTransformer`) ao dado abaixo. Qual o valor da variável `Arable` após o _pipeline_? Responda como um único float arredondado para três casas decimais.

# In[23]:


num_pipeline = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="median")),
    ("minmax_scaler", StandardScaler())
])


# In[24]:


test_country = [
    'Test Country', 'NEAR EAST', -0.19032480757326514,
    -0.3232636124824411, -0.04421734470810142, -0.27528113360605316,
    0.13255850810281325, -0.8054845935643491, 1.0119784924248225,
    0.6189182532646624, 1.0074863283776458, 0.20239896852403538,
    -0.043678728558593366, -0.13929748680369286, 1.3163604645710438,
    -0.3699637766938669, -0.6149300604558857, -0.854369594993175,
    0.263445277972641, 0.5712416961268142
]

test_country = np.asarray([test_country[2:]])

test_country


# In[72]:


def q4():
    # Filtrando somente as colunas numéricas
    numericals = []
    for col in countries.columns:
        if countries[col].dtypes!='object':
            numericals.append(col)
    num_pipeline = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("minmax_scaler", StandardScaler())])
    countries_pipe = countries.copy()
    countries_pipe = num_pipeline.fit_transform(countries_pipe[numericals])
    # Aplica o pipeline na lista
    result = num_pipeline.transform(test_country)
    # Encontra índice da coluna Arable
    idx = countries.columns.get_loc('Arable')
    return float(round(result[0, idx-2], 3))


# In[73]:


q4()


# ## Questão 5
# 
# Descubra o número de _outliers_ da variável `Net_migration` segundo o método do _boxplot_, ou seja, usando a lógica:
# 
# $$x \notin [Q1 - 1.5 \times \text{IQR}, Q3 + 1.5 \times \text{IQR}] \Rightarrow x \text{ é outlier}$$
# 
# que se encontram no grupo inferior e no grupo superior.
# 
# Você deveria remover da análise as observações consideradas _outliers_ segundo esse método? Responda como uma tupla de três elementos `(outliers_abaixo, outliers_acima, removeria?)` ((int, int, bool)).

# Devemos remover estes outliers de nossa análise? Vamos observar alguns dos países que tem valores altos e baixos de migração.

# In[51]:


quart1 = countries.Net_migration.quantile(0.25)
quart3 = countries.Net_migration.quantile(0.75)
iqr = quart1-quart3
out_down = countries[countries.Net_migration < quart1-1.5*iqr][['Country', 'Net_migration']]
out_up = countries[countries.Net_migration > quart3+1.5*iqr][['Country', 'Net_migration']]


# In[52]:


out_down


# In[29]:


out_up


# Primeiramente, definimos a variável Net_migration: é a diferença na taxa de imigrantes e emigrantes (pessoas saindo do país) durante um ano. Ou seja, quando há mais pessoas entrando no país, a taxa é positiva, já quando há mais pessoas saindo do país, a taxa é negativa.
# 
# Sabemos que geralmente países ricos, principalmente países europeus, recebem muitos imigrantes buscando condições melhores de vida, nestes países a taxa de migração provavelmente será positiva. Já em países pobres ocorre o contrário: pessoas saindo do país em busca de melhores condições, portanto provavelmente a taxa de migração será negativa.
# 
# Fazendo uma análise rápida, verificamos alguns pontos:
# 
# * Nos outliers superiores, onde a taxa de migração é fortemente positiva, verifica-se a presença de diversos países ricos.
# 
# * Nos outliers inferiores verifica-se a presença de diversos países com condições socioecnômicas piores.
# 
# Portanto, concluímos que esses outliers não podem ser retirados sem uma análise prévia, pois podem indicar dados reais que devem ser considerados na análise, mesmo se afastando muito do padrão dos demais dados.
# 
# Este tipo de decisão deve ser tomada com conhecimento do problema, não somente com códigos computacionais.

# In[30]:


def q5():
    quart1 = countries.Net_migration.quantile(0.25)
    quart3 = countries.Net_migration.quantile(0.75)
    iqr = quart3-quart1
    out_down = countries[countries.Net_migration < quart1-1.5*iqr].shape[0]
    out_up = countries[countries.Net_migration > quart3+1.5*iqr].shape[0]
    return (out_down, out_up, False)


# In[31]:


q5()


# ## Questão 6
# Para as questões 6 e 7 utilize a biblioteca `fetch_20newsgroups` de datasets de test do `sklearn`
# 
# Considere carregar as seguintes categorias e o dataset `newsgroups`:
# 
# ```
# categories = ['sci.electronics', 'comp.graphics', 'rec.motorcycles']
# newsgroup = fetch_20newsgroups(subset="train", categories=categories, shuffle=True, random_state=42)
# ```
# 
# 
# Aplique `CountVectorizer` ao _data set_ `newsgroups` e descubra o número de vezes que a palavra _phone_ aparece no corpus. Responda como um único escalar.

# In[134]:


def q6():
    categories = ["sci.electronics", "comp.graphics", "rec.motorcycles"]
    newsgroups = fetch_20newsgroups(subset="train", categories=categories,
                                    shuffle=True, random_state=42)
    count_vectorizer = CountVectorizer()
    x = count_vectorizer.fit_transform(newsgroups.data).toarray()
    c = count_vectorizer.vocabulary_['phone']
    # Encontrar forma mais elegante de fazer este exercício
    # Utilizando os métodos do módulo de vetorização
    teste = []
    for i, item in enumerate(x):
        if item[c] != 0:
            teste.append(item[c])
    return sum(teste)


# In[135]:


q6()


# ## Questão 7
# 
# Aplique `TfidfVectorizer` ao _data set_ `newsgroups` e descubra o TF-IDF da palavra _phone_. Responda como um único escalar arredondado para três casas decimais.

# In[210]:


def q7():
    categories = ["sci.electronics", "comp.graphics", "rec.motorcycles"]
    newsgroups = fetch_20newsgroups(subset="train", categories=categories,
                                    shuffle=True, random_state=42)
    vectorizer = TfidfVectorizer()
    x = vectorizer.fit_transform(newsgroups.data)
    c = vectorizer.vocabulary_['phone']
    # Retornamos a soma dos valores de tfidf em cada documento
    return float(round(np.sum(x[:, c]), 3))


# In[211]:


q7()


# In[ ]:





# In[ ]:




