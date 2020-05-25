#!/usr/bin/env python
# coding: utf-8

# # Desafio 1
# 
# Para esse desafio, vamos trabalhar com o data set [Black Friday](https://www.kaggle.com/mehdidag/black-friday), que reúne dados sobre transações de compras em uma loja de varejo.
# 
# Vamos utilizá-lo para praticar a exploração de data sets utilizando pandas. Você pode fazer toda análise neste mesmo notebook, mas as resposta devem estar nos locais indicados.
# 
# > Obs.: Por favor, não modifique o nome das funções de resposta.

# ## _Set up_ da análise

# In[1]:


import pandas as pd
import numpy as np


# In[3]:


black_friday = pd.read_csv("black_friday.csv")


# ## Inicie sua análise a partir daqui

# In[4]:


black_friday.head()


# In[5]:


print('Dimensões do dataset:')
print('Quantidade de linhas: {}'.format(black_friday.shape[0]))
print('Quantidade de linhas: {}'.format(black_friday.shape[1]))


# ### Colunas presentes no dataset

# In[6]:


black_friday.columns


# ### Informações gerais dos tipos de dados

# In[8]:


black_friday.dtypes


# In[24]:


black_friday.dtypes.value_counts()


# ### Contagem dos nulos

# In[10]:


black_friday.isnull().sum()


# Observa-se que duas colunas merecem mais atenção em relação aos nulos: Product_Category_2 e Product_Category_2.

# ### Resumo estatístico do dataset

# In[11]:


black_friday.describe()


# ### Número de valores únicos por cada coluna.

# In[14]:


black_friday.nunique()


# Observa-se uma grande quantidade de variáveis categóricas no dataset. Com excessão da variável `Purschase`, as demais variáveis, em uma primeira impressão, se mostram como variáveis categóricas.
# 
# A variável `Age`, por exemplo, poderia ser discreta, por se tratar da idade do usuário. Porém, ela é também categórica, pois as idades dos usuários estão divididas em faixas, como pode-se observar na célula abaixo.

# ###### black_friday.Age.unique()

# ## Questão 1
# 
# Quantas observações e quantas colunas há no dataset? Responda no formato de uma tuple `(n_observacoes, n_colunas)`.

# In[20]:


def q1():
    return black_friday.shape


# ## Questão 2
# 
# Há quantas mulheres com idade entre 26 e 35 anos no dataset? Responda como um único escalar.

# In[163]:


def q2():
    return black_friday[(black_friday.Gender=='F') & (black_friday.Age=='26-35')].shape[0]


# ## Questão 3
# 
# Quantos usuários únicos há no dataset? Responda como um único escalar.

# In[32]:


def q3():
    return black_friday.User_ID.nunique()


# ## Questão 4
# 
# Quantos tipos de dados diferentes existem no dataset? Responda como um único escalar.

# In[37]:


def q4():
    return black_friday.dtypes.nunique()


# ## Questão 5
# 
# Qual porcentagem dos registros possui ao menos um valor null (`None`, `ǸaN` etc)? Responda como um único escalar entre 0 e 1.

# In[45]:


def q5():
    return black_friday[black_friday.isnull().sum(axis=1)>=1].shape[0]/black_friday.shape[0]


# ## Questão 6
# 
# Quantos valores null existem na variável (coluna) com o maior número de null? Responda como um único escalar.

# In[160]:


def q6():
    return int(black_friday.Product_Category_3.isnull().sum())


# ## Questão 7
# 
# Qual o valor mais frequente (sem contar nulls) em `Product_Category_3`? Responda como um único escalar.

# In[69]:


def q7():
    return float(black_friday.Product_Category_3.mode())


# ## Questão 8
# 
# Qual a nova média da variável (coluna) `Purchase` após sua normalização? Responda como um único escalar.

# In[156]:


def q8():
    # Calcula o valor normalizado para a média
    return float((black_friday.Purchase.mean()-black_friday.Purchase.min())/(black_friday.Purchase.max()-black_friday.Purchase.min()))


# ## Questão 9
# 
# Quantas ocorrências entre -1 e 1 inclusive existem da variáel `Purchase` após sua padronização? Responda como um único escalar.

# In[97]:


def q9():
    # Faz a padronização da coluna
    padronizado = (black_friday.Purchase - black_friday.Purchase.mean())/black_friday.Purchase.std()
    return padronizado[(padronizado>=-1) & (padronizado<=1)].shape[0]


# ## Questão 10
# 
# Podemos afirmar que se uma observação é null em `Product_Category_2` ela também o é em `Product_Category_3`? Responda com um bool (`True`, `False`).

# In[151]:


def q10():
    nan_2 = black_friday.Product_Category_2.isnull().sum()
    # Verifica quantidade de observações onde ambos sejam nulos
    nan_23 = black_friday[black_friday.Product_Category_2.isnull() & black_friday.Product_Category_3.isnull()].shape[0]
    # Retorna True se a igualdade for verdadeira
    return bool(nan_2 == nan_23)

