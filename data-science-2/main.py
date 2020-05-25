#!/usr/bin/env python
# coding: utf-8

# # Desafio 4
# 
# Neste desafio, vamos praticar um pouco sobre testes de hipóteses. Utilizaremos o _data set_ [2016 Olympics in Rio de Janeiro](https://www.kaggle.com/rio2016/olympic-games/), que contém dados sobre os atletas das Olimpíadas de 2016 no Rio de Janeiro.
# 
# Esse _data set_ conta com informações gerais sobre 11538 atletas como nome, nacionalidade, altura, peso e esporte praticado. Estaremos especialmente interessados nas variáveis numéricas altura (`height`) e peso (`weight`). As análises feitas aqui são parte de uma Análise Exploratória de Dados (EDA).
# 
# > Obs.: Por favor, não modifique o nome das funções de resposta.

# ## _Setup_ geral

# In[1]:


import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as sct
import seaborn as sns


# In[2]:


# %matplotlib inline

# from IPython.core.pylabtools import figsize


sns.set()


# In[6]:


df = pd.read_csv("athletes.csv")


# In[7]:


def get_sample(df, col_name, n=100, seed=42):
    """Get a sample from a column of a dataframe.
    
    It drops any numpy.nan entries before sampling. The sampling
    is performed without replacement.
    
    Example of numpydoc for those who haven't seen yet.
    
    Parameters
    ----------
    df : pandas.DataFrame
        Source dataframe.
    col_name : str
        Name of the column to be sampled.
    n : int
        Sample size. Default is 100.
    seed : int
        Random seed. Default is 42.
    
    Returns
    -------
    pandas.Series
        Sample of size n from dataframe's column.
    """
    np.random.seed(seed)
    
    random_idx = np.random.choice(df[col_name].dropna().index, size=n, replace=False)
    
    return df.loc[random_idx, col_name]


# ## Inicia sua análise a partir daqui

# ### Visualizando alguns dados do dataset

# In[8]:


df.head()


# ### Principais estatísticas do dataset

# In[11]:


df.shape


# In[9]:


df.describe()


# In[10]:


df.isnull().sum()


# As variáveis de interesse (height e weight) possuem diversos valores nulos.

# In[13]:


df.dtypes


# ### Distribuição das variáveis de interesse

# In[21]:


fig, axes = plt.subplots(1,2, figsize=(12,4))
# Distribuição da altura
df.height.dropna().plot(kind='hist', ax=axes[0])
axes[0].set_xlabel('height')
# Distribuição do peso
df.weight.dropna().plot(kind='hist', ax=axes[1])
axes[1].set_xlabel('weight')


# Visualmente, podemos fazer algns comentários:
# 
# * A altura tem uma distribuição simétrica, porém a curtose indica uma conencetração de pontos na região central, indicando uma curva leptocúrtica. 
# * A distribuição do peso apresenta uma assimetria a direita.

# ## Questão 1
# 
# Considerando uma amostra de tamanho 3000 da coluna `height` obtida com a função `get_sample()`, execute o teste de normalidade de Shapiro-Wilk com a função `scipy.stats.shapiro()`. Podemos afirmar que as alturas são normalmente distribuídas com base nesse teste (ao nível de significância de 5%)? Responda com um boolean (`True` ou `False`).

# In[43]:


def q1():
    x = get_sample(df, col_name='height', n=3000, seed=42)
    shapiro = sct.shapiro(x)
    # Teste de Shapiro:
    # Se o p-value for menor que 0,05, H0 é rejeitada
    return shapiro[1]>0.05


# In[44]:


q1()


# __Para refletir__:
# 
# * Plote o histograma dessa variável (com, por exemplo, `bins=25`). A forma do gráfico e o resultado do teste são condizentes? Por que?
# * Plote o qq-plot para essa variável e a analise.
# * Existe algum nível de significância razoável que nos dê outro resultado no teste? (Não faça isso na prática. Isso é chamado _p-value hacking_, e não é legal).

# ## Questão 2
# 
# Repita o mesmo procedimento acima, mas agora utilizando o teste de normalidade de Jarque-Bera através da função `scipy.stats.jarque_bera()`. Agora podemos afirmar que as alturas são normalmente distribuídas (ao nível de significância de 5%)? Responda com um boolean (`True` ou `False`).

# In[126]:


def q2():
    x = get_sample(df, col_name='height', n=3000, seed=42)
    jb = sct.jarque_bera(x)
    # Teste de jarque_bera:
    # Se o p-value for menor que 0,05, H0 é rejeitado
    return bool(jb[1]>0.05)


# In[127]:


q2()


# __Para refletir__:
# 
# * Esse resultado faz sentido?

# ## Questão 3
# 
# Considerando agora uma amostra de tamanho 3000 da coluna `weight` obtida com a função `get_sample()`. Faça o teste de normalidade de D'Agostino-Pearson utilizando a função `scipy.stats.normaltest()`. Podemos afirmar que os pesos vêm de uma distribuição normal ao nível de significância de 5%? Responda com um boolean (`True` ou `False`).

# In[128]:


def q3():
    x = get_sample(df, col_name='weight', n=3000, seed=42)
    normal = sct.normaltest(x)
    # Se o p-value for menor que 0,05, H0 é rejeitado
    return bool(normal[1]>0.05)


# In[129]:


q3()


# __Para refletir__:
# 
# * Plote o histograma dessa variável (com, por exemplo, `bins=25`). A forma do gráfico e o resultado do teste são condizentes? Por que?
# * Um _box plot_ também poderia ajudar a entender a resposta.

# ## Questão 4
# 
# Realize uma transformação logarítmica em na amostra de `weight` da questão 3 e repita o mesmo procedimento. Podemos afirmar a normalidade da variável transformada ao nível de significância de 5%? Responda com um boolean (`True` ou `False`).

# In[130]:


def q4():
    x = get_sample(df, col_name='weight', n=3000, seed=42)
    x = np.log(x)
    normal = sct.normaltest(x)
    # Se o p-value for menor que 0,05, H0 é rejeitado
    return bool(normal[1]>0.05)


# In[131]:


q4()


# __Para refletir__:
# 
# * Plote o histograma dessa variável (com, por exemplo, `bins=25`). A forma do gráfico e o resultado do teste são condizentes? Por que?
# * Você esperava um resultado diferente agora?

# In[79]:


# Distribuição do peso
x = get_sample(df, col_name='weight', n=3000, seed=42)
x = np.log(x)
plt.hist(x, bins=25)
plt.show()


# > __Para as questão 5 6 e 7 a seguir considere todos testes efetuados ao nível de significância de 5%__.

# ## Questão 5
# 
# Obtenha todos atletas brasileiros, norte-americanos e canadenses em `DataFrame`s chamados `bra`, `usa` e `can`,respectivamente. Realize um teste de hipóteses para comparação das médias das alturas (`height`) para amostras independentes e variâncias diferentes com a função `scipy.stats.ttest_ind()` entre `bra` e `usa`. Podemos afirmar que as médias são estatisticamente iguais? Responda com um boolean (`True` ou `False`).

# In[202]:


bra = df[df.nationality=='BRA']
usa = df[df.nationality=='USA']
can = df[df.nationality=='CAN']


# In[203]:


def q5():
    test = sct.ttest_ind(bra.height.dropna(), usa.height.dropna(), equal_var=False)
    return bool(test[1]>0.05)


# In[204]:


q5()


# ## Questão 6
# 
# Repita o procedimento da questão 5, mas agora entre as alturas de `bra` e `can`. Podemos afimar agora que as médias são estatisticamente iguais? Reponda com um boolean (`True` ou `False`).

# In[205]:


def q6():
    test = sct.ttest_ind(bra.height.dropna(), can.height.dropna(), equal_var=False)
    return bool(test[1]>0.05)


# In[206]:


q6()


# ## Questão 7
# 
# Repita o procedimento da questão 6, mas agora entre as alturas de `usa` e `can`. Qual o valor do p-valor retornado? Responda como um único escalar arredondado para oito casas decimais.

# In[207]:


def q7():
    test = sct.ttest_ind(can.height.dropna(), usa.height.dropna(), equal_var=False)
    return float(round(test[1], 8))


# In[208]:


q7()


# In[ ]:





# __Para refletir__:
# 
# * O resultado faz sentido?
# * Você consegue interpretar esse p-valor?
# * Você consegue chegar a esse valor de p-valor a partir da variável de estatística?
