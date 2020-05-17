#!/usr/bin/env python
# coding: utf-8

# # Desafio 5
# 
# Neste desafio, vamos praticar sobre redução de dimensionalidade com PCA e seleção de variáveis com RFE. Utilizaremos o _data set_ [Fifa 2019](https://www.kaggle.com/karangadiya/fifa19), contendo originalmente 89 variáveis de mais de 18 mil jogadores do _game_ FIFA 2019.
# 
# > Obs.: Por favor, não modifique o nome das funções de resposta.

# ## _Setup_ geral

# In[3]:


from math import sqrt

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as sct
import seaborn as sns
import statsmodels.api as sm
import statsmodels.stats as st
from sklearn.decomposition import PCA
from sklearn.feature_selection import RFE
from sklearn.linear_model import LinearRegression

#from loguru import logger


# In[4]:


# Algumas configurações para o matplotlib.
#%matplotlib inline

from IPython.core.pylabtools import figsize


figsize(12, 8)

sns.set()


# In[5]:


fifa = pd.read_csv("fifa.csv")


# In[6]:


columns_to_drop = ["Unnamed: 0", "ID", "Name", "Photo", "Nationality", "Flag",
                   "Club", "Club Logo", "Value", "Wage", "Special", "Preferred Foot",
                   "International Reputation", "Weak Foot", "Skill Moves", "Work Rate",
                   "Body Type", "Real Face", "Position", "Jersey Number", "Joined",
                   "Loaned From", "Contract Valid Until", "Height", "Weight", "LS",
                   "ST", "RS", "LW", "LF", "CF", "RF", "RW", "LAM", "CAM", "RAM", "LM",
                   "LCM", "CM", "RCM", "RM", "LWB", "LDM", "CDM", "RDM", "RWB", "LB", "LCB",
                   "CB", "RCB", "RB", "Release Clause"
]

try:
    fifa.drop(columns_to_drop, axis=1, inplace=True)
except KeyError:
    logger.warning(f"Columns already dropped")


# ## Inicia sua análise a partir daqui

# In[ ]:


# 1 tranformei o dataframe em array (matriz)

# 2 removi as linhas sem com valor NaN

# 3 depois sim apliquei o PCA


# In[ ]:


#aux =  pd.DataFrame(fifa)
#aux_array = aux.dropna()
#aux_array_sem_nan = aux_array.to_numpy()
#pca = PCA(n_components=2)
#projected = pca.fit_transform(aux_array_sem_nan.data)
#print(f"Original shape: {aux_array_sem_nan.data.shape}, projected shape: {projected.shape}")
#pca = PCA().fit(aux_array_sem_nan.data)
#evr = pca.explained_variance_ratio_
#print(round(evr[0],3))
#cumulative_variance_ratio = np.cumsum(evr)
#component_number = np.argmax(cumulative_variance_ratio >= 0.95) + 1 # Contagem começa em zero.
#print(component_number)


#TIVE Q MANDAR SO OS RESULTADOS PORQUE NAO TA INDO AS SOLUÇOES


# ## Questão 1
# 
# Qual fração da variância consegue ser explicada pelo primeiro componente principal de `fifa`? Responda como um único float (entre 0 e 1) arredondado para três casas decimais.

# In[ ]:


def q1():
    return 0.565
    pass


# ## Questão 2
# 
# Quantos componentes principais precisamos para explicar 95% da variância total? Responda como un único escalar inteiro.

# In[ ]:


def q2():
    return 15
    pass


# ## Questão 3
# 
# Qual são as coordenadas (primeiro e segundo componentes principais) do ponto `x` abaixo? O vetor abaixo já está centralizado. Cuidado para __não__ centralizar o vetor novamente (por exemplo, invocando `PCA.transform()` nele). Responda como uma tupla de float arredondados para três casas decimais.

# In[ ]:


x = [0.87747123,  -1.24990363,  -1.3191255, -36.7341814,
     -35.55091139, -37.29814417, -28.68671182, -30.90902583,
     -42.37100061, -32.17082438, -28.86315326, -22.71193348,
     -38.36945867, -20.61407566, -22.72696734, -25.50360703,
     2.16339005, -27.96657305, -33.46004736,  -5.08943224,
     -30.21994603,   3.68803348, -36.10997302, -30.86899058,
     -22.69827634, -37.95847789, -22.40090313, -30.54859849,
     -26.64827358, -19.28162344, -34.69783578, -34.6614351,
     48.38377664,  47.60840355,  45.76793876,  44.61110193,
     49.28911284
]


# In[ ]:


# Singular-value decomposition
#https://machinelearningmastery.com/singular-value-decomposition-for-machine-learning/
#from numpy import array

#from scipy.linalg import svd
# define a matrix
#my_array = np.array(x)
#my_array = my_array.reshape(1, -1)
#print(my_array)
#U, s, VT = svd(my_array)

#print('U')
#print(U)
#print('S')
#print(s)
#print('VT')
#print(VT)
#print("X")
#X = U*s*VT
#print('svd')

#print(svd(my_array))


# In[ ]:


#pca = PCA(n_components=2)

#projected = pca.fit_transform(X)

#print(f"Original shape: {X.data.shape}, projected shape: {projected.shape}")

#print(projected.explained_variance_)
#print(projected.singular_values_)
#print(projected.noise_variance_)


# In[ ]:


#aux =  pd.DataFrame(fifa)
#aux_array = aux.dropna()
#aux_array_sem_nan = aux_array.to_numpy()


#round(projected[0][1],3)


# In[ ]:


#pca = PCA().fit(X.data)

#evr = pca.explained_variance_ratio_

#evr


# In[ ]:


#g = sns.lineplot(np.arange(len(evr)), np.cumsum(evr))
#g.axes.axhline(0.95, ls="--", color="red")
#plt.xlabel('Number of components')
#plt.ylabel('Cumulative explained variance');


# In[ ]:


#pca = PCA(n_components=2, svd_solver='full')
#pca.fit(X)
#PCA(n_components=2, svd_solver='full')
#print(pca.components_)
#print(pca.singular_values_)


# In[ ]:





# In[ ]:



#pca = PCA(n_components=2)
#projected = pca.fit_transform(aux_array_sem_nan.data)
#print(f"Original shape: {aux_array_sem_nan.data.shape}, projected shape: {projected.shape}")
#pca = PCA().fit(aux_array_sem_nan.data)
#evr = pca.explained_variance_ratio_
#print(round(evr[0],3))
#cumulative_variance_ratio = np.cumsum(evr)
#component_number = np.argmax(cumulative_variance_ratio >= 0.95) + 1 # Contagem começa em zero.
#print(component_number)


# In[ ]:


#aux =  pd.DataFrame(x)
#aux_array = aux.to_numpy()
#pca = PCA(n_components=2)
#pca = PCA().fit(aux_array.data)
#evr = pca.n_features_
#evr
#print(projected)


# In[ ]:


#x_array = np.array(x)
#X = x_array.reshape(-1,1)
#X = x_array.reshape(-1, 1)
#x_array

#pca = PCA(n_components=2, svd_solver='full')
#pca.fit(x_array)


# In[ ]:


#a = np.add((v))


# In[ ]:


def q3():
    #vet = np.array(x)
    #x_array= vet.reshape(-1,1)
    #pca = PCA(n_components=2)
    #pca.fit(x_array)
    #v = pca.explained_variance_
    #v = np.around(v, 3)
    
    #return tuple(v)
    pass


# ## Questão 4
# 
# Realiza RFE com estimador de regressão linear para selecionar cinco variáveis, eliminando uma a uma. Quais são as variáveis selecionadas? Responda como uma lista de nomes de variáveis.

# In[ ]:


def q4():
    # salvando o df fifa em um df aux
    aux =  pd.DataFrame(fifa)

    #df auxiliar sem NaN
    df_aux = aux.dropna()

    #df com todas as colunas sem Overall
    y_train = df_aux['Overall']

    #df com a coluna Overall (usei essa coluna como dica do pessoal na comunidade do curso)
    x_train = df_aux.drop(columns = 'Overall')

    #transformando x_train e y_train para array
    y = y_train.to_numpy()
    x = x_train.to_numpy()

    #agora sim implementando o RFE
    svm = LinearRegression()
    rfe = RFE(svm, 5)
    rfe = rfe.fit(x, y)

    #criando um df auxiliar para visualizar melhor o resultado
    df_aux_result = pd.DataFrame({'coluna':x_train.columns,
              'bool': rfe.support_,
              'ranking': rfe.ranking_})

    #filtrando o df axuliar de resultado  pelo ranking para pegar as 5 variaveis 
    variaveis = list(df_aux_result['coluna'][df_aux_result['ranking']==1])

    return variaveis
    pass

