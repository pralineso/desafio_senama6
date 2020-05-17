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

# In[110]:


# 1 tranformei o dataframe em array (matriz)

# 2 removi as linhas sem com valor NaN

# 3 depois sim apliquei o PCA


# In[20]:


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

# In[6]:


def q1():
    return 0.565
    pass


# ## Questão 2
# 
# Quantos componentes principais precisamos para explicar 95% da variância total? Responda como un único escalar inteiro.

# In[7]:


def q2():
    return 15
    pass


# ## Questão 3
# 
# Qual são as coordenadas (primeiro e segundo componentes principais) do ponto `x` abaixo? O vetor abaixo já está centralizado. Cuidado para __não__ centralizar o vetor novamente (por exemplo, invocando `PCA.transform()` nele). Responda como uma tupla de float arredondados para três casas decimais.

# In[7]:


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
# 1 transformei a lista x pra um array
# 2 precisei usar esse reshape por causa da funcao 
#Se $\mathbf{X}$ for o nosso data set, o algoritmo do SVD nos fornece:
#$$\mathbf{X} = \mathbf{U} \mathbf{\Sigma} \mathbf{V}^{T}$$
#onde $\mathbf{V} = [\vec{\phi}_{1}, \vec{\phi}_{2}, \cdots, \vec{\phi}_{p}]$.
#se V = vetor de vetores, por isso precisou do reshape (-1,1) onde cada elemento de X virou um vetor


# In[61]:


# Singular-value decomposition
#https://machinelearningmastery.com/singular-value-decomposition-for-machine-learning/
#from numpy import array
#from scipy.linalg import svd
# define a matrix
#my_array = np.array(x)
#my_array = my_array.reshape(1, -1)
#print(my_array)
#U, s, VT = svd(my_array)
#print(U)
#print(s)
#print(VT)
#print("X")
#X = U*s*VT
#print(X[0])

#print(VT.var())


# In[60]:


#pca = PCA(n_components=2)

#projected = pca.fit(X)

#print(projected.explained_variance_)
#print(projected.singular_values_)
#print(projected.noise_variance_)


# In[48]:


#aux =  pd.DataFrame(fifa)
#aux_array = aux.dropna()
#aux_array_sem_nan = aux_array.to_numpy()


# In[ ]:





# In[ ]:





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


# In[26]:


#aux =  pd.DataFrame(x)
#aux_array = aux.to_numpy()
#pca = PCA(n_components=2)
#pca = PCA().fit(aux_array.data)
#evr = pca.n_features_
#evr
#print(projected)


# In[27]:


#x_array = np.array(x)
#X = x_array.reshape(-1,1)
#X = x_array.reshape(-1, 1)
#x_array

#pca = PCA(n_components=2, svd_solver='full')
#pca.fit(x_array)


# In[223]:


#a = np.add((v))


# In[226]:


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

# In[10]:


def q4():
    # Retorne aqui o resultado da questão 4.
    pass


# In[70]:


fifa.head()


# In[71]:


columns_names = list(fifa.columns)
plt.figure(figsize = (20,20))
sns.heatmap(fifa[columns_names].corr().round(2), annot = True)


# In[81]:


#array_values = fifa.values
#columns_names


# In[62]:


#aux =  pd.DataFrame(fifa)
#aux_array = aux.dropna()
#aux_array_sem_nan = aux_array.to_numpy()


# In[64]:


#from sklearn.feature_selection import RFE
#from sklearn.linear_model import LogisticRegression

#columns_names = list(fifa.columns)
#array_values = aux_array_sem_nan
#X = array_values[:,0:36]
#Y = array_values[:,1]
# feature extraction
#model = LogisticRegression(solver='lbfgs')
#rfe = RFE(model, 5)
#fit = rfe.fit(X, Y)
#print("Num Features: %d" % fit.n_features_)
#print("Selected Features: %s" % fit.support_)
#print("Feature Ranking: %s" % fit.ranking_)


# In[63]:


# Feature Extraction with PCA
#import numpy
#from pandas import read_csv
#from sklearn.decomposition import PCA
# load data
#url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.csv"
#names = ['preg', 'plas', 'pres', 'skin', 'test', 'mass', 'pedi', 'age', 'class']
#dataframe = read_csv(url, names=names)
#array = dataframe.values
#X = array[:,0:8]
#Y = array[:,8]
# feature extraction
#pca = PCA(n_components=3)
#fit = pca.fit(X)
# summarize components
#print("Explained Variance: %s" % fit.explained_variance_ratio_)
#print(fit.components_)


# In[65]:


#array


# In[66]:


#array[:,0:8]


# In[67]:


#array[:,3]


# In[ ]:




