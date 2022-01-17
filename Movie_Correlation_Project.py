#!/usr/bin/env python
# coding: utf-8

# In[4]:


# Import Libraries

import pandas as pd
import numpy as np
import seaborn as sns

import matplotlib.pyplot as plt
import matplotlib.mlab as mlab
import matplotlib
plt.style.use('ggplot')
from matplotlib.pyplot import figure

get_ipython().run_line_magic('matplotlib', 'inline')
matplotlib.rcParams['figure.figsize'] = (12,8) # Adjusting coefficient of the plots

# Read in data

df = pd.read_csv('/Users/AnnaBonzano/Desktop/movies.csv')


# In[5]:


# Look at the data

df.head()


# In[9]:


# Looking for any missing data

for col in df.columns:
    percent_missing = np.mean(df[col].isnull())
    print('{} - {}%'.format(col, percent_missing))


# In[8]:


df = df.dropna(how='any',axis=0) 


# In[10]:


# Data types for columns

df.dtypes


# In[11]:


# Changing data type

df['budget'] = df['budget'].astype('int64')
df['gross'] = df['gross'].astype('int64')


# In[16]:


df


# In[20]:


del df["yearcorrect"]


# In[25]:


df = df.sort_values(by=['gross'], inplace=False, ascending=False)


# In[24]:


pd.set_option('display.max_rows', None)


# In[26]:


# Drop duplicates

df.drop_duplicates()


# In[35]:


# Scatter plot with budget vs gross

plt.scatter(x=df['budget'], y=df['gross'])

plt.title('Budget vs Gross Earnings')

plt.xlabel('Budget for Film')

plt.ylabel('Gross Earnings')

plt.show()


# In[28]:


df.head()


# In[40]:


# Plot budget vs gross using seaborn

sns.regplot(x='budget', y='gross', data=df, line_kws={"color":"blue"})


# In[44]:


# Looking at correlation

df.corr(method='pearson') 


# In[46]:


correlation_matrix = df.corr(method='pearson') 

sns.heatmap(correlation_matrix, annot=True)

plt.title('Correlation Matrix for Numeric Features')

plt.xlabel('Movie Features')

plt.ylabel('Movie Features')

plt.show()


# In[47]:


# Looking at non-numeric columns

df_numerized = df

for col_name in df_numerized.columns:
    if(df_numerized[col_name].dtype == 'object'):
        df_numerized[col_name] = df_numerized[col_name].astype('category')
        df_numerized[col_name] = df_numerized[col_name].cat.codes
        
df_numerized


# In[48]:


correlation_matrix = df_numerized.corr(method='pearson') 

sns.heatmap(correlation_matrix, annot=True)

plt.title('Correlation Matrix for Numeric Features')

plt.xlabel('Movie Features')

plt.ylabel('Movie Features')

plt.show()


# In[50]:


correlation_mat = df_numerized.corr()

corr_pairs = correlation_mat.unstack()

corr_pairs


# In[51]:


sorted_pairs = corr_pairs.sort_values()

sorted_pairs


# In[52]:


high_corr = sorted_pairs[(sorted_pairs) > 0.5]

high_corr


# In[ ]:




