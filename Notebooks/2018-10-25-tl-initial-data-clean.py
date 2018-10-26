#!/usr/bin/env python
# coding: utf-8

# # Iris

# ## Configure Notebook

# In[1]:


from itertools import count

import numpy as np
import pandas as pd
from sklearn.preprocessing import scale
from sklearn.preprocessing import power_transform
from sklearn.preprocessing import quantile_transform
from sklearn.preprocessing import label_binarize
from sklearn.utils import Bunch

from plotly.offline import init_notebook_mode
from plotly.offline import iplot
from plotly.offline import plot
from plotly.plotly import image as _; ishow = _.ishow
import plotly.graph_objs as go
import plotly.figure_factory as ff
import plotly.tools as tls


# In[2]:


# plot graphs offline/locally
init_notebook_mode()


# In[3]:


# ensure current location is project root
get_ipython().run_line_magic('cd', '"/dev/GitHub repositories/iris"')


# In[4]:


# manual sanity check
get_ipython().run_line_magic('ls', '"Data/raw"')


# ## Import Data

# In[5]:


with open('Data/raw/target_names.csv') as f:  # only 1 column, so it's basically a .txt format
    target_names = [line.rstrip() for line in f]
target = np.fromfile('Data/raw/target.csv', dtype=np.dtype(int), sep=',')
target = pd.Categorical(target)
target = target.rename_categories({code: name for code, name in enumerate(target_names)})
target = pd.Series(target, name='species')


# In[6]:


with open('Data/raw/feature_names.txt') as f:
    feature_names = [line.rstrip() for line in f]
data = np.fromfile('Data/raw/data.csv', dtype=np.dtype(float), sep=',')
data = np.reshape(data, (len(target), -1))
data = pd.DataFrame(data, columns=feature_names)


# In[7]:


df = pd.concat([data, target], axis=1)


# ## Explore Raw Distributions

# In[8]:


df


# In[9]:


df.describe()


# In[10]:


iplot(dict(
    data=[
        dict(type='violin',
             name=col,
             y=df[col],
             box=dict(visible=True))
        for col in df.columns
        if col != 'species'
    ],
    layout=dict(
        title="Population Distribution by Feature"
    )
))


# In[11]:


df.groupby(by='species')['species'].count()


# ### Plot Distributions by Feature (juxtapose categories)

# In[12]:


gb = df.groupby(by='species')

labels = [key for key, _ in gb]
titles = [col for col in df.columns if col != 'species']

groupses = ([group[col] for _, group in gb] for col in titles)
figs = [ff.create_distplot(groups, labels, bin_size=0.1) for groups in groupses]

for fig, title in zip(figs, titles):
    fig.layout.title = title
    fig.layout.yaxis['showticklabels'] = False
    
it = iter(figs)


# In[13]:


iplot(next(it))


# In[14]:


iplot(next(it))


# In[15]:


iplot(next(it))


# In[16]:


iplot(next(it))


# ### Plot Distributions by Category (juxtapose features)

# In[17]:


gb = df.groupby(by='species')

titles = [key for key, _ in gb]
labels = [col for col in df.columns if col != 'species']

def make_figure(groups, title):
    fig = tls.make_subplots(rows=1, cols=4)
    for data, label, j in zip(groups, labels, count(1)):
        trace = dict(type='violin',
                     name=label,
                     y=data,
                     box=dict(visible=True),
                     showlegend=False)
        fig.append_trace(trace, 1, j)
    fig.layout.title = title
    return fig

groupses = ([group[col] for col in labels] for _, group in gb)
figs = [make_figure(groups, title)
        for groups, title in zip(groupses, titles)]
    
it = iter(figs)


# In[18]:


iplot(next(it))


# In[19]:


iplot(next(it))


# In[20]:


iplot(next(it))


# ## Preprocess Data

# In[21]:


raw_target = df['species']
raw_features = df[[col for col in df.columns if col != 'species']]


# ### Standardize Features

# In[22]:


standardized_features = scale(raw_features)
standardized_feature_names = [name.replace('(cm)', '(stddev)') for name in feature_names]


# In[23]:


iplot(
    dict(data=[dict(type='violin',
                    name=name,
                    y=data,
                    box=dict(visible=True))
               for name, data in zip(standardized_feature_names,
                                     (standardized_features[:, j] for j in count()))
        ],
        layout=dict(
            title="Standardized Population Distribution by Feature"
        )
    )
)


# ### Power Transform Features

# In[24]:


power_transformed_features = power_transform(raw_features, standardize=True)
power_transformed_feature_names = [name.partition(' (cm)')[0] for name in feature_names]


# In[25]:


iplot(
    dict(data=[dict(type='violin',
                    name=name,
                    y=data,
                    box=dict(visible=True))
               for name, data in zip(power_transformed_feature_names,
                                     (power_transformed_features[:, j] for j in count()))
        ],
        layout=dict(
            title="Power-Transformed (and Standardized) Population Distribution by Feature"
        )
    )
)


# ### Quantile Transform Features (to Gaussian)

# In[26]:


quantile_transformed_features = quantile_transform(standardized_features, output_distribution='uniform')
quantile_transformed_feature_names = [name.partition(' (stddev)')[0] for name in standardized_feature_names]


# In[27]:


iplot(
    dict(data=[dict(type='violin',
                    name=name,
                    y=data,
                    box=dict(visible=True))
               for name, data in zip(quantile_transformed_feature_names,
                                     (quantile_transformed_features[:, j] for j in count()))
        ],
        layout=dict(
            title="Quantile-Transformed (and Standardized) Population Distribution by Feature"
        )
    )
)


# ### Encode Categorical Target

# In[28]:


codes = raw_target.values.codes
classes = list(set(codes))

encoded_target = label_binarize(codes, classes)
encoded_target_categories = raw_target.values.categories


# In[29]:


encoded_target


# ### Compose Preprocessed Dataset

# In[30]:


preprocessed = Bunch()
preprocessed['data'] = power_transformed_features
preprocessed['feature_names'] = power_transformed_feature_names
preprocessed['target'] = encoded_target
preprocessed['target_names'] = list(encoded_target_categories)


# The power-transformed feature set was chosen over the standardized-only feature
# set because upon informal visual inspection it appears to eliminate a slight but
# consistent negative skew across all features that is apparent in the standardized-
# only feature set. This suggests that an ancillary "power law" effect may indeed
# be present.
# 
# Likewise, the power-transformed feature set was chosen over the quantile-
# transformed to Gaussian feature set because the latter does not preserve the
# bimodal distribution of each of petal length and petal width, which are both
# quite clearly valuable for identifying the "setosa" target category as may be
# observed from their respective feature distribution plots.

# In[31]:


preprocessed

