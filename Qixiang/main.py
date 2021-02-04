import pandas as pd
import matplotlib as plt
from model_functions import data_split, data_vectorizer, logistic_l1
from read_manual_data import data_manual_labelled
from read_zotero import read_zotero
plt.interactive(True) # set this so you can view the plot in the plot view

# %% data set by Kaandorp and Sophie
kaandorp_data = read_zotero('./data/Sophie2021_Zotero_Positive.csv')
sophie_data = read_zotero('./data/Kaandorp2020_Zotero_Positive.csv')

# %% merge the three data sets
data_merged = pd.concat([data_manual_labelled, kaandorp_data, sophie_data])
print(data_merged.describe())

# %% save the merged data set
# data_merged.to_csv("./data/data_merged.csv", index=False)

# %% concatenate title and abstract
data_merged['Title_Abs'] = data_merged['Title'] + data_merged['Abstract']

# %% data inspection
title_len = [len(x.split()) for x in data_merged['Title'].tolist()]
abstract_len = [len(x.split()) for x in data_merged['Abstract'].tolist()]

data_len = pd.DataFrame(list(zip(title_len, abstract_len)),
                        columns=['title_len', 'abstract_len'])

print(data_len.head(5))
data_len.plot.hist(bins = 50)


# %% data preprocessing
dataset_train, dataset_test = data_split(data_merged, seed=123, test_size=.20)
print(dataset_train)

# %%
x_train, y_train, x_test, y_test = data_vectorizer(dataset_train, dataset_test, var='Title', n_features=100)

# %%
test_classification = logistic_l1(x_train, y_train, x_test, y_test, scoring="f1")
print(test_classification)

# %%



# %% training and testing split (5 iterations)

# model training

# prediction on unlabelled data
