import pandas as pd
from read_manual_data import data_manual_labelled
from read_zotero import read_zotero

# %% data set by Kaandorp
kaandorp_data = read_zotero('./data/Sophie2021_Zotero_Positive.csv')

# %% data set by Sophie
sophie_data = read_zotero('./data/Kaandorp2020_Zotero_Positive.csv')

# %% merge the three data sets
data_merged = pd.concat([data_manual_labelled, kaandorp_data, sophie_data])
print(data_merged.describe())

# %% save the merged data set
data_merged.to_csv("./data/data_merged.csv", index=False)


# %% data inspection
title_len = [len(x.split()) for x in data_merged['Title'].tolist()]
abstract_len = [len(x.split()) for x in data_merged['Abstract'].tolist()]

data_len = pd.DataFrame(list(zip(title_len, abstract_len)),
                        columns=['title_len', 'abstract_len'])

print(data_len)

# %%
data_len.plot.hist()


# %% training and testing split (5 iterations)

# %% data preprocessing

# model training

# prediction on unlabelled data
