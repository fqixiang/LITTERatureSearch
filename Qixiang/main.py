import pandas as pd
#import matplotlib as plt
from model_functions import data_split, data_split_powerful, data_vectorizer, data_vectorizer_powerful, logistic_l1, find_highlight_word, feature_importance
#from read_manual_data import data_manual_labelled
from read_zotero import read_zotero
#plt.interactive(True) # set this so you can view the plot in the plot view

# %% data preparation
# data set by Kaandorp and Sophie
kaandorp_data = read_zotero('./data/Sophie2021_Zotero_Positive.csv', relevance=None)
sophie_data = read_zotero('./data/Kaandorp2020_Zotero_Positive.csv', relevance=None)

# data set scopus 2020-2021
scopus2021_data = read_zotero('./data/scopus2020-2021.csv', relevance=None)

# combine the three relevance data sets above
data_reference = pd.concat([kaandorp_data, sophie_data, scopus2021_data])

# data
data_annotated = pd.read_csv('./data/data_merged_test.csv')[['Key', 'Relevance']]

# left join the annotated data with the reference data
data_full = pd.merge(data_annotated, data_reference, how='left', on='Key')

# %% read the unannotated data set
data_all = read_zotero('./data/scopus2020-2021.csv', relevance=None)
data_complete = data_all[['Key']].merge(data_full, 'outer', 'Key')
data_complete.update(data_all)

# %%
# data_complete.to_csv('./data/data_complete_test.csv', index=False, encoding="utf-8-sig")
# data_full.to_csv('./data/data_full_test.csv', index=False, encoding="utf-8-sig")

# %%
sum(data_full.Keywords == '')

# # %% merge the three data sets
# data_merged = pd.concat([data_manual_labelled, kaandorp_data, sophie_data])
# print(data_merged.describe())

# # %% save the merged data set
# # data_merged.to_csv("./data/data_merged.csv", index=False)
#
# # %% concatenate title and abstract
# data_merged['Title_Abs'] = data_merged['Title'] + data_merged['Abstract']
#
# # %% data inspection
# title_len = [len(x.split()) for x in data_merged['Title'].tolist()]
# abstract_len = [len(x.split()) for x in data_merged['Abstract'].tolist()]
#
# data_len = pd.DataFrame(list(zip(title_len, abstract_len)),
#                         columns=['title_len', 'abstract_len'])
#
# print(data_len.head(5))
# data_len.plot.hist(bins = 50)


# # %% data preprocessing
# dataset_train, dataset_test = data_split(data_full, seed=123, test_size=.20)
#
# x_train, y_train, x_test, y_test = data_vectorizer(dataset_train, dataset_test,
#                                                    var=['Title', 'Abstract', 'Keywords', 'Venue'],
#                                                    n_features=100)

# %%
dataset_train, dataset_test, dataset_new = data_split_powerful(data_complete, seed=123, test_size=.20)

x_train, y_train, x_test, y_test, x_new = data_vectorizer_powerful(dataset_train, dataset_test, dataset_new,
                                                   var=['Title', 'Abstract', 'Keywords', 'Venue'],
                                                   n_features=100)

# %%
results = logistic_l1(x_train, y_train, x_test, y_test, scoring="f1")
print(results)

# we need:
# 1. a distribution of prediction confidence
# 2. the predicted probabilities, ranked by uncertainties
# 3. the ones predicted to be relevant
# 4.

# %%
new_prob = results['model'].predict_proba(x_new)[:,1]
new_key = dataset_new.Key.to_list()

# %%
df_prediction = pd.DataFrame({'Key': new_key, 'Prob': new_prob})

# %%
df_prediction.sort_values(by=['Prob'], ascending=False)

# %%
df_prediction = df_prediction.assign(Certainty = abs(df_prediction.Prob - 0.5))

# %%
df_prediction.Prob = df_prediction.Prob.round(2)
df_prediction.Certainty = df_prediction.Certainty.round(2)

# %%
df_prediction.sort_values(by=['Certainty'], ascending=True).Key.to_list()


# %% feature importance
word_ls, feature_fig = feature_importance(model=results['model'],
                                          x=x_train,
                                          y=y_train,
                                          threshold=0.01)

# %%
feature_fig.show()

# %%
dataset_new[dataset_new.Key == 'KSX44NIU']

# %% highlight abstract
find_highlight_word(word_ls, dataset_new.Abstract[1])

# %%
key_uncertain_ls = 'KSX44NIU'.split()
# %%
key_uncertain_ls

# %%
new_key = key_uncertain_ls[0]

# %%
new_key
# %%
new_record = dataset_new[dataset_new.Key == new_key]

# %%
dataset_new.Title[dataset_new.Key == new_key].iloc[0]

# %%
new_title = new_record.Title
# %%
new_title[0]
# %%
new_abstract = new_record.Abstract
new_link = new_record.Link

# %%
remaining_titles = data_complete.Title[data_complete.Relevance.isna()]
# %%
remaining_titles.iloc[0]



# %% training and testing split (5 iterations)



# model training

# prediction on unlabelled data


# %% join the data set with keywords, journal name and link

