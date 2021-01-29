import pandas

# read the manually coded data set
data_manual = pandas.read_csv("./Data/first_dataset_20210128.csv")

# print column names
# print(data_manual.columns)

# rename to abstract and relevance
data_manual = data_manual.rename(columns={'Abstract Note': 'Abstract', 'Qixiang': 'Relevance'})
# print(data_manual.columns)

# keep only key title, abstract and decision
data_manual = data_manual[['Key', 'Title', 'Abstract', 'Relevance']]
# print(data_manual.columns)

# make a new data set with labelled rows: data_labelled
data_manual_labelled = data_manual.dropna(subset=['Relevance'])
# print(data_manual_labelled)

# keep only the relevant and irrelevant rows
data_manual_labelled = data_manual_labelled.loc[data_manual_labelled['Relevance'].isin(['relevant', 'irrelevant'])]
# print(data_manual_labelled)

# recode relevant to 1 and irrelevant to 0
data_manual_labelled = data_manual_labelled.replace(dict(relevant=1, irrelevant=0))
# print(data_manual_labelled)

# make a new data set with the unlabelled rows: data_unlabelled
data_unlabelled = data_manual[data_manual['Relevance'].isnull()]
# print(data_unlabelled)
