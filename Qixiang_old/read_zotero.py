import pandas

# a function to read data from zotero csv files
def read_zotero(fpath):
    data = pandas.read_csv(fpath)
    data = data.rename(columns={'Abstract Note': 'Abstract'})
    data = data[['Key', 'Title', 'Abstract']]
    data['Relevance'] = 1
    return data

