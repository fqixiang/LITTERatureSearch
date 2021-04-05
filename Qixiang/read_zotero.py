import pandas

# a function to read data from zotero csv files
def read_zotero(fpath, relevance=1):
    data = pandas.read_csv(fpath, encoding='utf-8')

    data['Keywords'] = data['Manual Tags'].fillna('') + data['Automatic Tags'].fillna('')
    data = data.rename(columns={'Abstract Note': 'Abstract',
                                'Publication Title': 'Venue',
                                'Url': 'Link'})
    data = data[['Key', 'Title', 'Keywords', 'Venue', 'Abstract', 'Link']]

    if relevance is None:
        return data

    else:
        data['Relevance'] = relevance
        return data

