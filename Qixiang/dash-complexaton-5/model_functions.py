from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
import pandas as pd
from sklearn.linear_model import LogisticRegression
from scipy.stats import uniform, randint
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import accuracy_score, balanced_accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from nltk.stem.porter import *
import plotly.graph_objects as go
from sklearn.inspection import permutation_importance


def data_split(dataset, seed, test_size):
    dataset_train, dataset_test = train_test_split(dataset, test_size=test_size, random_state=seed)
    return dataset_train, dataset_test

def data_split_powerful(dataset, seed, test_size):
    # data for the modelling
    df_model = dataset.dropna(subset=['Relevance'])

    # data to be predicted
    df_new = dataset[dataset.Relevance.isna()]

    dataset_train, dataset_test = train_test_split(df_model, test_size=test_size, random_state=seed)

    return dataset_train, dataset_test, df_new

def data_vectorizer(dataset_train, dataset_test, var, n_features):
    # learn from the training set
    vectorizer = CountVectorizer(lowercase=True,
                                 stop_words='english',
                                 analyzer='word',
                                 max_df=1.0,
                                 min_df=1,
                                 max_features=n_features)

    feature_columns_train = dataset_train[var].fillna('').agg(' '.join, axis=1)
    feature_columns_test = dataset_test[var].fillna('').agg(' '.join, axis=1)

    vectorizer.fit(feature_columns_train.to_list())
    feature_names = vectorizer.get_feature_names()

    # transform both training and testing set
    features_train = pd.DataFrame(vectorizer.transform(feature_columns_train).toarray(),
                                  columns=feature_names)
    features_test = pd.DataFrame(vectorizer.transform(feature_columns_test).toarray(),
                                 columns=feature_names)

    y_train = dataset_train['Relevance']
    y_test = dataset_test['Relevance']

    return features_train, y_train, features_test, y_test


def data_vectorizer_powerful(dataset_train, dataset_test, dataset_new, var, n_features):
    # learn from the training set
    vectorizer = CountVectorizer(lowercase=True,
                                 stop_words='english',
                                 analyzer='word',
                                 max_df=1.0,
                                 min_df=1,
                                 max_features=n_features)

    feature_columns_train = dataset_train[var].fillna('').agg(' '.join, axis=1)
    feature_columns_test = dataset_test[var].fillna('').agg(' '.join, axis=1)
    feature_columns_new = dataset_new[var].fillna('').agg(' '.join, axis=1)

    vectorizer.fit(feature_columns_train.to_list())
    feature_names = vectorizer.get_feature_names()

    # transform both training and testing set
    features_train = pd.DataFrame(vectorizer.transform(feature_columns_train).toarray(),
                                  columns=feature_names)
    features_test = pd.DataFrame(vectorizer.transform(feature_columns_test).toarray(),
                                 columns=feature_names)
    features_new = pd.DataFrame(vectorizer.transform(feature_columns_new).toarray(),
                                 columns=feature_names)

    y_train = dataset_train['Relevance']
    y_test = dataset_test['Relevance']

    return features_train, y_train, features_test, y_test, features_new


def logistic_l1(x_train, y_train, x_test, y_test, scoring):
    # initiate model
    model_lasso = LogisticRegression(penalty='l1',
                                     fit_intercept=True,
                                     class_weight='balanced',
                                     random_state=123,
                                     solver='liblinear',
                                     max_iter=100)

    # hyper-parameter search
    params = {'C': uniform(1e-5, 1)}
    search_Lasso = RandomizedSearchCV(model_lasso,
                                      param_distributions=params,
                                      random_state=42,
                                      n_iter=200,
                                      cv=5,
                                      verbose=1,
                                      n_jobs=2,
                                      scoring=scoring,
                                      return_train_score=True)
    search_Lasso.fit(x_train, y_train)

    # fit the best parameter value to the model
    model_lasso_tuned = LogisticRegression(penalty='l1',
                                           C=search_Lasso.best_params_['C'],
                                           fit_intercept=True,
                                           class_weight='balanced',
                                           random_state=123,
                                           solver='liblinear',
                                           max_iter=100)

    model_lasso_tuned.fit(x_train, y_train)
    # train_pred = model_lasso_tuned.predict(x_train)
    # train_classification = classification_report(y_true=y_train, y_pred=train_pred)

    test_pred_class = model_lasso_tuned.predict(x_test)
    test_pred_prob = model_lasso_tuned.predict_proba(x_test)[:,1]

    accuracy = accuracy_score(y_true=y_test, y_pred=test_pred_class)
    balanced_acc = balanced_accuracy_score(y_true=y_test, y_pred=test_pred_class)
    precision = precision_score(y_true=y_test, y_pred=test_pred_class, pos_label=1)
    recall = recall_score(y_true=y_test, y_pred=test_pred_class, pos_label=1)
    f1 = f1_score(y_true=y_test, y_pred=test_pred_class, pos_label=1)
    auc = roc_auc_score(y_true=y_test, y_score=test_pred_prob)

    results = {'predicted_class': test_pred_class,
               'predicted_prob': test_pred_prob,
               'accuracy': accuracy,
               'balanced_acc': balanced_acc,
               'precision': precision,
               'recall': recall,
               'f1': f1,
               'auc': auc,
               'model': model_lasso_tuned}

    return results


def find_highlight_word(important_words, text):
    stemmer = PorterStemmer()

    text_split = text.split()
    text_split_stemmed = [stemmer.stem(word) for word in text_split]
    important_words_stemmed = [stemmer.stem(word) for word in important_words]

    for i, word in enumerate(text_split_stemmed):
        if word in important_words_stemmed:
            text_split[i] = '**{}**'.format(text_split[i])

    text_highlighted = ' '.join(word for word in text_split)

    return text_highlighted



def feature_importance(model, x, y, threshold=0.01):

    result = permutation_importance(model,
                                    x,
                                    y,
                                    n_repeats=10,
                                    scoring='f1',
                                    random_state=42)

    retained_word_index = result.importances_mean > threshold
    perm_sorted_idx = result.importances_mean[retained_word_index].argsort()

    word_ls = x.columns[retained_word_index][perm_sorted_idx]

    fig = go.Figure(go.Bar(
        x=result.importances_mean[retained_word_index][perm_sorted_idx],
        y=word_ls,
        orientation='h'
        ))

    return word_ls, fig