from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
import pandas as pd
from sklearn.linear_model import LogisticRegression
from scipy.stats import uniform, randint
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import classification_report


def data_split(dataset, seed, test_size):
    dataset_train, dataset_test = train_test_split(dataset, test_size=test_size, random_state=seed)
    return dataset_train, dataset_test


def data_vectorizer(dataset_train, dataset_test, var, n_features):
    # learn from the training set
    vectorizer = CountVectorizer(lowercase=True,
                                 stop_words='english',
                                 analyzer='word',
                                 max_df=1.0,
                                 min_df=1,
                                 max_features=n_features)
    vectorizer.fit(dataset_train[var].to_list())
    feature_names = vectorizer.get_feature_names()

    # transform both training and testing set
    features_train = pd.DataFrame(vectorizer.transform(dataset_train[var]).toarray(),
                                  columns=feature_names)
    features_test = pd.DataFrame(vectorizer.transform(dataset_test[var]).toarray(),
                                 columns=feature_names)

    y_train = dataset_train['Relevance']
    y_test = dataset_test['Relevance']

    return features_train, y_train, features_test, y_test


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

    test_pred = model_lasso_tuned.predict(x_test)
    test_classification = classification_report(y_true=y_test, y_pred=test_pred)

    return test_classification
