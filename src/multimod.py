import pandas as pd
import numpy as np
from sklearn.linear_model import SGDClassifier, LogisticRegression
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import classification_report, confusion_matrix

import pickle

def multimodal_wrapper(data, add_features, feature='text', method='svm', target='fraudulent', k=1000, max_iter=500, min_df=3, path='svm_mult.pickle'):
    """SVM multimodal pipeline"""
    X = data[[feature]+add_features]
    y = data[target]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    vec = TfidfVectorizer(min_df=min_df, stop_words='english', sublinear_tf=True, ngram_range=(1,2))
    score_func = SelectKBest(chi2, k=k)
    if method=='svm':
        mod = SGDClassifier(loss='squared_hinge', penalty='l2', alpha=1e-5, random_state=42, max_iter=max_iter, tol=None)
    else:
        mod = LogisticRegression(random_state=42, max_iter=max_iter, C=1, penalty='elasticnet', solver='saga', l1_ratio=0.5)
    
    # save tf-idf to dataframe
    X_train1 = vec.fit_transform(X_train[feature])
    X_train1 = score_func.fit_transform(X_train1, y_train)
    train_data = pd.DataFrame(X_train1.toarray(), columns=score_func.get_support(indices=True))

    y_train = y_train.reset_index(); y_train.drop(columns={'index'}, inplace=True)
    X_train = X_train.reset_index(); X_train.drop(columns={'index'}, inplace=True)
    train_data = train_data.reset_index(); train_data.drop(columns={'index'}, inplace=True)
    train_data = pd.concat([y_train, X_train[add_features], train_data], axis=1) # append tf-idf features to meda features and target

    model = mod.fit(train_data.iloc[:,1:], train_data[target]) # model training

    # tranform test data for prediction and evaluation
    X_test1 = vec.transform(X_test[feature])
    X_test1 = score_func.transform(X_test1)
    test_data = pd.DataFrame(X_test1.toarray(), columns=score_func.get_support(indices=True))

    y_test = y_test.reset_index(); y_test.drop(columns={'index'}, inplace=True)
    X_test = X_test.reset_index(); X_test.drop(columns={'index'}, inplace=True)
    test_data = test_data.reset_index(); test_data.drop(columns={'index'}, inplace=True)
    test_data = pd.concat([y_test, X_test[add_features], test_data], axis=1) # append tf-idf features to meda features and target

    print(classification_report(np.array(y_test), model.predict(test_data.iloc[:,1:])))
    print(confusion_matrix(np.array(y_test), model.predict(test_data.iloc[:,1:])))

    with open(path, 'wb') as f:
        pickle.dump(model, f)

