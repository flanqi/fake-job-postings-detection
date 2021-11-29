import pandas as pd
import numpy as np

from sklearn.pipeline import Pipeline
from sklearn.linear_model import SGDClassifier
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.metrics import classification_report, confusion_matrix

import pickle

def svm_wrapper(data, feature, target, k=1000, max_iter=500, min_df=3, loss='hinge', alpha=1e-4, path='svm.picle'):
    """SVM pipeline"""
    X = data[feature]
    y = data[target]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    vec = TfidfVectorizer(min_df=min_df, stop_words='english', sublinear_tf=True, ngram_range=(1,2))
    score_func = SelectKBest(chi2, k=k)
    mod = SGDClassifier(loss=loss, penalty='l2', alpha=alpha, random_state=42, max_iter=max_iter, tol=None)
    pipeline = Pipeline([('vectorizer', vec), 
                         ('score_function', score_func),
                         ('model', mod)])

    model = pipeline.fit(X_train, y_train)

    print(classification_report(np.array(y_test), model.predict(X_test)))
    print(confusion_matrix(np.array(y_test), model.predict(X_test)))

    with open(path, 'wb') as f:
        pickle.dump(model, f)
