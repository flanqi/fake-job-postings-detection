import pandas as pd
import numpy as np

from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.metrics import classification_report, confusion_matrix

import pickle

def lr_wrapper(data, feature, target, score_func=chi2, k=1000, norm='l2', C=1, max_iter=500, penalty='l2', path='lr.picle'):
    """Logistic regression pipeline"""
    X = data[feature]
    y = data[target]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    vec = TfidfVectorizer(min_df=3, stop_words='english', sublinear_tf=True, norm=norm, ngram_range=(1,2))
    score_func = SelectKBest(score_func, k=k)
    mod = LogisticRegression(random_state=42, max_iter=max_iter, C=C, penalty=penalty, solver='saga', l1_ratio=0.5)
    pipeline = Pipeline([('vectorizer', vec), 
                         ('score_function', score_func),
                         ('model', mod)])

    model = pipeline.fit(X_train, y_train)

    print(classification_report(np.array(y_test), model.predict(X_test)))
    print(confusion_matrix(np.array(y_test), model.predict(X_test)))

    with open(path, 'wb') as f:
        pickle.dump(model, f)