import nltk
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords
import string
import numpy as np
import pandas as pd

def impute(data, feature='text'):
    """Imputation"""

    # categorical features
    unspecified_impute_features = ['employment_type', 'required_education', 'industry',
                                   'required_experience', 'function', 'location', 'department']
    data[unspecified_impute_features] = data[unspecified_impute_features].fillna('Unspecified')

    # textual features
    data['company_profile'] = data['company_profile'].fillna('')
    data['description'] = data['description'].fillna('')
    data['requirements'] = data['requirements'].fillna('')
    data['text'] = data['company_profile'] + data['description'] + data['requirements']
    data['text'] = data['text'].replace({'':'Missing'})

    # regrouping
    data['benefits'] = np.where(data['benefits'].isna(), 1, 0) # missing 1
    data['required_education'] = np.where(data['required_education'].str.contains("Vocational"), 'Vocational', data['required_education'])

    # salary extraction and imputation
    data[['salary_lower', 'salary_upper']] = data['salary_range'].str.split('-', 1, expand=True)
    data['salary_lower'] = np.where(data['salary_lower'].str.isnumeric(), data['salary_lower'], np.nan)
    data['salary_lower'] = data['salary_lower'].astype(float)
    data['salary_lower'] = data['salary_lower'].fillna(data['salary_lower'].median())
    data['salary_upper'] = np.where(data['salary_upper'].str.isnumeric(), data['salary_upper'], np.nan)
    data['salary_upper'] = data['salary_upper'].astype(float)
    data['salary_upper'] = data['salary_upper'].fillna(data['salary_upper'].median())
    # normalize salary
    data['salary_lower'] = (data['salary_lower']-min(data['salary_lower']))/(max(data['salary_lower'])-min(data['salary_lower']))
    data['salary_upper'] = (data['salary_upper']-min(data['salary_upper']))/(max(data['salary_upper'])-min(data['salary_upper']))

def preprocess(data, col):
    """Preprocess text"""
    st = PorterStemmer()
    stopwords_dict = stopwords.words('english') # all stopwords in English
    data[col] = data[col].apply(lambda x: " ".join([st.stem(i) for i in x.split() if i not in stopwords_dict]).lower().translate(str.maketrans('', '', string.punctuation)))

def create_ngrams(text, n=2):
    """Create n-grams given a review"""
    unigrams = nltk.word_tokenize(text) # text should already be cleaned
    unigrams_joined = ' '.join(unigrams)

    if len(unigrams) > 1:
        bigrams = list(map(lambda x: '_'.join(x), zip(unigrams, unigrams[1:])))
    else:
        bigrams = []

    bigrams_joined = ' '.join(bigrams)

    if n == 2:
        return bigrams_joined
    elif n == 1.5:
        return unigrams_joined + ' ' + bigrams_joined

def ohe(data, cols=['required_education', 'required_experience', 'employment_type']):
    """One hot encode non-textual features"""

    dummies = pd.get_dummies(data[cols])

    data_new = pd.concat([data, dummies], axis=1)      
    data_new.drop(cols, inplace=True, axis=1)
    return data_new

def imbalance_correct(pred_prob, p=0.048378, ps=0.5):
    O = p/(1-p)
    Os = ps/(1-ps)
    corrected_pred_prob = pred_prob*O/(Os-pred_prob*(Os-O))
    return corrected_pred_prob

