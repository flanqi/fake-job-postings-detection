# general
import pandas as pd
import numpy as np
import itertools
import warnings
import json
import pickle
import argparse

# nlp
import nltk
import gensim

# sklearn
from sklearn.pipeline import Pipeline
from sklearn.linear_model import SGDClassifier
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.metrics import classification_report, confusion_matrix

# custom modules
from src.preprocess import impute, preprocess, create_ngrams, ohe
from src.svm import svm_wrapper
from src.logistic_regression import lr_wrapper
from src.cnn import cnn_wrapper
from src.multimod import multimodal_wrapper

if __name__ == '__main__':
    warnings.filterwarnings("ignore")

    # settings
    pd.set_option('display.max_rows', 50)
    pd.set_option('display.max_columns', 20)
    pd.set_option('display.width', 100)

    # load the news dataset
    path = 'data/fake_job_postings.csv'
    df = pd.read_csv(path)
    print('Shape of dataframe is {}'.format(df.shape))
    # df = open(path, encoding='utf8').readlines()
    # df = [json.loads(x) for x in df]
    # df = pd.DataFrame(df)

    print('----- percentage for each class -----')
    print(pd.DataFrame(df.groupby('fraudulent').description.count()/len(df)).rename(columns={'description':'percentage'}))

    # concat title and content
    feature = 'text'; target = 'fraudulent'

    # upsampling
    neg = df[df.fraudulent == 0]
    pos = df[df.fraudulent == 1].sample(n=len(neg), replace=True, random_state=42)
    df = pd.concat([neg, pos])
    df = df.sample(frac=1).reset_index(drop=True) # shuffle rows
    print('----- percentage for each class after upsampling -----')
    print(pd.DataFrame(df.groupby('fraudulent').description.count()/len(df)).rename(columns={'description':'percentage'}))

    # print(df.head())
    # print(df.tail())

    # data cleaning, remove stopwords and perform stemming
    impute(df, feature) # this will automatically create text col
    preprocess(df, feature)
    print('----- preprocess finished -----')

    # create unigrams / bigrams
    df['bigram'] = list(map(lambda x: create_ngrams(x,2), df[feature]))
    df['mixedgram'] = list(map(lambda x: create_ngrams(x,1.5), df[feature]))
    print('----- bigrams and mixedgrams created -----')

    parser = argparse.ArgumentParser(description="Create and/or add data to database")
    subparsers = parser.add_subparsers(dest='subparser_name')

    svm = subparsers.add_parser("svm", description="train svm model")
    lr = subparsers.add_parser("lr", description="train logistic regression model")
    cnn = subparsers.add_parser("cnn", description="train cnn model")
    multimodal = subparsers.add_parser("multimodal", description="train multimodal svm model")
    multimodal.add_argument("--method", default="svm",
                        help="ML model to train multimodal learning")

    args = parser.parse_args()
    sp_used = args.subparser_name

    # modeling
    if sp_used == 'svm':
        params = [[feature, 'bigram', 'mixedgram'],
                  ['hinge', 'squared_hinge'], # loss
                  [1e-5,1e-6]] # alpha
        for element in itertools.product(*params):
            feat, loss, alpha = element
            print('----- training output/svm_{}_{}_{}.pickle -----'.format(feat, loss, alpha))
            svm_wrapper(data=df, feature=feat, target=target, loss=loss, alpha=alpha, path='output/svm_{}_{}_{}.pickle'.format(feat, loss, alpha))
    elif sp_used == 'lr':
        params = [[feature, 'bigram', 'mixedgram'],
                  ['l2', 'elasticnet'], # penalty
                  [1, .5]] # C
        for element in itertools.product(*params):
            feat, penalty, C = element
            print('----- training output/lr_{}_{}_{}.pickle -----'.format(feat, penalty, C))
            lr_wrapper(data=df, feature=feat, target=target, C=C, penalty=penalty, path='output/lr_{}_{}_{}.pickle'.format(feat, penalty, C))
    elif sp_used == 'cnn':
        params = [[64, 128], # emb_size
                  [16, 32], # num filters
                  [4, 8, 16]] # kernal size
        for element in itertools.product(*params):
            emb_size, num_filters, kernel_size = element
            print('----- training output/cnn_{}_{}_{}.pickle -----'.format(emb_size, num_filters, kernel_size))
            cnn_wrapper(data=df, emb_size=emb_size, num_filters=num_filters, kernel_size=kernel_size,
                         epochs=5, path='output/cnn_{}_{}_{}.pickle'.format(emb_size, num_filters, kernel_size),
                         path2='output/tokenizer_{}_{}_{}.pickle'.format(emb_size, num_filters, kernel_size))
    elif sp_used == 'multimodal':
        df = ohe(df) # one hot encoding
        print('----- one hot encoding finished -----')
        add_features = ['benefits', 'has_company_logo', 'has_questions', 'telecommuting', # binary
                        'salary_lower', 'salary_upper', # numerical
                        'required_education.*', 'required_experience.*', 'employment_type.*'] # categorical
        add_features = df.filter(regex='|'.join(add_features)).columns.to_list()

        print('----- training model -----')
        multimodal_wrapper(data=df, add_features=add_features, method=args.method, k=10, path='{}_mult10.pickle'.format(args.method))
        multimodal_wrapper(data=df, add_features=add_features, method=args.method, k=100, path='{}_mult100.pickle'.format(args.method))
        multimodal_wrapper(data=df, add_features=add_features, method=args.method, k=1000, path='{}_mult1000.pickle'.format(args.method))