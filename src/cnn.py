import numpy as np
import pickle
from sklearn.model_selection import train_test_split
from tensorflow.keras.layers import Dense, Dropout, Embedding, Flatten
from tensorflow.keras.layers import Conv1D, MaxPooling1D
from tensorflow.keras.models import Sequential
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer

def build_cnn_classifier(vocab_size, max_length, emb_size, num_filters, kernel_size):
    model = Sequential()
    model.add(Embedding(vocab_size, emb_size, input_length=max_length)) # max_len x emb_size
    
    model.add(Conv1D(filters=num_filters, kernel_size=kernel_size, activation='relu'))
    model.add(MaxPooling1D(pool_size=2))
    model.add(Flatten())
    model.add(Dense(10, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))
    print(model.summary())
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model

def cnn_wrapper(data, feature='text', target='fraudulent', path='cnn.pickle', path2='tokenizer.pickle', epochs=10, emb_size=128, num_filters=32, kernel_size=8):
    X = data[feature]
    y = data[target]
    train_texts, test_texts, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    train_texts = np.array(train_texts); test_texts = np.array(test_texts)
    
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(train_texts)
    max_length = max([len(s.split()) for s in train_texts])
    print(max_length)

    encoded_train_texts = tokenizer.texts_to_sequences(train_texts)
    X_train = pad_sequences(encoded_train_texts, maxlen=max_length, padding='post')
    y_train = np.array(y_train)
    
    encoded_test_texts = tokenizer.texts_to_sequences(test_texts)
    X_test = pad_sequences(encoded_test_texts, maxlen=max_length, padding='post')
    y_test = np.array(y_test)
    
    vocab_size = len(tokenizer.word_index) + 1
    print(vocab_size)

    model = build_cnn_classifier(vocab_size, max_length, emb_size, num_filters, kernel_size)
    model.fit(X_train, y_train, epochs=epochs, verbose=2)
    loss, acc = model.evaluate(X_test, y_test, verbose=0)
    print('Test accuracy: %f' % acc)

    with open(path, 'wb') as f:
        pickle.dump(model, f)
    with open(path2, 'wb') as f:
        pickle.dump(tokenizer, f)