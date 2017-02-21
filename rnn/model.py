from keras.models import Sequential
from keras.layers.core import Dense, Activation, Dropout
from keras.layers.recurrent import LSTM
from keras import backend as K

import tensorflow as tf
import numpy as np
import random
import json
import pickle


def build_model(window, len_chars):
    model = Sequential()
    model.add(LSTM(512, return_sequences=True, input_shape=(window, len_chars)))
    model.add(Dropout(0.2))
    model.add(LSTM(512, return_sequences=False))
    model.add(Dropout(0.2))
    model.add(Dense(len_chars))
    model.add(Activation('softmax'))

    model.compile(loss='categorical_crossentropy', optimizer='rmsprop')
    
    return model

def sample(preds, temperature=1.0):
    # helper function to sample an index from a probability array
    preds = np.asarray(preds).astype('float64')
    preds = np.log(preds) / temperature
    
    exp_preds = np.exp(preds)
    preds =  exp_preds / np.sum(exp_preds)
    
    return np.argmax(np.random.multinomial(1, preds, 1))


if __name__ == '__main__':
    files = ['../data/shakespeare.txt']
    text = ''

    for filename in files:
        with open(filename) as f:
            for line in f:
                line = line.strip()
                if len(line) > 0 and not line.isdigit():
                    line = line.translate(None, ':;,.!()?')
                    text += '$' + line.lower() + '\n'

    chars = set(text)
    print('Total chars:', len(chars))
    
    # Build character lookup table
    char_indices = dict((c, i) for i, c in enumerate(chars))
    indices_char = dict((i, c) for i, c in enumerate(chars))
    
    # Build training sequences
    maxlen = 25
    step = 2
    sentences = []
    next_chars = []
    for i in range(0, len(text) - maxlen, step):
        sentences.append(text[i: i + maxlen])
        next_chars.append(text[i + maxlen])
    
    print('Number sequences:', len(sentences))

    # One-hot encode sequences
    X = np.zeros((len(sentences), maxlen, len(chars)), dtype=np.bool)
    y = np.zeros((len(sentences), len(chars)), dtype=np.bool)

    for i, sentence in enumerate(sentences):
        for t, char in enumerate(sentence):
            X[i, t, char_indices[char]] = 1
        y[i, char_indices[next_chars[i]]] = 1
        
    model = build_model(maxlen, len(chars))
    
    # train the model, output generated text after each iteration
    for iteration in range(1, 60):
        print
        print('-' * 50)
        print('Iteration', iteration)
        model.fit(X, y, batch_size=128, nb_epoch=1)

        generated = '$'*17 + 'love is '
        sentence = generated
        
        print('----- Generating with start: %s \n' %generated)

        for diversity in [0.2, 0.5, 1.0]:
            print
            print('----- Diversity:', diversity)

            sys.stdout.write(generated[17:])
            for i in range(150):
                x = np.zeros((1, maxlen, len(chars)))
                for t, char in enumerate(sentence):
                    x[0, t, char_indices[char]] = 1.

                preds = model.predict(x, verbose=0)[0]
                next_index = sample(preds, diversity)
                next_char = indices_char[next_index]

                generated += next_char
                sentence = sentence[1:] + next_char

                if (next_char != '$'):
                    sys.stdout.write(next_char)
                    sys.stdout.flush()
            print
