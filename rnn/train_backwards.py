from keras.models import Sequential
from keras.layers.core import Dense, Activation, Dropout
from keras.layers.recurrent import LSTM

import tensorflow as tf
import numpy as np
import random
import json
import pickle

from model import build_model, sample

if __name__=='__main__':
    files = ['../data/shakespeare.txt', '../data/shakespeare_xtra.txt']
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

    char_indices = dict((c, i) for i, c in enumerate(chars))
    indices_char = dict((i, c) for i, c in enumerate(chars))

    maxlen = 25 # Window size

    # Train in reverse, so we can construct lines from the back for rhyme
    step = 3
    sentences = []
    next_chars = []
    for i in range(0, len(text) - maxlen, step):
        sentences.append(text[i + maxlen: i: -1])
        next_chars.append(text[i])
    
    print('Number sequences:', len(sentences))
    
    # One hot encode sequences
    X = np.zeros((len(sentences), maxlen, len(chars)), dtype=np.bool)
    y = np.zeros((len(sentences), len(chars)), dtype=np.bool)

    for i, sentence in enumerate(sentences):
        for t, char in enumerate(sentence):
            X[i, t, char_indices[char]] = 1
        y[i, char_indices[next_chars[i]]] = 1
        
    model = build_model(maxlen, len(chars))
    
    # Train the model, output generated text after each iteration
    for iteration in range(1, 60):
        print
        print('-' * 50)
        print('Iteration', iteration)
        model.fit(X, y, batch_size=128, nb_epoch=1)

        for diversity in [0.2, 0.5, 1.0]:
            print
            print('----- Diversity:', diversity)
            
            generated = " and love's embrace\n"
            sentence = '$'*(maxlen - len(generated)) + generated[::-1]
            print('----- Generating with end: %s' %generated)
    
            for i in range(150):
                x = np.zeros((1, maxlen, len(chars)))
                for t, char in enumerate(sentence):
                    x[0, t, char_indices[char]] = 1.

                preds = model.predict(x, verbose=0)[0]
                next_index = sample(preds, diversity)
                next_char = indices_char[next_index]
                
                generated = next_char + generated
                sentence = sentence[1:] + next_char
        
            print generated
            print
            
            
    # serialize model to JSON
    model_json = model.to_json()
    with open("../models/char_rnn.json", "w") as json_file:
        json_file.write(model_json)
    # serialize weights to HDF5
    model.save_weights("../models/char_rnn.h5")
    print "Saved model to disk"
