from keras.models import Sequential
from keras.layers.core import Dense, Activation, Dropout
from keras.layers.recurrent import LSTM
from keras.models import model_from_json

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

    maxlen = 25

    json_file = open('../models/backwards_char_rnn.json', 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    
    model = model_from_json(loaded_model_json)
    # Load weights
    model.load_weights("../models/backwards_char_rnn.h5")

    generated = " and love's embrace"
    sentence = '$'*(maxlen - len(generated)) + generated[::-1]
    print('----- Generating with end: %s' %generated)

    diversity = 0.2
    for i in range(400):
        x = np.zeros((1, maxlen, len(chars)))
        for t, char in enumerate(sentence):
            x[0, t, char_indices[char]] = 1.

        preds = model.predict(x, verbose=0)[0]
        next_index = sample(preds, diversity)
        next_char = indices_char[next_index]
                
        generated = next_char + generated
        sentence = sentence[1:] + next_char
        
    print generated
        
