from keras.models import Sequential
from keras.layers.core import Dense, Activation, Dropout
from keras.layers.recurrent import LSTM
from keras.models import model_from_json

import tensorflow as tf
import numpy as np
import random
import json
import pickle
import gensim

from model import build_model, sample

import sys
sys.path.append('..')

from utils import syl_count


vocab = json.load(open('../models/words/vocab.json'))
for k in vocab.keys():
    vocab[int(k)] = vocab.pop(k)

inverted_vocab = json.load(open('../models/words/inverted_vocab.json'))

meter = json.load(open('../models/words/meter.json'))
inverted_meter = json.load(\
    open('../models/words/inverted_meter.json'))
        
word2vec = gensim.models.Word2Vec.load('../models/word2vec.bin')

rhyme = json.load(open('../models/words/rhyme.json'))
inverted_rhyme = json.load( \
    open('../models/words/inverted_rhyme.json'))

pos = json.load(open('../models/words/pos.json'))
inverted_pos = json.load( \
    open('../models/words/inverted_pos.json'))



def end_next(prev_end):
    """
    Find the next end word given previous, finding a similar word that
    ends in stressed.
    """
    try:
        w, p = zip(*word2vec.most_similar(prev_end, topn=10))
    except KeyError:
        return np.random.choice(inverted_rhyme.keys())

    w = list(w)
    # Make sure it starts out with unstressed
    ends = []
    for word in w:
        if word == prev_end:
            continue
        if word not in inverted_rhyme:
            continue
        ends.append(word)

    return np.random.choice(ends)


def end_next_volta(prev_end):
    try:
        w, p = zip(*word2vec.most_similar(positive=["rich", prev_end], \
                                              negative=["poor"], topn=10))
    except KeyError:
        return np.random.choice(inverted_rhyme.keys())
        
    w = list(w)
    # Make sure it starts out with unstressed
    ends = []
    for word in w:
        if word == prev_end:
            continue
        if word not in inverted_rhyme:
            continue
        ends.append(word)

    return np.random.choice(ends)


def end_next_rhyme(prev_rhyme):
    """
    Find the next end word given previous, and a word that must rhyme 
    with it.
    """
    ending = inverted_rhyme[prev_rhyme][0]
        
    rhymes = rhyme[ending]

    threshold_similarity = 0.1
    best_words = []
    for r in rhymes:
        if r == prev_rhyme:
            continue
        try:
            sim = word2vec.similarity(prev_rhyme, r)
            if sim > threshold_similarity:
                best_words.append(r)
        except KeyError:
            # probably a stopword
            best_words.append(r)

    if len(best_words) == 0:
        return np.random.choice(rhymes)

    return np.random.choice(best_words)


if __name__=='__main__':        
    files = ['../data/shakespeare.txt', '../data/shakespeare_xtra.txt', \
                 '../data/spenser.txt']
    text = ''
    
    for filename in files:
        with open(filename) as f:
            for line in f:
                line = line.strip()
                if len(line) > 0 and not line.isdigit():
                    line = line.translate(None, ':;,.!()?&')
                    text += '$' + line.lower() + '\n'

    chars = set(text)
    print('Total chars:', len(chars))

    char_indices = dict((c, i) for i, c in enumerate(chars))
    indices_char = dict((i, c) for i, c in enumerate(chars))

    # Encodes abab cdcd efef gg rhyme scheme 
    rhyme_scheme = {0:2, 1:3, 4:6, 5:7, 8:10, 9:11, 12:13}
    
    maxlen = 25 # Window length
    json_file = open('../models/backwards_char_rnn.json', 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    
    model = model_from_json(loaded_model_json)
    # Load weights
    model.load_weights("../models/backwards_char_rnn.h5")
    
    # Ending sequence
    lines = text.split("\n")
    generated = np.random.choice([line[-25:] for line in lines])
    print('----- Generating with end: %s' % generated)
    
    end_words = [''] * 14
    end_words[-1] = generated.split(' ')[-1]
    for i in xrange(12, -1, -1):
        if i in rhyme_scheme:
            end_words[i] = end_next_rhyme( \
                end_words[rhyme_scheme[i]])
        elif i == 11:
            end_words[i] = end_next_volta(end_words[-1])
        elif i % 4 == 3:
            end_words[i] = end_next(end_words[-1])
        else:
            end_words[i] = end_next(end_words[i + 1])
            
    sentence = ''
    diversity = 0.5
    sonnet = ''
    for i in xrange(13, -1, -1):
        line = generated
        sentence = (sentence + '$\n' + generated[::-1])[-25:]

        while True:
            x = np.zeros((1, maxlen, len(chars)))
            for t, char in enumerate(sentence):
                x[0, t, char_indices[char]] = 1.

            preds = model.predict(x, verbose=0)[0]
            next_index = sample(preds, diversity)
            next_char = indices_char[next_index]
            
            # Ignore special characters
            if (next_char == '\n') or (next_char == '$'):
                sentence = sentence[1:] + next_char
                continue
            
            # Check syllables
            if (next_char == ' '): 
                syls = sum([syl_count(str(w)) for w in line.split(' ')])
                if syls >= 10:
                    break

            line = next_char + line
            sentence = sentence[1:] + next_char
            
        if ((i + 1) % 4 == 0) or (i == 13):
            line += '.\n'
        else:
            line += ',\n'
        sonnet = line + sonnet
 
        if i != 0:
            generated = ' ' + end_words[i - 1]
        
    print sonnet
        
