import nltk
import string
import json
import numpy as np

from hmmlearn import hmm

from sklearn.externals import joblib

import sys
sys.path.append('..')

from preprocess import split_lines

if __name__ == '__main__':
    files = ['../data/shakespeare.txt', '../data/shakespeare_xtra.txt']
    
    lines = []
    for filename in files:
        lines.extend(split_lines(filename)[::-1])
    
    print('Number sequences:', len(lines))
    
    vocab = json.load( \
        open('../models/shakespeare_words/shakespeare_vocab.json'))
    # Change to integer keys
    for k in vocab.keys():
        vocab[int(k)] = vocab.pop(k)
        
    inverted_vocab = json.load( \
        open('../models/shakespeare_words/shakespeare_inverted_vocab.json'))
    
    X = np.concatenate([[inverted_vocab[x] for x in lines[i]] \
                            for i in range(len(lines))])
    
    X = X.reshape(-1, 1) # Need column vector
    lengths = np.array([len(line) for line in lines])
    
    model = hmm.MultinomialHMM(n_components=100, n_iter=1000, verbose=True)

    with np.errstate(divide='ignore'):
        model.fit(X, lengths)
        print

    joblib.dump(model, "../models/backwards_hmm_100.pkl")
    print 'Saved model to disk'
    print

    # Test a prediction
    for i in range(14):
        sample, hidden = model.sample(5)
        generate = map(lambda x: vocab[x], sample.T[0])
        print " ".join(generate[::-1]) # Since generated in reverse
        
