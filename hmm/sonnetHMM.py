import numpy as np
import json
import gensim
import re

from sklearn.externals import joblib

import sys
sys.path.append('..')

from utils import syl_count, random_pick 

class BackwardsSonnetHMM:
    
    def __init__(self, A, O, A_start):
        """
        Accepts numpy arrays as arguments.

        Parameters:
           A:           Transition matrix.
           O:           Observation matrix.
           A_start:     Starting transition probabilities.
           hidden:      Number of states.
           obs:         Number of observations.
        """

        self.A = np.array(A)
        self.O = np.array(O)
        self.A_start = np.array(A_start)
        
        self.hidden, self.obs = self.O.shape
        
        self.vocab = json.load( \
            open('../models/words/vocab.json'))
        # Change to integer keys
        for k in self.vocab.keys():
            self.vocab[int(k)] = self.vocab.pop(k) 
        
        self.inverted_vocab = json.load( \
            open('../models/words/inverted_vocab.json'))
        
        self.meter = json.load( \
            open('../models/words/meter.json'))
        self.inverted_meter = json.load( \
            open('../models/words/inverted_meter.json'))
        
        self.rhyme = json.load( \
            open('../models/words/rhyme.json'))
        self.inverted_rhyme = json.load( \
            open('../models/words/inverted_rhyme.json'))

        self.pos = json.load( \
            open('../models/words/pos.json'))
        self.inverted_pos = json.load( \
            open('../models/words/inverted_pos.json'))
        
        self.word2vec = gensim.models.Word2Vec.load( \
            '../models/word2vec.bin')
        

    def filter_next(self, num_syllables, prev_word, probs):
        """
        Filters possible words to preserve meter and syllable count of the line,
        and normalizes the new probabilities.
        """
        new_probs = np.copy(probs)
        
        # Filter based on meter, and keep syllables 10
        invalid_meter = []
        for k in self.meter.keys():
            m = map(int, k.split(','))
            if m[-1] != ((num_syllables + 1) % 2):
                invalid_meter.extend([self.inverted_vocab[w] for w in self.meter[k]])
    
            if len(m) + num_syllables > 10:
                invalid_meter.extend([self.inverted_vocab[w] for w in self.meter[k]])
                
        new_probs[invalid_meter] = 0
        
        # prioritizes preserving meter
        if np.sum(new_probs) == 0:
            for k in self.meter.keys():
                m = map(int, k.split(','))
                if m[-1] == ((num_syllables + 1) % 2) and len(m) + num_syllables <= 10:
                    new_probs[[self.inverted_vocab[w] for w in self.meter[k]]] = 1e-5
            
        # grammar rules
        
        # do a few cases to implement some basic grammar rules. note this is 
        # training backwards so the rules are a little weird.
        # need to multiply parts of speech by probability transition matrix?
        # requires tagging a POS with cmu nltk pos_tag
            
        invalid_pos = []
        for k in self.inverted_pos[prev_word]:
            for tag in self.pos.keys():
                if k[:2] == tag[:2]:
                    # No consecutive POS
                    invalid_pos.extend([self.inverted_vocab[w] for w in self.pos[tag]])
            
            # Not preposition before verb
            if k[:2] == 'VB':
                invalid_pos.extend([self.inverted_vocab[w] for w in self.pos['IN']])

        for i in invalid_pos:
            new_probs[i] = min(new_probs[i], 1e-20)

        for word in self.inverted_vocab.keys():
            try:
                if self.word2vec.similarity(word, prev_word) < 0.01:
                    new_probs[self.inverted_vocab[word]] = min(\
                        new_probs[self.inverted_vocab[word]], 1e-3)
            except:
                # Stopwords, no need to decrease probability
                continue

        with np.errstate(divide='ignore',invalid='ignore'):
            new_probs = np.divide(new_probs, np.sum(new_probs))
        
            new_probs[new_probs == np.inf] = 0
            new_probs = np.nan_to_num(new_probs)
    
        return new_probs


    def generate_line(self, end_word):
        """
        Generates a single line, given the last word.
        """

        emission = []
        num_syllables = 0
        
        last = self.inverted_vocab[end_word]
        state = random_pick(range(self.hidden), \
                                np.divide(self.O[:, last], np.sum(self.O[:, last])))
            
        num_syllables += len(self.inverted_meter[end_word][0].split(','))
        emission.append(end_word)

        prev_word = end_word
        while num_syllables < 10:
            # Sample next observation.
            next_probs = self.filter_next(num_syllables, prev_word, self.O[state, :])    
            next_obs= random_pick(range(self.obs), next_probs)
            
            next_word = self.vocab[next_obs]    
            # If only ' or - show up in word, then skip
            if not re.search('[a-z]+', next_word): 
                continue
                
            emission.append(next_word)
            stresses = self.inverted_meter[next_word][0].split(',')
            
            num_syllables += len(stresses)
            prev_word = next_word
            
            next_state = random_pick(range(self.hidden), self.A[state, :])
            state = next_state
                
        return emission[::-1]

    
    def end_next_volta(self, prev_end):
        try:
            w, p = zip(*self.word2vec.most_similar(positive=["rich", prev_end], \
                                                       negative=["poor"], topn=10))
        except KeyError:
            return np.random.choice(self.inverted_rhyme.keys())
        
        w = list(w)
        # Make sure it starts out with unstressed
        ends = []
        for word in w:
            try:
                stresses = self.inverted_meter[word][0].split(',')
            except:
                continue

            if word not in self.inverted_rhyme:
                continue

            if (stresses[-1] == '1'):
                ends.append(word)
    
        return np.random.choice(ends)


    def end_next(self, prev_end):
        """
        Find the next end word given previous, finding a similar word that
        ends in stressed.
        """
        try:
            w, p = zip(*self.word2vec.most_similar(prev_end, topn=10))
        except KeyError:
            return np.random.choice(self.inverted_rhyme.keys())
        
        w = list(w)
        # Make sure it starts out with unstressed
        ends = []
        for word in w:
            if word == prev_end: continue

            try:
                stresses = self.inverted_meter[word][0].split(',')
            except:
                continue

            if word not in self.inverted_rhyme:
                continue

            if (stresses[-1] == '1'):
                ends.append(word)
    
        return np.random.choice(ends)


    def end_next_rhyme(self, prev_rhyme):
        """
        Find the next end word given previous, and a word that must rhyme 
        with it.
        """
        ending = self.inverted_rhyme[prev_rhyme][0]
        
        rhymes = self.rhyme[ending]

        threshold_similarity = 0.1
        best_words = []
        for rhyme in rhymes:
            if rhyme == prev_rhyme:
                continue
            stresses = self.inverted_meter[rhyme][0].split(',')
            if stresses[-1] == '0':
                continue

            try:
                sim = self.word2vec.similarity(prev_rhyme, rhyme)
                if sim > threshold_similarity:
                    best_words.append(rhyme)
            
            except KeyError:
                # probably a stopword
                best_words.append(rhyme)

        if len(best_words) == 0:
            return np.random.choice(rhymes)

        return np.random.choice(best_words)


    def generate_sonnet(self, end_word=None):
        """
        Generate a 14 line sonnet.
        """
        if not end_word:
            ends = []
            for k in self.meter.keys():
                m = map(int, k.split(','))

                if m[-1] == 1:
                    ends.extend([w for w in self.meter[k] and \
                                     w in self.inverted_rhyme])
            end_word = np.random.choice(ends)

        sonnet = ''
        for i in xrange(14):
            line = self.generate_line(end_word)
            end_word = line[-1]
            
            sonnet += ' '.join(line)
            if ((i + 1) % 4 == 0) or (i == 13):
                sonnet += '.\n'
            else:
                sonnet += ',\n'
            
            end_word = self.end_next(end_word)
        return sonnet


    def generate_sonnet_rhyme(self, end_word=None):
        """
        Generate a 14 line sonnet.
        """
        # Encodes abab cdcd efef gg rhyme scheme 
        rhyme_scheme = {2:0, 3:1, 6:4, 7:5, 10:8, 11:9, 13:12}
        
        if not end_word:
            ends = []
            for k in self.meter.keys():
                m = map(int, k.split(','))
                if m[-1] == 1:
                    ends.extend([w for w in self.meter[k] and \
                                     w in self.inverted_rhyme])
            end_word = np.random.choice(ends)

        sonnet = ''

        # Generate all the end words initially
        end_words = [''] * 14
        end_words[0] = end_word
        for i in xrange(1, 14):
            if i in rhyme_scheme:
                end_words[i] = self.end_next_rhyme( \
                    end_words[rhyme_scheme[i]])
            elif i == 8:
                end_words[i] = self.end_next_volta(end_words[0])
            elif i % 4 == 0:
                end_words[i] = self.end_next(end_words[0])
            else:
                end_words[i] = self.end_next(end_words[i - 1])
                
                
        for i in xrange(14):
            line = self.generate_line(end_words[i])
            
            sonnet += ' '.join(line)
            if ((i + 1) % 4 == 0) or (i == 13):
                sonnet += '.\n'
            else:
                sonnet += ',\n'
            
        return sonnet


def supervised_learning(X, Y, hidden, obs):
    '''
    Trains the HMM using the Maximum Likelihood closed form solutions
    for the transition and observation matrices on a labeled
    datset (X, Y). Note that this method does not return anything, but
    instead updates the attributes of the HMM object.
    '''
    # Calculate each element of A using the M-step formulas.

    A = np.zeros((hidden, hidden))
    for i in range(len(Y)):
        for j in range(1, len(Y[i])):
            A[Y[i][j-1], Y[i][j]] += 1
            
    for i in range(hidden):
        A[i, :] = np.divide(A[i, :], np.sum(A[i, :]))
                
    # Calculate each element of O using the M-step formulas.

    O = np.zeros((hidden, obs))
    # Y and X are the same size
    for i in range(len(Y)):
        for j in range(len(Y[i])):
            O[Y[i][j], X[i][j]] += 1
        
    for i in range(hidden):
        O[i, :] = np.divide(O[i, :], np.sum(O[i, :]))

    A_start = np.zeros((1, hidden))        
    for i in range(len(Y)):
        A_start[0, Y[i][0]] += 1

    A_start = np.divide(A_start, np.sum(A_start))

    return A, O, A_start


if __name__ == '__main__':
    model = joblib.load('../models/backwards_hmm_10.pkl')
    
    A = model.transmat_
    O = model.emissionprob_
    A_start = model.startprob_

    hmm = BackwardsSonnetHMM(A, O, A_start)
    
    while True:
        try:
            seed = np.random.choice(hmm.inverted_rhyme.keys())
            sonnet = hmm.generate_sonnet_rhyme(seed)
            
            print sonnet
            break
        except:
            continue

    #lines = sonnet.split("\n")
    #for line in lines:
    #    line = line[:-1]
    #    for word in line.split(' '):
    #        print word, hmm.inverted_pos[word][0]
    #    print
