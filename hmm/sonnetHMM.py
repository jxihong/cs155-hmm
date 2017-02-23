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
            open('../models/shakespeare_words/shakespeare_vocab.json'))
        # Change to integer keys
        for k in self.vocab.keys():
            self.vocab[int(k)] = self.vocab.pop(k) 
        
        self.inverted_vocab = json.load( \
            open('../models/shakespeare_words/shakespeare_inverted_vocab.json'))
        
        self.meter = json.load( \
            open('../models/shakespeare_words/shakespeare_meter.json'))
        self.inverted_meter = json.load( \
            open('../models/shakespeare_words/shakespeare_inverted_meter.json'))
        
        self.rhyme = json.load( \
            open('../models/shakespeare_words/shakespeare_rhyme.json'))
        self.inverted_rhyme = json.load( \
            open('../models/shakespeare_words/shakespeare_inverted_rhyme.json'))

        self.pos = json.load( \
            open('../models/shakespeare_words/shakespeare_pos.json'))
        self.inverted_pos = json.load( \
            open('../models/shakespeare_words/shakespeare_inverted_pos.json'))
        
        self.word2vec = gensim.models.Word2Vec.load( \
            '../models/word2vec.bin')

    
    def filter_next(self, num_syllables, prev_word, probs):
        """
        Filters possible words to preserve meter and syllable count of the line,
        and normalizes the new probabilities.
        """
        new_probs = np.copy(probs)
        
        # Filter based on meter, and keep syllables 10
        invalid = []
        for k in self.meter.keys():
            m = map(int, k.split(','))
            if m[-1] != ((num_syllables + 1) % 2):
                invalid.extend([self.inverted_vocab[w] for w in self.meter[k]])
        
            if len(m) + num_syllables > 10:
                invalid.extend([self.inverted_vocab[w] for w in self.meter[k]])

        # grammar rules
        for k in self.pos.keys():
            # do a few cases to implement some basic grammar rules. note this is 
            # training backwards so the rules are a little weird.
            # need to multiply parts of speech by probability transition matrix?
            # requires tagging a POS with cmu nltk pos_tag
            if "NN" in self.inverted_pos[prev_word]:
                for k in self.pos.keys():
                    if k in ["NN",]:
                        invalid.extend([self.inverted_vocab[w] for w in self.pos[k]])

        
        new_probs[invalid] = 0
        with np.errstate(divide='ignore'):
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

        
    def end_next(self, prev_end):
        """
        Find the next end word given previous, finding a similar word that
        ends in stressed.
        """
        w, p = zip(*self.word2vec.most_similar(prev_end, topn=30))
        
        w = list(w)
        # Make sure it starts out with unstressed
        ends = []
        for word in w:
            stresses = self.inverted_meter[word][0].split(',')
            
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

        max_similarity = 0.
        best_word = prev_rhyme
        for rhyme in rhymes:
            if rhyme == prev_rhyme:
                continue
            stresses = self.inverted_meter[rhyme][0].split(',')
            if stresses[-1] == '0':
                continue
            
            try:
                sim = self.word2vec.similarity(prev_rhyme, rhyme)
                if sim > max_similarity:
                    best_word = rhyme
                    max_similarity = sim
            except KeyError:
                if best_word == prev_rhyme:
                    best_word = rhyme

        return best_word


    def generate_sonnet(self, end_word=None):
        """
        Generate a 14 line sonnet.
        """
        if not end_word:
            ends = []
            for k in self.meter.keys():
                m = map(int, k.split(','))
                if m[-1] == 1:
                    ends.extend([w for w in self.meter[k]])
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
                    ends.extend([w for w in self.meter[k]])
            end_word = np.random.choice(ends)

        sonnet = ''
        end_words = []
        for i in xrange(14):
            line = self.generate_line(end_word)
            end_word = line[-1]
            end_words.append(end_word) # Add to list of end words

            sonnet += ' '.join(line)
            if ((i + 1) % 4 == 0) or (i == 13):
                sonnet += '.\n'
            else:
                sonnet += ',\n'
            
            if (i + 1) in rhyme_scheme:
                end_word = self.end_next_rhyme( \
                    end_words[rhyme_scheme[i + 1]])
            else:
                end_word = self.end_next(end_word)

        return sonnet


if __name__ == '__main__':
    model = joblib.load('../models/backwards_hmm_50.pkl')
    
    A = model.transmat_
    O = model.emissionprob_
    A_start = model.startprob_

    hmm = BackwardsSonnetHMM(A, O, A_start)
    
    print hmm.generate_sonnet("love")
    print hmm.generate_sonnet_rhyme("love")
