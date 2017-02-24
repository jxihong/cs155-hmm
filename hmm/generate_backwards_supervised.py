import numpy as np
import json
import re

from sonnetHMM import BackwardsSonnetHMM

if __name__ == '__main__':
    A = np.loadtxt("../models/pos_A.txt")
    O = np.loadtxt("../models/pos_O.txt")
    A_start = np.loadtxt("../models/pos_A_start.txt")

    hmm = BackwardsSonnetHMM(A, O, A_start)

    while True:
        try:
            seed = np.random.choice(hmm.inverted_rhyme.keys())
            sonnet = hmm.generate_sonnet_rhyme(seed)

            print sonnet
            #lines = sonnet.split("\n")
            #for line in lines:
            #    line = line[:-1]
            #    for word in line.split(' '):
            #        try:
            #            print word, hmm.inverted_pos[word][0]
            #        except:
            #            continue
            break
        except:
            continue
