import numpy as np
import json
import re

from graph import create_graph

from sonnetHMM import BackwardsSonnetHMM


#visualization tools
if __name__ == '__main__':
	A = np.loadtxt("../models/pos_A.txt")
	O = np.loadtxt("../models/pos_O.txt")
	A_start = np.loadtxt("../models/pos_A_start.txt")

	A_conc = np.vstack((A_start,A))

	vocab = json.load( \
	    open('../models/words/vocab.json'))
	# Change to integer keys
	for k in vocab.keys():
	    vocab[int(k)] = vocab.pop(k) 

	inverted_vocab = json.load( \
	    open('../models/words/inverted_vocab.json'))

	meter = json.load( \
	    open('../models/words/meter.json'))
	inverted_meter = json.load( \
	    open('../models/words/inverted_meter.json'))

	rhyme = json.load( \
	    open('../models/words/rhyme.json'))
	inverted_rhyme = json.load( \
	    open('../models/words/inverted_rhyme.json'))

	pos = json.load( \
	    open('../models/words/pos.json'))
	inverted_pos = json.load( \
	    open('../models/words/inverted_pos.json'))

	for i in range(len(O)):
	    top_n = O[i].argsort()[-10:][::-1]
	    print "\nHidden state " + str(i)
	    for j in top_n:
	        print vocab[j] + ",",
	    print ""
	    for j in top_n:
	        print inverted_pos[vocab[j]][0] + ",",
	    print " "
	    for j in top_n:
	        print inverted_meter[vocab[j]][0] + ",",
	    print
	create_graph(A_conc.T).render("36_state_backward")