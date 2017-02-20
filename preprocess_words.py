import nltk
import string
import json

from nltk.tokenize import RegexpTokenizer
from nltk.corpus import cmudict

from utils import syl_count

d_pronoun = cmudict.dict()

def split_sonnets(filename):
    """
    Tokenizes the file and returns a list of tokens for
    each sonnet in the file. (Doesn't look like it's needed 
    since splitting by lines is easier and works as well)
    """
    # Keep apostrophes and hyphens in a word 
    tokenizer = RegexpTokenizer('[\w|\'|-]+') 

    sonnets = []
    with open(filename) as f:
        sonnet = []
        sonnetBegin = False
        
        for line in f:
            line = line.strip()
            if (line.isdigit()):
                sonnetBegin = True
                continue
            if (len(line) > 0):
                line = line.lower()
                tokens = tokenizer.tokenize(line)
                
                sonnet.extend(tokens)
            if len(line) == 0:
                if sonnetBegin:
                    sonnets.append(sonnet)
                    sonnet = []
                    sonnetBegin = False

    return sonnets


def split_lines(filename):
    """
    Tokenizes the file and returns a list of tokens for
    each line of poetry in the file.
    """
    # Keep apostrophes and hyphens
    tokenizer = RegexpTokenizer('[\w|\'|-]+') 

    line_tokens = []
    with open(filename) as f:
        for line in f:
            line = line.strip()
            if (line.isdigit()):
                continue
            if (len(line) > 0):
                line = line.lower()
                tokens = tokenizer.tokenize(line)
                
                line_tokens.append(tokens)

    return line_tokens


def parse_line(line):
    """
    Parses each word in a line of a sonnet for rhyme and meter
    """
    def syl(pronunciation):
        """
        Helper function to find number of syllables given cmudict pronounciation
        """
        return len([i[-1] for i in pronunciation if \
                i[-1].isdigit()])
    
    tot = 0
    for word in line:
        sk = ''
        ml = ''
        
        try:
            pronounciation = d_pronoun[word][0]
            s = syl(pronounciation)
            
            sk = ','.join(pronounciation[-2:])
        except (KeyError):
            s = syl_count(word)
        
        stress = []
        for i in xrange(s):
            if (tot + i) % 2 == 0:
                stress.append(0)
            else:
                stress.append(1)
        
        mk = ','.join(str(i) for i in stress)
        
        tot += s                
        
        yield word, sk, mk
        

if __name__=='__main__':
    filename = 'data/shakespeare.txt'
    
    line_tokens = split_lines(filename)
    
    meter = {}
    rhyme = {}
    vocab = set()
    
    for line in line_tokens:
        for word, sk, mk in parse_line(line):
            # Save meter of word
            if len(sk) > 0:
                if sk in rhyme.keys():
                    rhyme[sk].add(word)
                else:
                    rhyme[sk] = set()
                    rhyme[sk].add(word)

            # Save rhyme of word
            if len(mk) > 0:
                if mk in meter.keys():
                    meter[mk].add(word)
                else:
                    meter[mk] = set()
                    meter[mk].add(word)
                    
            vocab.add(word) # Save word

    # Convert all the sets to lists, since sets aren't serializable
    vocab = list(vocab)

    for k, v in rhyme.items():
        rhyme[k] = list(v)

    for k, v in meter.items():
        meter[k] = list(v)

    with open('models/shakespeare_vocab.json', 'w') as f:
        json.dump(vocab, f)
        
    with open('models/shakespeare_rhyme.json', 'w') as f:
        json.dump(rhyme, f)
        
    with open('models/shakespeare_meter.json', 'w') as f:
        json.dump(meter, f)
