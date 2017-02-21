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


def parse_rhyme(word):
    """
    Parses each word in a line for rhyme.
    """
    k = ''
    try:
        pronounciation = d_pronoun[word][-1]
        k = ','.join(pronounciation[-2:])
        
    except (KeyError):
        # Can't do anything if word is not in dictionary
        pass

    return word, k
    

def parse_line(line):
    """
    Parses each word in a line of a sonnet for rhyme and meter. (Assumes
    iambic pentameter).
    """
    def syl(pronunciation):
        """
        Helper function to find number of syllables given cmudict pronounciation
        """
        stress = [i[-1] for i in pronunciation if \
                      i[-1].isdigit()]
        
        # Get rid of secondary stress, need only stressed and unstressed
        for i in xrange(len(stress)):
            if stress[i] == '2':
                stress[i] = '1'
        return stress

    tot = 0
    for word in line:
        sk = ''
        ml = ''
        
        try:
            pronounciation = d_pronoun[word][-1]
            stress = syl(pronounciation)

            s = len(stress)
            mk = ','.join(stress)
            
        except (KeyError):
            # Manually count syllables if not in dictionary
            s = syl_count(word)
        
            stress = []
            for i in xrange(s):
                if (tot + i) % 2 == 0:
                    stress.append(0)
                else:
                    stress.append(1)

            mk = ','.join(str(i) for i in stress)

        sk = parse_rhyme(word)[1]
        
        tot += s                        
        yield word, sk, mk
        

if __name__=='__main__':
    files = ['data/shakespeare.txt', 'data/shakespeare_xtra.txt']
    
    line_tokens = []
    for filename in files:
        line_tokens.extend(split_lines(filename))

    meter = {}
    rhyme = {}
    seen = set()
    
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
                    
            seen.add(word) # Save word

    # Convert all the sets to lists, since sets aren't serializable
    seen = list(seen)
    vocab = dict((i, c) for i, c in enumerate(seen))
    inverted_vocab = dict((c, i) for i, c in enumerate(seen))

    for k, v in rhyme.items():
        rhyme[k] = list(v)

    for k, v in meter.items():
        meter[k] = list(v)

    with open('models/shakespeare_vocab.json', 'w') as f:
        json.dump(vocab, f)
    
    with open('models/shakespeare_inverted_vocab.json', 'w') as f:
        json.dump(inverted_vocab, f)

    with open('models/shakespeare_rhyme.json', 'w') as f:
        json.dump(rhyme, f)
        
    with open('models/shakespeare_meter.json', 'w') as f:
        json.dump(meter, f)
