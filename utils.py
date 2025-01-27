import string
import random
import re

def syl_count(w):
    """
    Roughly counts the number of syllables in a word (use when 
    NLTK cannot find word in its vocabulary.
    """
    word = w.lower()
    word = word.translate(None, string.punctuation)
    
    syls = 0 #added syllable number
    disc = 0 #discarded syllable number
 
    if len(word) <= 3 :
        syls = 1
        return syls
 
    # Remove trailing e's
    if word[-1:] == "e" :
        if word[-2:] == "le":
            pass
        else:
            disc+=1
     
    if word[-2:] == "ed" or word[-2:] == "es":
        if word[-3:] == "ted" or word[-3:] == "tes" or word[-3:] == "ses" \
            or word[-3:] == "ied" or word[-3:] == "ies" :
                pass
        else:
            disc+=1
    
    # Count consecutive vowels as one
    numVowels = len(re.findall(r'[aeoui]+', word))
    
    # Consider a few exceptions I found from perusing data
    if word[-1:] == "y" and word[-2] not in "aeoui" :
        syls +=1
        
    for i,j in enumerate(word) :
        if j == "y" :
            if (i != 0) and (i != len(word)-1) :
                if word[i-1] not in "aeoui" and word[i+1] not in "aeoui" :
                    syls+=1

    if word[:3] == "tri" and word[3] in "aeoui" :
        syls+=1
 
    if word[:2] == "bi" and word[2] in "aeoui" :
        syls+=1
 
    if word[-3:] == "ian": 
        if word[-4:] == "cian" or word[-4:] == "tian" :
            pass
        else:
            syls+=1
    
    if word[:5] == "where":
        disc += 1
        
    return max(numVowels - disc + syls, 1)


def invert_map(d):
    """
    Invert a dictionary of lists, where lists contain non-unique items.
    """
    inverse = {}
    for k in d:
        for v in d[k]:
            inverse.setdefault(v, []).append(k)
    return inverse


def random_pick(l, probs):
    """ 
    Probabilistic random picking according
    to a probability distribution
    """
    x = random.uniform(0, 1)
    cumulative_probability = 0.0

    for item, prob in zip(l, probs):
        cumulative_probability += prob
        if x < cumulative_probability: 
            break
    return item
