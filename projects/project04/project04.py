
import os
import pandas as pd
import numpy as np
import requests
import time
import re

# ---------------------------------------------------------------------
# Question #1
# ---------------------------------------------------------------------

def get_book(url):
    """
    get_book that takes in the url of a 'Plain Text UTF-8' book and 
    returns a string containing the contents of the book.

    The function should satisfy the following conditions:
        - The contents of the book consist of everything between 
        Project Gutenberg's START and END comments.
        - The contents will include title/author/table of contents.
        - You should also transform any Windows new-lines (\r\n) with 
        standard new-lines (\n).
        - If the function is called twice in succession, it should not 
        violate the robots.txt policy.

    :Example: (note '\n' don't need to be escaped in notebooks!)
    >>> url = 'http://www.gutenberg.org/files/57988/57988-0.txt'
    >>> book_string = get_book(url)
    >>> book_string[:20] == '\\n\\n\\n\\n\\nProduced by Chu'
    True
    """
    text = requests.get(url).text
    time.sleep(5) # Delay for 5 seconds

    # Transform \r\n with \n newline
    subtext = re.sub(r'\r\n', '\n', text) 

    # Extract content from START to END
    pat = '\*\*\* START OF THIS PROJECT GUTENBERG EBOOK .* \*\*\*([\S\s]*)\*\*\* END OF THIS PROJECT GUTENBERG EBOOK'
    content = re.search(pat, subtext).group(1)
    return content
    
# ---------------------------------------------------------------------
# Question #2
# ---------------------------------------------------------------------


def tokenize(book_string):
    """
    tokenize takes in book_string and outputs a list of tokens 
    satisfying the following conditions:
        - The start of any paragraph should be represented in the 
        list with the single character \x02 (standing for START).
        - The end of any paragraph should be represented in the list 
        with the single character \x03 (standing for STOP).
        - Tokens in the sequence of words are split 
        apart at 'word boundaries' (see the regex lecture).
        - Tokens should include no whitespace.

    :Example:
    >>> test_fp = os.path.join('data', 'test.txt')
    >>> test = open(test_fp, encoding='utf-8').read()
    >>> tokens = tokenize(test)
    >>> tokens[0] == '\x02'
    True
    >>> tokens[9] == 'dead'
    True
    >>> sum([x == '\x03' for x in tokens]) == 4
    True
    >>> '(' in tokens
    True
    """
    begin = re.sub(r'^(\n *)*', '\x02', book_string) # Begin of file
    final = re.sub(r'(\n *)+$', '\x03', begin) # End of file
    middle = re.sub(r'(\n *){2,}', '\x03\x02 ', final) # Middle start/stop of parag
    no_line = re.sub(r'\n', ' ', middle) # Other new lines with space

    slist = re.split(pattern=r'(?=\x02|\x03)|(?<=\x02|\x03)|\b', string=no_line)
    slist = list(map(str.strip, filter(str.strip, slist))) # Filter empty, strip space
    return slist

    
# ---------------------------------------------------------------------
# Question #3
# ---------------------------------------------------------------------


class UniformLM(object):
    """
    Uniform Language Model class.
    """

    def __init__(self, tokens):
        """
        Initializes a Uniform languange model using a
        list of tokens. It trains the language model
        using `train` and saves it to an attribute
        self.mdl.
        """
        self.mdl = self.train(tokens)
        
    def train(self, tokens):
        """
        Trains a uniform language model given a list of tokens.
        The output is a series indexed on distinct tokens, and
        values giving the (uniform) probability of a token occuring
        in the language.

        :Example:
        >>> tokens = tuple('one one two three one two four'.split())
        >>> unif = UniformLM(tokens)
        >>> isinstance(unif.mdl, pd.Series)
        True
        >>> set(unif.mdl.index) == set('one two three four'.split())
        True
        >>> (unif.mdl == 0.25).all()
        True
        """
        if len(tokens) == 0: # Empty tokens
            return pd.Series(dtype=float).astype(float)
        
        index = pd.Series(tokens).unique() # Word index
        return pd.Series([1 / len(index)] * len(index), index=index)
    
    def probability(self, words):
        """
        probability gives the probabiliy a sequence of words
        appears under the language model.
        :param: words: a tuple of tokens
        :returns: the probability `words` appears under the language
        model.

        :Example:
        >>> tokens = tuple('one one two three one two four'.split())
        >>> unif = UniformLM(tokens)
        >>> unif.probability(('five',))
        0
        >>> unif.probability(('one', 'two')) == 0.0625
        True
        """
        prob = 1
        for w in words: # Loop through words
            if w not in self.mdl.index: # Not in tokens
                return 0
            else: # Get prob
                prob *= self.mdl.loc[w] 
        return prob
        
    def sample(self, M):
        """
        sample selects tokens from the language model of length M, returning
        a string of tokens.

        :Example:
        >>> tokens = tuple('one one two three one two four'.split())
        >>> unif = UniformLM(tokens)
        >>> samp = unif.sample(1000)
        >>> isinstance(samp, str)
        True
        >>> len(samp.split()) == 1000
        True
        >>> s = pd.Series(samp.split()).value_counts(normalize=True)
        >>> np.isclose(s, 0.25, atol=0.05).all()
        True
        """
        ran = self.mdl.sample(M, replace=True)
        return ' '.join(ran.index)


# ---------------------------------------------------------------------
# Question #4
# ---------------------------------------------------------------------


class UnigramLM(object):
    
    def __init__(self, tokens):
        """
        Initializes a Unigram languange model using a
        list of tokens. It trains the language model
        using `train` and saves it to an attribute
        self.mdl.
        """
        self.mdl = self.train(tokens)
    
    def train(self, tokens):
        """
        Trains a unigram language model given a list of tokens.
        The output is a series indexed on distinct tokens, and
        values giving the probability of a token occuring
        in the language.

        :Example:
        >>> tokens = tuple('one one two three one two four'.split())
        >>> unig = UnigramLM(tokens)
        >>> isinstance(unig.mdl, pd.Series)
        True
        >>> set(unig.mdl.index) == set('one two three four'.split())
        True
        >>> unig.mdl.loc['one'] == 3 / 7
        True
        """
        if len(tokens) == 0: # Empty tokens
            return pd.Series(dtype=float).astype(float)
        return pd.Series(tokens).value_counts(normalize=True)
    
    def probability(self, words):
        """
        probability gives the probabiliy a sequence of words
        appears under the language model.
        :param: words: a tuple of tokens
        :returns: the probability `words` appears under the language
        model.

        :Example:
        >>> tokens = tuple('one one two three one two four'.split())
        >>> unig = UnigramLM(tokens)
        >>> unig.probability(('five',))
        0
        >>> p = unig.probability(('one', 'two'))
        >>> np.isclose(p, 0.12244897959, atol=0.0001)
        True
        """
        prob = 1
        for w in words: # Loop through words
            if w not in self.mdl.index: # Not in tokens
                return 0
            else: # Get prob
                prob *= self.mdl.loc[w] 
        return prob
        
    def sample(self, M):
        """
        sample selects tokens from the language model of length M, returning
        a string of tokens.

        >>> tokens = tuple('one one two three one two four'.split())
        >>> unig = UnigramLM(tokens)
        >>> samp = unig.sample(1000)
        >>> isinstance(samp, str)
        True
        >>> len(samp.split()) == 1000
        True
        >>> s = pd.Series(samp.split()).value_counts(normalize=True).loc['one']
        >>> np.isclose(s, 0.41, atol=0.05).all()
        True
        """
        ran = np.random.choice(self.mdl.index, p=self.mdl.values, size=M)
        return ' '.join(ran)
        
    
# ---------------------------------------------------------------------
# Question #5,6,7,8
# ---------------------------------------------------------------------

class NGramLM(object):
    
    def __init__(self, N, tokens):
        """
        Initializes a N-gram languange model using a
        list of tokens. It trains the language model
        using `train` and saves it to an attribute
        self.mdl.
        """

        self.N = N
        ngrams = self.create_ngrams(tokens)

        self.ngrams = ngrams
        self.mdl = self.train(ngrams)

        if N < 2:
            raise Exception('N must be greater than 1')
        elif N == 2:
            self.prev_mdl = UnigramLM(tokens)
        else:
            mdl = NGramLM(N-1, tokens)
            self.prev_mdl = mdl

    def create_ngrams(self, tokens):
        """
        create_ngrams takes in a list of tokens and returns a list of N-grams. 
        The START/STOP tokens in the N-grams should be handled as 
        explained in the notebook.

        :Example:
        >>> tokens = tuple('\x02 one two three one four \x03'.split())
        >>> bigrams = NGramLM(2, [])
        >>> out = bigrams.create_ngrams(tokens)
        >>> isinstance(out[0], tuple)
        True
        >>> out[0]
        ('\\x02', 'one')
        >>> out[2]
        ('two', 'three')
        """
        ngrams = []
        for i in range(len(tokens)- self.N + 1):
            ngrams.append(tuple(tokens[i:i+self.N]))
        return ngrams
        
    def train(self, ngrams):
        """
        Trains a n-gram language model given a list of tokens.
        The output is a dataframe indexed on distinct tokens, with three
        columns (ngram, n1gram, prob).

        :Example:
        >>> tokens = tuple('\x02 one two three one four \x03'.split())
        >>> bigrams = NGramLM(2, tokens)
        >>> set(bigrams.mdl.columns) == set('ngram n1gram prob'.split())
        True
        >>> bigrams.mdl.shape == (6, 3)
        True
        >>> bigrams.mdl['prob'].min() == 0.5
        True
        """
        unique_ngrams = pd.Series(ngrams).unique()
        # ngram counts C(w_1, ..., w_n)
        n1grams, c_ngram, c_n1gram = [], [], []
        for ngram in ngrams:
            n1grams.append(ngram[:-1]) # Construct n1gram
            c_ngram.append(ngrams.count(ngram)) # ngram occurrence
        
        # n-1 gram counts C(w_1, ..., w_(n-1))
        for n1gram in n1grams:
            c_n1gram.append(n1grams.count(n1gram))

        # Create the conditional probabilities
        probs = np.array(c_ngram) / np.array(c_n1gram)
        
        # Put it all together
        ngram_col = pd.Series(ngrams, name='ngram')
        n1gram_col = pd.Series(n1grams, name='n1gram')
        prob_col = pd.Series(probs, name='prob')

        # print(c_ngram, c_n1gram)
        df = pd.DataFrame([ngram_col, n1gram_col, prob_col]).T
        no_dup = df.drop_duplicates('ngram').reset_index(drop=True)
        return no_dup
    
    def probability(self, words):
        """
        probability gives the probabiliy a sequence of words
        appears under the language model.
        :param: words: a tuple of tokens
        :returns: the probability `words` appears under the language
        model.

        :Example:
        >>> tokens = tuple('\x02 one two one three one two \x03'.split())
        >>> bigrams = NGramLM(2, tokens)
        >>> p = bigrams.probability('two one three'.split())
        >>> np.isclose(p, (1/4)*(1/2)*(1/3))
        True
        >>> bigrams.probability('one two five'.split()) == 0
        True
        """
        if len(words) == 0:
            return 0
        
        prob = 1
        model = self.mdl
        
        words_ngram = NGramLM(self.N, []).create_ngrams(words) # Create NGram model for words
        for ngram in words_ngram:
            # Never seen before ngram or n-1gram
            if (ngram not in list(model['ngram'])) or (ngram[:-1] not in list(model['n1gram'])):
                return 0
            if isinstance(self, NGramLM):
                prob *= model[model['ngram'] == ngram]['prob'].values[0]
        
        def recur_prob(model, w):
            prob = 1
            prev_mod = model.prev_mdl
            if isinstance(prev_mod, UnigramLM): # Unigram base case
                prob *= prev_mod.mdl[w[0]]
            else:
                words_n1gram = NGramLM(prev_mod.N, []).create_ngrams(w) # Create NGram model for words
                prob *= prev_mod.mdl[prev_mod.mdl['ngram'] == words_n1gram[0]]['prob'].values[0]
                prob *= recur_prob(prev_mod, words_n1gram[0]) # Recursive call
            return prob

        prob *= recur_prob(self, words_ngram[0])
        
        return prob

    def sample(self, M):
        """
        sample selects tokens from the language model of length M, returning
        a string of tokens.

        :Example:
        >>> tokens = tuple('\x02 one two three one four \x03'.split())
        >>> bigrams = NGramLM(2, tokens)
        >>> samp = bigrams.sample(3)
        >>> len(samp.split()) == 4  # don't count the initial START token.
        True
        >>> samp[:2] == '\\x02 '
        True
        >>> set(samp.split()) <= {'\\x02', '\\x03', 'one', 'two', 'three', 'four'}
        True
        """
        # Helper function to get mdls
        def recur_mdl(model, lst):
            if isinstance(model, UnigramLM): # Base case
                return
    
            recur_mdl(model.prev_mdl, lst)
            lst.append(model)
            return lst
        
        tokens = ['\x02'] # START token

        # Use a helper function to generate sample tokens of length `length`
        mdls = recur_mdl(self, []) # List of models

        if M <= self.N: # Before model ngrams
            mdls = mdls[:M]
        else: # If reach model ngrams
            for _ in range(M - self.N + 1): # Append additional used models
                mdls.append(mdls[-1])

        tups = tuple('\x02'.split()) # First word depend on '\x02'
        for mdl in mdls: # Loop through used models
            probs = mdl.mdl[mdl.mdl['n1gram'] == tups] # Get ngrams and probability dataframe
            if len(probs.ngram) == 0: # No word to choose
                ran = '\x03' # Append '\x03'
                break
            else:
                random = np.random.choice(probs.ngram, p=probs.prob) # Choose token based on probs
                ran = random[-1]
                
                if mdl.N < self.N: # If still smaller than N
                    tups = random
                else: # ngram models
                    tups = random[1:]

            tokens.append(ran) # Append
        
        for _ in range(M - len(tokens)): # Fill the gap of missing due to '\x03'
            tokens.append('\x03')
        
        # Transform the tokens to strings
        return ' '.join(tokens)

# ---------------------------------------------------------------------
# DO NOT TOUCH BELOW THIS LINE
# IT'S FOR YOUR OWN BENEFIT!
# ---------------------------------------------------------------------


# Graded functions names! DO NOT CHANGE!
# This dictionary provides your doctests with
# a check that all of the questions being graded
# exist in your code!

GRADED_FUNCTIONS = {
    'q01': ['get_book'],
    'q02': ['tokenize'],
    'q03': ['UniformLM'],
    'q04': ['UnigramLM'],
    'q05': ['NGramLM']
}


def check_for_graded_elements():
    """
    >>> check_for_graded_elements()
    True
    """
    
    for q, elts in GRADED_FUNCTIONS.items():
        for elt in elts:
            if elt not in globals():
                stmt = "YOU CHANGED A QUESTION THAT SHOULDN'T CHANGE! \
                In %s, part %s is missing" %(q, elt)
                raise Exception(stmt)

    return True
