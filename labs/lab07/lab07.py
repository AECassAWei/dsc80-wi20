import os
import pandas as pd
import numpy as np
import requests
import json
import re



# ---------------------------------------------------------------------
# Question # 1
# ---------------------------------------------------------------------


####################
#  Regex
####################



# ---------------------------------------------------------------------
# Problem 1
# ---------------------------------------------------------------------

def match_1(string):
    """
    A string that has a [ as the third character and ] as the sixth character.
    >>> match_1("abcde]")
    False
    >>> match_1("ab[cde")
    False
    >>> match_1("a[cd]")
    False
    >>> match_1("ab[cd]")
    True
    >>> match_1("1ab[cd]")
    False
    >>> match_1("ab[cd]ef")
    True
    >>> match_1("1b[#d] _")
    True
    >>> match_1(" `[  ] _")
    True
    """
    #Your Code Here
    pattern = '^.{2}\[.{2}\]'

    #Do not edit following code
    prog = re.compile(pattern)
    return prog.search(string) is not None


def match_2(string):
    """
    Phone numbers that start with '(858)' and
    follow the format '(xxx) xxx-xxxx' (x represents a digit)
    Notice: There is a space between (xxx) and xxx-xxxx

    >>> match_2("(123) 456-7890")
    False
    >>> match_2("858-456-7890")
    False
    >>> match_2("(858)45-7890")
    False
    >>> match_2("(858) 456-7890")
    True
    >>> match_2("(858)456-789")
    False
    >>> match_2("(858)456-7890")
    False
    >>> match_2("a(858) 456-7890")
    False
    >>> match_2("(858) 456-7890b")
    False
    """
    #Your Code Here
    pattern = '^\(858\) \d{3}-\d{4}$'

    #Do not edit following code
    prog = re.compile(pattern)
    return prog.search(string) is not None


def match_3(string):
    """
    Find a pattern whose length is between 6 to 10
    and contains only word character, white space and ?.
    This string must have ? as its last character.

    >>> match_3("qwertsd?")
    True
    >>> match_3("qw?ertsd?")
    True
    >>> match_3("ab c?")
    False
    >>> match_3("ab   c ?")
    True
    >>> match_3(" asdfqwes ?")
    False
    >>> match_3(" adfqwes ?")
    True
    >>> match_3(" adf!qes ?")
    False
    >>> match_3(" adf!qe? ")
    False
    """
    #Your Code Here
    pattern = '^[\w \\?]{5,9}\?$'

    #Do not edit following code
    prog = re.compile(pattern)
    return prog.search(string) is not None


def match_4(string):
    """
    A string that begins with '$' and with another '$' within, where:
        - Characters between the two '$' can be anything except the 
        letters 'a', 'b', 'c' (lower case).
        - Characters after the second '$' can only have any number 
        of the letters 'a', 'b', 'c' (upper or lower case), with every 
        'a' before every 'b', and every 'b' before every 'c'.
            - E.g. 'AaBbbC' works, 'ACB' doesn't.

    >>> match_4("$$AaaaaBbbbc")
    True
    >>> match_4("$!@#$aABc")
    True
    >>> match_4("$a$aABc")
    False

    >>> match_4("$iiuABc")
    False
    >>> match_4("123$Abc")
    False
    >>> match_4("$$Abc")
    True
    >>> match_4("$qw345t$AAAc")
    False
    >>> match_4("$s$Bca")
    False
    >>> match_4("$s12B5rf';.t A$aABbbBBcCcCc")
    True
    """
    #Your Code Here
    pattern = '^\$[^abc]*\$[Aa]+[Bb]+[Cc]+$'

    #Do not edit following code
    prog = re.compile(pattern)
    return prog.search(string) is not None


def match_5(string):
    """
    A string that represents a valid Python file name including the extension.
    *Notice*: For simplicity, assume that the file name contains only letters, numbers and an underscore `_`.

    >>> match_5("dsc80.py")
    True
    >>> match_5("dsc80py")
    False
    >>> match_5("dsc80..py")
    False
    >>> match_5("dsc80+.py")
    False
    >>> match_5("Prj78Je_.py")
    True
    >>> match_5(".py")
    False
    """

    #Your Code Here
    pattern = '^[\w]+\.py$'

    #Do not edit following code
    prog = re.compile(pattern)
    return prog.search(string) is not None


def match_6(string):
    """
    Find patterns of lowercase letters joined with an underscore.
    >>> match_6("aab_cbb_bc")
    False
    >>> match_6("aab_cbbbc")
    True
    >>> match_6("aab_Abbbc")
    False
    >>> match_6("abcdef")
    False
    >>> match_6("ABCDEF_ABCD")
    False
    """

    #Your Code Here
    pattern = '^[a-z]+\_[a-z]+$'

    #Do not edit following code
    prog = re.compile(pattern)
    return prog.search(string) is not None


def match_7(string):
    """
    Find patterns that start with and end with a _
    >>> match_7("_abc_")
    True
    >>> match_7("abd")
    False
    >>> match_7("bcd")
    False
    >>> match_7("_ncde")
    False
    >>> match_7("__")
    True
    >>> match_7("_")
    False
    """
    
    #Your Code Here
    pattern = '^\_.*\_$'

    #Do not edit following code
    prog = re.compile(pattern)
    return prog.search(string) is not None


def match_8(string):
    """
    Apple registration numbers and Apple hardware product serial numbers
    might have the number "0" (zero), but never the letter "O".
    Serial numbers don't have the number "1" (one) or the letter "i".

    Write a line of regex expression that checks
    if the given Serial number belongs to a genuine Apple product.

    >>> match_8("ASJDKLFK10ASDO")
    False
    >>> match_8("ASJDKLFK0ASDo")
    True
    >>> match_8("JKLSDNM01IDKSL")
    False
    >>> match_8("ASDKJLdsi0SKLl")
    False
    >>> match_8("ASDJKL9380JKAL")
    True
    >>> match_8("ASDJKL9380JKALIIIo000")
    True
    """

    #Your Code Here
    pattern = '^[^O1i]*$'

    #Do not edit following code
    prog = re.compile(pattern)
    return prog.search(string) is not None


def match_9(string):
    """
    Check if a given ID number is from Los Angeles (LAX), San Diego(SAN) or
    the state of New York (NY). ID numbers have the following format SC-NN-CCC-NNNN.
        - SC represents state code in uppercase
        - NN represents a number with 2 digits
        - CCC represents a three letter city code in uppercase
        - NNNN represents a number with 4 digits
    
    >>> match_9('NY-32-NYC-1232')
    True
    >>> match_9('ca-23-SAN-1231')
    False
    >>> match_9('MA-36-BOS-5465')
    False
    >>> match_9('CA-56-LAX-7895')
    True
    >>> match_9('MA-11-LAX-0101')
    False
    >>> match_9('NY-36-BOS-5465')
    True
    """

    #Your Code Here
    pattern = '(^NY)|(^CA-[0-9]{2}-((LAX)|(SAN))-[0-9]{4}$)'

    #Do not edit following code
    prog = re.compile(pattern)
    return prog.search(string) is not None


def match_10(string):
    """
    Given an input string, cast it to lower case, remove spaces/punctuations, 
    and return a list of every 3-character substring that satisfy the following:
        - The first character doesn't start with 'a' or 'A'
        - The last substring (and only the last substring) can be shorter than 
        3 characters, depending on the length of the input string.
        - The substrings cannot overlap
    
    >>> match_10('ABCdef')
    ['def']
    >>> match_10(' DEFaabc !g ')
    ['def', 'cg']
    >>> match_10('Come ti chiami?')
    ['com', 'eti', 'chi']
    >>> match_10('and')
    []
    >>> match_10( "Ab..DEF")
    ['def']
    >>> match_10( "Ab..AEF")
    []
    >>> match_10( " t^racf_a..AEFcj")
    ['trc', 'j']
    """
    no_space = string.replace(' ', '') # Remove space
    lower_case = no_space.lower() # Cast to lower space
    complete = re.sub(r'a.{2}', '', lower_case) # Remove a 3-char 
    no_punc = re.sub(r'[^A-Za-z0-9 ]', '', complete) # Remove punctuations
    com_sequence = re.findall(r'..?.?', no_punc) # Find complete three-character sequence
    return com_sequence


# ---------------------------------------------------------------------
# Question # 2
# ---------------------------------------------------------------------

def extract_personal(s):
    """
    :Example:
    >>> fp = os.path.join('data', 'messy.test.txt')
    >>> s = open(fp, encoding='utf8').read()
    >>> emails, ssn, bitcoin, addresses = extract_personal(s)
    >>> emails[0] == 'test@test.com'
    True
    >>> ssn[0] == '423-00-9575'
    True
    >>> bitcoin[0] == '1BvBMSEYstWetqTFn5Au4m4GFg7xJaNVN2'
    True
    >>> addresses[0] == '530 High Street'
    True
    """
    # Pattern of each category
    email_pat = r'([A-Za-z0-9]+@[A-Za-z0-9]+(\.[A-Za-z0-9]+)+)'
    ssn_pat = r'[0-9]{3}-[0-9]{2}-[0-9]{4}'
    bit_pat = r'[13][A-Za-z0-9]{26,33}' 
    add_pat = r'(\d+( [A-z]+)+)' 

    # Get email address
    email_group = re.findall(email_pat, s)
    emails = [group[0] for group in email_group]

    # Get social security number
    ssns = re.findall(ssn_pat, s) 
    # ssns = [group[0] for group in ssn_group]

    # Get bitcoin address
    bits = re.findall(bit_pat, s)

    # Get address
    adds_group = re.findall(add_pat, s)
    adds = [group[0] for group in adds_group]
    return (emails, ssns, bits, adds)


# ---------------------------------------------------------------------
# Question # 3
# ---------------------------------------------------------------------

def tfidf_data(review, reviews):
    """
    :Example:
    >>> fp = os.path.join('data', 'reviews.txt')
    >>> reviews = pd.read_csv(fp, header=None, squeeze=True)
    >>> review = open(os.path.join('data', 'review.txt'), encoding='utf8').read().strip()
    >>> out = tfidf_data(review, reviews)
    >>> out['cnt'].sum()
    85
    >>> 'before' in out.index
    True
    """
    # Check review frequency
    words = review.split() # Split review words
    uniquewords = pd.Series(words).unique() # Get unique review words
    freq = dict.fromkeys(uniquewords, 0) # Get frequency
    for word in words:
        freq[word] += 1

    # Check number of documents word appear
    all_uniquewords = reviews.str.split().apply(lambda x: pd.Series(x).unique()) # All words in all reviews
    all_freq = dict.fromkeys(uniquewords, 0) # Get the number of documents word appear
    for word in uniquewords: # Word
        for un in all_uniquewords: # Single review of reviews
            if word in un:
                all_freq[word] += 1 # Appear in one review

    check = pd.DataFrame(columns=['cnt', 'tf', 'idf', 'tfidf'], index=uniquewords)  # tfidf dataframe
    for word in uniquewords:
        re_pat = '\\b%s\\b' % word # Word pattern
        cnt = freq.get(word) # Frequency
        tf = cnt / len(words) # tf
        idf = np.log(len(reviews) / all_freq.get(word)) # idf
        # Put into dataframe
        check.loc[word, 'cnt'], check.loc[word, 'tfidf'] = cnt, tf * idf
        check.loc[word, 'tf'], check.loc[word, 'idf'] = tf, idf

    return check


def relevant_word(out):
    """
    :Example:
    >>> fp = os.path.join('data', 'reviews.txt')
    >>> reviews = pd.read_csv(fp, header=None, squeeze=True)
    >>> review = open(os.path.join('data', 'review.txt'), encoding='utf8').read().strip()
    >>> out = tfidf_data(review, reviews)
    >>> relevant_word(out) in out.index
    True
    """
    return out.sort_values('tfidf', ascending=False).index[0]


# ---------------------------------------------------------------------
# Question # 4
# ---------------------------------------------------------------------

def hashtag_list(tweet_text):
    """
    :Example:
    >>> testdata = [['RT @DSC80: Text-cleaning is cool! #NLP https://t.co/xsfdw88d #NLP1 #NLP1']]
    >>> test = pd.DataFrame(testdata, columns=['text'])
    >>> out = hashtag_list(test['text'])
    >>> (out.iloc[0] == ['NLP', 'NLP1', 'NLP1'])
    True
    """
    hashtag_pat = '#(\w+)' # Hash Tag pattern
    prog = re.compile(hashtag_pat) # Compile
    tags = tweet_text.apply(lambda x: prog.findall(x)) # Find all
    return tags


def most_common_hashtag(tweet_lists):
    """
    :Example:
    >>> testdata = [['RT @DSC80: Text-cleaning is cool! #NLP https://t.co/xsfdw88d #NLP1 #NLP1']]
    >>> test = hashtag_list(pd.DataFrame(testdata, columns=['text'])['text'])
    >>> most_common_hashtag(test).iloc[0]
    'NLP1'
    """
    freq = pd.Series(tweet_lists.sum()) # Total Frequency Series
    counts = freq.value_counts() # Count occurrences

    def most_common(tweet_list):
        """Helper function to compute one single list"""
        if len(tweet_list) == 0: # Empty list
            return np.nan

        if len(tweet_list) == 1: # One elem
            return tweet_list[0]

        for com in counts.index: # Highest to lowest freq
            if com in tweet_list: # Check if in tweet
                return com
    
    return tweet_lists.apply(most_common)


# ---------------------------------------------------------------------
# Question # 5
# ---------------------------------------------------------------------


def create_features(ira):
    """
    :Example:
    >>> testdata = [['RT @DSC80: Text-cleaning is cool! #NLP https://t.co/xsfdw88d #NLP1 #NLP1']]
    >>> test = pd.DataFrame(testdata, columns=['text'])
    >>> out = create_features(test)
    >>> anscols = ['text', 'num_hashtags', 'mc_hashtags', 'num_tags', 'num_links', 'is_retweet']
    >>> ansdata = [['text cleaning is cool', 3, 'NLP1', 1, 1, True]]
    >>> ans = pd.DataFrame(ansdata, columns=anscols)
    >>> (out == ans).all().all()
    True
    """

    def tag_list(tweet_text):
        """Helper function to get the tag list"""
        tag_pat = '@\w+' # Hash Tag pattern
        prog = re.compile(tag_pat) # Compile
        tags = tweet_text.apply(lambda x: prog.findall(x)) # Find all
        # tags = tup.apply(lambda x: [group[0] for group in x]) # Get full tags
        return tags

    def link_list(tweet_text):
        """Helper function to get the hyperlink list"""
        link_pat = 'https?:\/\/(?! )*.' # Hash Tag pattern
        prog = re.compile(link_pat) # Compile
        links = tweet_text.apply(lambda x: prog.findall(x)) # Find all
        return links

    def is_retweet(tweet_text):
        """Helper function to check if retweet"""
        rt_pat = '^RT' # Hash Tag pattern
        prog = re.compile(rt_pat) # Compile
        rt = tweet_text.apply(lambda x: prog.findall(x)) # Find all
        return rt.apply(lambda x: True if len(x) != 0 else False)

    def clean_text(tweet_text):
        """Helper function to clean single text string"""
        remove_rt = re.sub(r'^RT', '', tweet_text) # Remove Retweet
        remove_hash = re.sub(r'#\w+', '', remove_rt) # Remove hashtags
        remove_tags = re.sub(r'@\w+', '', remove_hash) # Remove tags
        remove_link = re.sub('https?:\/\/(?! )*.+', '', remove_tags) # Remove links
        substitute = re.sub(r'[^A-Za-z0-9 ]', ' ', remove_link) # Remove non-alphanumeric
        space = re.sub(r' +', ' ', substitute) # Fix space
        lower = space.lower().strip() # Lowercase
        return lower

    tweet_text = ira['text']
    tweet_lists = hashtag_list(tweet_text) # List of hashtags
    num_hashtags = tweet_lists.apply(len) # Number of hashtags
    mc_hashtags = most_common_hashtag(tweet_lists) # Most common hashtags
    num_tags = tag_list(tweet_text).apply(len) # Number of tags
    num_links = link_list(tweet_text).apply(len) # Number of links
    is_retweet = is_retweet(tweet_text) # If tweet is retweeted
    text = tweet_text.apply(clean_text) # Cleaned text
    
    # Make dataframe
    cols = ['text', 'num_hashtags', 'mc_hashtags', 'num_tags', 'num_links', 'is_retweet']
    content = [text, num_hashtags, mc_hashtags, num_tags, num_links, is_retweet]
    index = ira.index
    return pd.DataFrame(dict(zip(cols, content)), index = ira.index)

# ---------------------------------------------------------------------
# DO NOT TOUCH BELOW THIS LINE
# IT'S FOR YOUR OWN BENEFIT!
# ---------------------------------------------------------------------


# Graded functions names! DO NOT CHANGE!
# This dictionary provides your doctests with
# a check that all of the questions being graded
# exist in your code!

GRADED_FUNCTIONS = {
    'q01': ['match_%d' % x for x in range(1, 10)],
    'q02': ['extract_personal'],
    'q03': ['tfidf_data', 'relevant_word'],
    'q04': ['hashtag_list', 'most_common_hashtag'],
    'q05': ['create_features']
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
