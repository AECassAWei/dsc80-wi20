import os
import pandas as pd
import numpy as np
import requests
import bs4
import json


# ---------------------------------------------------------------------
# Question # 1
# ---------------------------------------------------------------------


def question1():
    """
    NOTE: You do NOT need to do anything with this function.

    The function for this question makes sure you
    have a correctly named HTML file in the right
    place. Note: This does NOT check if the supplementary files
    needed for your page are there!

    >>> os.path.exists('lab06_1.html')
    True
    """

    # Don't change this function body!
    # No python required; create the HTML file.

    return


# ---------------------------------------------------------------------
# Question # 2
# ---------------------------------------------------------------------

def extract_book_links(text):
    """
    :Example:
    >>> fp = os.path.join('data', 'products.html')
    >>> out = extract_book_links(open(fp, encoding='utf-8').read())
    >>> url = 'scarlet-the-lunar-chronicles-2_218/index.html'
    >>> out[1] == url
    True
    """
    soup = bs4.BeautifulSoup(text, 'lxml')
    urls = []
    for book in soup.find_all('article', attrs={'class':'product_pod'}):
        # title = book.find('h3').a.get('title') # Title of the book
        url = book.find('h3').a.get('href') # Url of the book
        rating = book.find('p', attrs={'class':'star-rating'}).get('class')[1] # Rating of the book
        price = float(book.find('p', attrs={'class':'price_color'}).text.strip('£')) # Price of the book

        if (rating in ['Four', 'Five']) and (price < 50): # Rating at least four star, price less than £50
            urls.append(url)
    return urls # Answer


def get_product_info(text, categories):
    """
    :Example:
    >>> fp = os.path.join('data', 'Frankenstein.html')
    >>> out = get_product_info(open(fp, encoding='utf-8').read(), ['Default'])
    >>> isinstance(out, dict)
    True
    >>> 'Category' in out.keys()
    True
    >>> out['Rating']
    'Two'
    """
    soup = bs4.BeautifulSoup(text, 'lxml')
    headers = ['Availability', 'Category', 'Description', 'Number of reviews', 'Price (excl. tax)', 'Price (incl. tax)', 'Product Type', 'Rating', 'Tax', 'Title', 'UPC'] # Headers of the dictionary

    # Type, Category and Title
    gens = soup.find('body', attrs={'id':'default'}).find('div', attrs={'class':'container-fluid'}).find_all('li')
    ptype, category, title = gens[1].text.strip(), gens[2].text.strip(), gens[3].text.strip() # Get product type, category, title
    if category not in categories: # Book not what we want
            return None

    # Availability, Description, Number of reviews, Price, Tax, Rating, UPC
    article = soup.find('article', attrs={'class':'product_page'})
    rating = article.find('p', attrs={'class':'star-rating'}).get('class')[1].strip() # Rating of the book

    desrc_head = article.find('div', attrs={'class':'sub-header', 'id':'product_description'})
    description = desrc_head.findNext('p').text.strip() # Decription

    infos = article.find('table', attrs={'class':['table', 'table-striped']})
    infos_head = infos.find_all('th')
    for head in infos_head: # Loop through headers
        if head.text == 'UPC': # UPC
            upc = head.findNext('td').text.strip()
        elif head.text == 'Price (excl. tax)': # Price (excl. tax)
            pri_ex_tax = head.findNext('td').text.strip()
        elif head.text == 'Price (incl. tax)': # Price (incl. tax)
            pri_in_tax = head.findNext('td').text.strip()
        elif head.text == 'Tax': # Tax
            tax = head.findNext('td').text.strip()
        elif head.text == 'Availability': # Availability
            avail = head.findNext('td').text.strip()
        elif head.text == 'Number of reviews': # Number of reviews
            no_rev = head.findNext('td').text.strip()

    content = [avail, category, description, no_rev, pri_ex_tax, pri_in_tax, ptype, rating, tax, title, upc]
    return dict(zip(headers, content))


def scrape_books(k, categories):
    """
    :param k: number of book-listing pages to scrape.
    :returns: a dataframe of information on (certain) books
    on the k pages (as described in the question).

    :Example:
    >>> out = scrape_books(1, ['Mystery'])
    >>> out.shape
    (1, 11)
    >>> out['Rating'][0] == 'Four'
    True
    >>> out['Title'][0] == 'Sharp Objects'
    True
    """
    df = pd.DataFrame() # Contain the result
    url = 'http://books.toscrape.com/'
    web = requests.get(url)
    for _ in range(k):
        # print(_)
        url_text = web.text # Get text of page
        book_links = extract_book_links(url_text) # Get book_links at least four star, price less than 50

        if len(book_links) != 0: # If no book of four/five star, and price less than 50
            for link in book_links: # Loop through book links
                if link[0:10] != 'catalogue/': # Fix no found urls
                    link = 'catalogue/' + link
                # print(url + link)
                book_text = requests.get(url + link).text
                dic = get_product_info(book_text, categories)
                df = df.append(dic, ignore_index=True) # Append met book to df

        # Get next page url
        soup = bs4.BeautifulSoup(url_text, 'html.parser')
        nxt = soup.find('li', attrs={'class':'next'}) # Find next button section
        if nxt == None: # Last page, no next url
            continue
        
        app = nxt.find('a').get('href') # Next page url appendant
        if app[0:10] != 'catalogue/': # Fix no found urls
            app = 'catalogue/' + app
        print(app)
        web = requests.get(url + app)
    return df

# ---------------------------------------------------------------------
# Question 3
# ---------------------------------------------------------------------

def stock_history(ticker, year, month):
    """
    Given a stock code and month, return the stock price details for that month
    as a dataframe

    >>> history = stock_history('BYND', 2019, 6)
    >>> history.shape == (20, 13)
    True
    >>> history.label.iloc[0]
    'June 03, 19'
    """
    url = 'https://financialmodelingprep.com/api/v3/historical-price-full/' + ticker
    response = requests.get(url)
    data = response.text
    js = json.loads(data) # Load json data
    
    # Construct dataframe from js file
    df = pd.DataFrame()
    for day in js.get('historical'):
        df = df.append(day, ignore_index=True, sort=False)
    # df = df.assign(year=pd.to_datetime(df['date']).dt.year)
    # df = df.assign(month=pd.to_datetime(df['date']).dt.month)
    return df[(pd.to_datetime(df['date']).dt.year == year) & (pd.to_datetime(df['date']).dt.month == month)]


def stock_stats(history):
    """
    Given a stock's trade history, return the percent change and transactions
    in billion dollars.

    >>> history = stock_history('BYND', 2019, 6)
    >>> stats = stock_stats(history)
    >>> len(stats[0]), len(stats[1])
    (7, 6)
    >>> float(stats[0][1:-1]) > 30
    True
    >>> float(stats[1][:-1]) > 1
    True
    """
    open_id = pd.to_datetime(history['date']).idxmin()
    close_id = pd.to_datetime(history['date']).idxmax()
    close_p = history.loc[close_id, 'close'] # Close price for month
    open_p = history.loc[open_id, 'open'] # Opening price for month
    pct_change = (close_p - open_p) / open_p * 100

    tot_tran = ((history['close'] + history['open']) / 2 * history['volume'] / 1000000000).sum()
    
    change = '%+.2f' % (pct_change) + '%'
    tran = '%.2f' % (tot_tran) + 'B'
    return (change, tran)


# ---------------------------------------------------------------------
# Question # 4
# ---------------------------------------------------------------------

def get_comments(storyid):
    """
    Returns a dataframe of all the comments below a news story
    >>> out = get_comments(18344932)
    >>> out.shape
    (18, 5)
    >>> out.loc[5, 'by']
    'RobAtticus'
    >>> out.loc[5, 'time'].day
    31
    """
    news_endpoint = 'https://hacker-news.firebaseio.com/v0/item/' + str(storyid) + '.json'
    news = json.loads(requests.get(news_endpoint).text) # Story info
    kids = news.get('kids') # Kids of story
    
    ids = get_kids_ids(kids, [])
    df = pd.DataFrame()
    cols = ['id', 'by', 'parent', 'text', 'time']
    for idx in ids: 
        kids_endpoint = 'https://hacker-news.firebaseio.com/v0/item/' + str(idx) + '.json'
        com = json.loads(requests.get(kids_endpoint).text)

        by = com.get('by') # Author
        parent = com.get('parent') # Parent id
        text = com.get('text') # Comment text
        time = pd.to_datetime(com.get('time')) # Time of comment # How to change it to datetime??????????????????????????????

        content = [idx, by, parent, text, time] # Content list
        comment = dict(zip(cols, content)) # Zip into dataframe
        df = df.append(comment, ignore_index=True) # Append to answer
    return df


# Helper function to get kids ids with recursion
def get_kids_ids(kids, ids):
    """
    Get all the kids ids within kids.
    
    :param kids: a list of kids ids
    :param ids: id list to append to and return
    :return: a list of kids ids (recursive)
    """
    for kid in kids: # Loop through to check if comment
        kids_endpoint = 'https://hacker-news.firebaseio.com/v0/item/' + str(kid) + '.json'
        info = json.loads(requests.get(kids_endpoint).text)
        
        if not info.get('dead'): # Not dead comment 
            ids.append(info.get('id')) # Append to list ids
            
            if (info.get('kids') != None) and (info.get('type') == 'comment'): # Has kids, is comment
                kids_ids = get_kids_ids(info.get('kids'), []) # Kids id list

                if len(kids_ids) != 0: # Has kids
                    for idx in kids_ids:
                        ids.append(idx) # Append kids to answer

    return ids


# ---------------------------------------------------------------------
# DO NOT TOUCH BELOW THIS LINE
# IT'S FOR YOUR OWN BENEFIT!
# ---------------------------------------------------------------------


# Graded functions names! DO NOT CHANGE!
# This dictionary provides your doctests with
# a check that all of the questions being graded
# exist in your code!

GRADED_FUNCTIONS = {
    'q01': ['question1'],
    'q02': ['extract_book_links', 'get_product_info', 'scrape_books'],
    'q03': ['stock_history', 'stock_stats'],
    'q04': ['get_comments']
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
                In %s, part %s is missing" % (q, elt)
                raise Exception(stmt)

    return True
