import requests


def url_list():
    """
    A list of urls to scrape.

    :Example:
    >>> isinstance(url_list(), list)
    True
    >>> len(url_list()) > 1
    True
    """
    url_list = []
    for num in range(0, 26):
        url_list.append('http://example.webscraping.com/places/default/index/' + str(num))
    return url_list


def request_until_successful(url, N):
    """
    impute (i.e. fill-in) the missing values of each column 
    using the last digit of the value of column A.

    :Example:
    >>> resp = request_until_successful('http://quotes.toscrape.com', N=1)
    >>> resp.ok
    True
    >>> resp = request_until_successful('http://example.webscraping.com/', N=1)
    >>> isinstance(resp, requests.models.Response) or (resp is None)
    True
    """    
    if N==0:
        return None
    
    rq = requests.get(url)
    
    if rq.status_code != 200 and N>0:
        return request_until_successful(url, N-1)
    
    return rq
