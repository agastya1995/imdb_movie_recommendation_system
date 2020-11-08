import numpy as np
from difflib import SequenceMatcher

def get_movie_id(title, data):
    '''
    Takes in the title of the movie, and returns the movie id, title, genres and year
    returns None if no match is found 
    '''
    title = title.lower()
    details = data[data['title'] == title]
    if len(details) == 0:
        return None
    else:
        movie_id = details['ID'].values[0]
        title = details['title'].values[0].title()
        genres = details['genres'].values[0]
        year = details['year'].values[0]

        return (movie_id, title, genres, year)


def top_cosine_similarity(movie_id, data, top_n=11):
    '''
    Returns the top n most similar titles based on the movie matrix of svd decomposition
    '''
    movie_row = data[movie_id, :]
    magnitude = np.sqrt(np.einsum('ij, ij -> i', data, data))
    similarity = np.dot(movie_row, data.T) / (magnitude[movie_id] * magnitude)
    sort_indexes = np.argsort(-similarity)
    return sort_indexes[1:top_n]


def return_similar_titles(search_item, list_of_items, top_n=10, threshold=None):
    '''
    Returns the top n most similar items from the list of items
    list of items must be a pandas series
    threshold - the similarity ratio threshold (calculated using python's inbuilt sequence matcher)
    '''
    ratios = list_of_items.map(lambda x: SequenceMatcher(a=search_item, b=x).ratio())
    ratios.sort_values(ascending=False, inplace=True)
    if threshold:
        indexes = ratios[ratios>threshold].index
        indexes = indexes[:top_n]
    else:
        indexes = ratios[:top_n].index
    return list_of_items.iloc[indexes]
    