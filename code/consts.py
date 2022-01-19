from collections import namedtuple

# DATA_PATH = '/Users/saifun/Documents/HUJI/3 semester/67978_Needle_in_a_Data_Haystack/final_project/twitter/hebrew_twitter/{}'
# DATA_PATH = '/Users/mariatseytlin/Documents/Msc/Needle in Data Haystack/project/{}'
DATA_PATH = '/Users/mariatseytlin/Documents/Msc/Needle in Data Haystack/project/test_data/{}'

YEARS = list(range(2018, 2022))

PATHS = {
    year: DATA_PATH.format('twitter_data_' + str(year))
    for year in YEARS
}

FRAMES = {
    year: DATA_PATH.format('frame_' + str(year) + '.csv')
    for year in YEARS
}

MONTHS = ['{:02d}'.format(month) for month in range(1, 13)]

"""
Semantic representation related consts
"""
HEAD = 'head'
POS = 'pos'
WORD = 'word'
GENDER = 'gender'
DEPREL = 'deprel'
Info = namedtuple('Info', [WORD, HEAD, POS, GENDER, DEPREL])
Mismatch = namedtuple('Mismatch', [WORD, GENDER])
FEMININE = 'Fem'
MASCULINE = 'Masc'
GENDERS = (MASCULINE, FEMININE)
NOUN_POS = 'NOUN'
NUM_POS = 'NUM'
ADJ_POS = 'ADJ'
VERB_POS = 'VERB'
SUBJECT_DEPREL = 'nsubj'
ROOT = -1