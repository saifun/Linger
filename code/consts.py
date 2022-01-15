from collections import namedtuple

DATA_PATH = '/Users/saifun/Documents/HUJI/3 semester/67978_Needle_in_a_Data_Haystack/final_project/twitter/hebrew_twitter/{}'

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
FEMININE = 'Fem'
MASCULINE = 'Masc'
GENDERS = (MASCULINE, FEMININE)
NOUN_POS = 'NOUN'
NUM_POS = 'NUM'
ADJ_POS = 'ADJ'
VERB_POS = 'VERB'
ROOT = -1