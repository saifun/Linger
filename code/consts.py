from collections import namedtuple

DATA_PATH = '/Users/mariatseytlin/Documents/Msc/Needle in Data Haystack/project/{}'
TEMP_PATH = 'temp/dump_track.csv'

YEARS = list(range(2018, 2022))

PATHS = {
    year: DATA_PATH.format('twitter_data_' + str(year))
    for year in YEARS
}

SUBFILES_PATH = {
    year: DATA_PATH.format('twitter_data_' + str(year) + '/subfiles_twitter_data_' + str(year))
    for year in YEARS
}

GENDER_MISMATCH_PATHS = {
    year: DATA_PATH.format('twitter_data_' + str(year) + '/gender_mismatch_' + str(year))
    for year in YEARS
}

FUTURE_VERB_PATHS = {
    year: DATA_PATH.format('twitter_data_' + str(year) + '/future_verb_' + str(year))
    for year in YEARS
}

FRAMES = {
    year: DATA_PATH.format('frame_' + str(year) + '.csv')
    for year in YEARS
}

MONTHS = ['{:02d}'.format(month) for month in range(1, 13)]

HEB_CAHRS_START = "\u0590"
HEB_CAHRS_END = "\u05EA"
RECORDS_COUNT_STATS = {
    2018: 15523736,
    2019: 23823005,
    2020: 52823817,
    2021: 13168095,
}

"""
Semantic representation related consts
"""
HEAD = 'head'
POS = 'pos'
WORD = 'word'
GENDER = 'gender'
TENSE = 'tense'
NUMBER = 'number'
PERSON = 'person'
DEPREL = 'deprel'
Info = namedtuple('Info', [WORD, HEAD, POS, GENDER, TENSE, NUMBER, PERSON, DEPREL])
NumberInfo = namedtuple('NumberInfo', [WORD, POS, GENDER, TENSE, NUMBER, PERSON])
Mismatch = namedtuple('Mismatch', [WORD, GENDER])
FEMININE = 'Fem'
MASCULINE = 'Masc'
GENDERS = (MASCULINE, FEMININE)
NOUN_POS = 'NOUN'
NUM_POS = 'NUM'
ADJ_POS = 'ADJ'
VERB_POS = 'VERB'
PRONOUN_POS = 'PRON'
FUTURE_TENSE = 'Fut'
SINGULAR_NUMBER = 'Sing'
THIRD_PERSON = '3'
FIRST_PERSON = '1'
SUBJECT_DEPREL = 'nsubj'
ROOT = -1
