from utilities import generate_sentences
from consts import YEARS, PATHS

if __name__ == '__main__':
    for year in YEARS:
        generate_sentences(PATHS[year])