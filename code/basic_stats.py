import os
import pandas as pd
from collections import defaultdict
from utilities import get_most_common_tokens_from_column, open_csv_files_from_path
from consts import YEARS, PATHS

INTERESTING_WORDS = ['בומר', 'להשים', 'חיסון', 'אותכם', 'חיימי', 'פיפי', 'ביבי']


def count_words_by_month(words):
    word_count_per_month = {word: defaultdict(int) for word in words}
    for year in YEARS:
        for single_day_posts, filepath in open_csv_files_from_path(PATHS[year]):
            all_elements_by_count = get_most_common_tokens_from_column(single_day_posts, column_name='text')
            filename = os.path.splitext(os.path.basename(filepath))[0]
            month = filename.split('-')[1]
            for word in word_count_per_month:
                word_count_per_month[word][f'{year}-{month}'] += all_elements_by_count.get(word, 0)


def write_data_to_csv(word_count):
    for word in word_count:
        word_df = pd.DataFrame(list(word_count[word].items()), columns=['month', 'count'])
        word_df.to_csv(f'results/{word}_count.csv', index=False, header=True)


count_words_by_month(INTERESTING_WORDS)
