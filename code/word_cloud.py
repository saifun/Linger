import json
import string
from collections import defaultdict
from utilities import get_most_common_tokens_from_column, open_csv_files_from_path
from visualize import visualize_word_cloud
from consts import YEARS, PATHS

WORDS_DISTRIBUTION_FILE = 'results/word_cloud/all_words.json'
NOT_STOP_WORDS_DISTRIBUTION_FILE = 'results/word_cloud/not_stop_words.json'
STOP_WORDS_FILE = 'hebrew_stopwords.txt'


def create_word_count():
    total_word_count = defaultdict(int)
    for year in YEARS:
        for single_day_posts, filepath in open_csv_files_from_path(PATHS[year]):
            all_elements_by_count = get_most_common_tokens_from_column(single_day_posts, column_name='text')
            for word, count in all_elements_by_count.items():
                total_word_count[word] += count
    print(total_word_count)
    with open(WORDS_DISTRIBUTION_FILE, 'w') as output_file:
        output_file.write(json.dumps(total_word_count))


def create_word_count_without_stop_words():
    with open(WORDS_DISTRIBUTION_FILE, 'r') as all_words_file:
        all_words = json.loads(all_words_file.read())
    with open(STOP_WORDS_FILE, 'r') as stop_word_file:
        stop_words = stop_word_file.read().split('\n')
    words_without_stopwords = {
        word: count for word, count in all_words.items()
        if word not in stop_words
    }
    with open(NOT_STOP_WORDS_DISTRIBUTION_FILE, 'w') as output_file:
        output_file.write(json.dumps(words_without_stopwords))


def _clean_words_from_non_hebrew():
    with open(WORDS_DISTRIBUTION_FILE, 'r') as input_file:
        all_words = json.loads(input_file.read())
    filtered_words = {word: count for word, count in all_words.items() if _is_hebrew_word(word)}
    with open(WORDS_DISTRIBUTION_FILE, 'w') as output:
        output.write(json.dumps(filtered_words))


def _is_hebrew_word(word):
    return word != '' and not word[0] in string.ascii_letters and word[0] not in string.punctuation


if __name__ == '__main__':
    # create_word_count()
    visualize_word_cloud(WORDS_DISTRIBUTION_FILE)
    # create_word_count_without_stop_words()
    visualize_word_cloud(NOT_STOP_WORDS_DISTRIBUTION_FILE)
