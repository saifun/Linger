import re
import glob
import pandas as pd
from collections import Counter
from semantic_tree import SemanticTree


def open_csv_files_from_path(path):
    all_files = glob.glob(path + "/*.csv")
    for filename in all_files:
        try:
            df = pd.read_csv(filename, encoding='utf-8')
            print('yielding! - ' + filename)
            yield df, filename
        except:
            print(filename)


def generate_sentences(path):
    for single_day_posts, filename in open_csv_files_from_path(path):
        posts = get_single_column(single_day_posts, 'text')
        month = filename.split('-')[1]
        for post in posts:
            sentences = re.split('\.|\?|\n', post)
            for sentence in sentences:
                if sentence == '':
                    continue
                yield sentence, month, SemanticTree(sentence)


def get_most_common_tokens_from_column(df, column_name):
    column_values = get_single_column(df, column_name)
    all_tokens = get_all_tokens_from_array(column_values)
    return get_most_common_values_from_array(all_tokens)


def get_most_common_values_from_column(df, column_name):
    district_names = get_single_column(df, column_name)
    return get_most_common_values_from_array(district_names)


def get_most_common_values_from_array(array):
    array = filter(lambda value: value is not None, array)
    return dict(Counter(array).most_common())


def get_single_column(df, column_name):
    return df[column_name].dropna().to_numpy()


def get_df_lines_by_condition(df, column_name, value):
    return df.loc[df[column_name] == value]


def get_all_tokens_from_array(array):
    # return [item for sublist in map(lambda settlement_name: settlement_name.split(), array) for item in sublist]
    return [item for sublist in map(lambda word: re.split(',| |\.|\?|\n', word), array) for item
            in sublist]


def create_all_words_histogram(all_text_df):
    all_text_df.str.split().map(lambda x: len(x)).hist()
