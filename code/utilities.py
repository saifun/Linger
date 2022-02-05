import re
import glob
import pandas as pd
from collections import Counter


def open_csv_files_from_path(path):
    all_files = glob.glob(path + "/*.csv")
    for filename in all_files:
        try:
            df = pd.read_csv(filename, encoding='utf-8')
            print('yielding! - ' + filename)
            yield df, filename
        except:
            print(filename)


def get_posts_from_corpus(path):
    for single_day_posts, filename in open_csv_files_from_path(path):
        yield from get_multiple_columns(single_day_posts, ['text', 'created_at'])


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


def get_multiple_columns(df, column_names):
    return df[column_names].dropna().to_numpy()


def get_df_lines_by_condition(df, column_name, value):
    return df.loc[df[column_name] == value]


def get_all_tokens_from_array(array):
    # return [item for sublist in map(lambda settlement_name: settlement_name.split(), array) for item in sublist]
    return [item for sublist in map(lambda word: re.split(',| |\.|\?|\n', word), array) for item
            in sublist]


# def get_df_from_result(question_number, results):
#     return pandas.DataFrame([[question_number, res[1], res[0]] for res in results], columns=COLUMNS)

def create_all_words_histogram(all_text_df):
    all_text_df.str.split().map(lambda x: len(x)).hist()

# def plot_top_words_barchart(all_text_df):
#     corpus = get_corpus(all_text_df)
#     counter = Counter(corpus)
#     most = counter.most_common()
#     word_list, count_list = [], []
#     for word, count in most[:20]:
#         word_list.append(word)
#         count_list.append(count)
#
#     seaborn.barplot(x=count_list, y=invert_words(word_list))
#
#
# def plot_top_ngrams_barchart(all_text_df, n=2):
#     corpus = get_corpus(all_text_df)
#     top_n_bigrams = get_top_ngrams_from_corpus(all_text_df, n)[:20]
#     word_list, count_list = map(list, zip(*top_n_bigrams))
#     seaborn.barplot(x=count_list, y=invert_words(word_list))
