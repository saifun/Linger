import pandas
from collections import Counter
import re

def get_most_common_tokens_from_column(df, column_name):
    column_values = get_single_column(df, column_name)
    all_tokens = get_all_tokens_from_array(column_values)
    return get_most_common_values_from_array(all_tokens)


def get_most_common_values_from_column(df, column_name):
    district_names = get_single_column(df, column_name)
    return get_most_common_values_from_array(district_names)


def get_most_common_values_from_array(array):
    array = filter(lambda value: value is not None, array)
    return Counter(array).most_common()


def get_single_column(df, column_name):
    return df[column_name].dropna().to_numpy()


def get_df_lines_by_condition(df, column_name, value):
    return df.loc[df[column_name] == value]

def get_all_tokens_from_array(array):
    # return [item for sublist in map(lambda settlement_name: settlement_name.split(), array) for item in sublist]
    return [item for sublist in map(lambda settlement_name: re.split(',| |\.|\?|\n', settlement_name), array) for item in sublist]

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