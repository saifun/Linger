import re
import glob
import pandas as pd
from collections import Counter
from semantic_tree import SemanticTree
from stanza_processor import Processor
from consts import PATHS, YEARS, TEMP_PATH

processor = Processor()


def open_csv_files_from_path(path):
    all_files = glob.glob(path + "/*.csv")
    for filename in all_files:
        try:
            # df = pd.read_csv(filename, encoding='utf-8')
            # print('yielding! - ' + filename)
            # yield df, filename
            df_iter = pd.read_csv(filename, chunksize=200, iterator=True, encoding='utf-8')
            print('yielding! - ' + filename)
            yield df_iter, filename
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


def is_chunk_visited(dump_track_df, filename, chunk_num):
    return filename in set(dump_track_df['visited']) and \
           chunk_num in set(dump_track_df[dump_track_df["visited"] == filename]['chunk_num'])


def generate_sentences_for_single_day(path):
    for posts_iterator, filename in open_csv_files_from_path(path):
        dump_track_df = pd.read_csv(TEMP_PATH)
        month = filename.split('-')[1]
        chunk_num = 0
        for partial_posts in posts_iterator:
            chunk_num +=1
            if not is_chunk_visited(dump_track_df, filename, chunk_num):
                print("yield chunk " + str(chunk_num))
                posts = partial_posts["text"].dropna()
                sentences_for_partial_posts = posts.str.split(r'\.|\?|\n').explode('sentences')
                sentences_for_partial_posts = sentences_for_partial_posts.replace('', float('NaN')).dropna().to_numpy()
                yield processor.get_stanza_analysis_multiple_sentences(sentences_for_partial_posts), month, filename, chunk_num
            else:
                yield None, month, filename, chunk_num


def separate_all_files_to_sub_files():
    for year in YEARS:
        path = PATHS[year]
        all_files = glob.glob(path + "/*.csv")
        for filepath in all_files:
            filename = filepath.split('/')[-1].split('.')[0]
            separate_file_to_sub_files(filepath, filename, year)


# def separate_file_to_sub_files(filepath, filename, year):
#     try:
#         df = pd.read_csv(filepath)
#         third = len(df) // 3
#         df1 = df[:third + 1]
#         df2 = df[third + 1: 2 * third + 1]
#         df3 = df[2 * third + 1:]
#         dir = '/'.join(filepath.split('/')[:-1])
#         final_filename = dir + '/subfiles_twitter_data_' + str(year) + '/' + filename
#         df1.to_csv(final_filename + "_part1.csv")
#         df2.to_csv(final_filename + "_part2.csv")
#         df3.to_csv(final_filename + "_part3.csv")
#     except:
#         print(filename)


def separate_file_to_sub_files(filepath, filename, year):
    try:
        df = pd.read_csv(filepath)
        tenth = len(df) // 10
        df1 = df[:tenth + 1]
        df2 = df[tenth + 1: 2 * tenth + 1]
        df3 = df[2 * tenth + 1: 3 * tenth + 1]
        df4 = df[3 * tenth + 1: 4 * tenth + 1]
        df5 = df[4 * tenth + 1: 5 * tenth + 1]
        df6 = df[5 * tenth + 1: 6 * tenth + 1]
        df7 = df[6 * tenth + 1: 7 * tenth + 1]
        df8 = df[7 * tenth + 1: 8 * tenth + 1]
        df9 = df[8 * tenth + 1: 9 * tenth + 1]
        df10 = df[9 * tenth + 1:]
        dir = '/'.join(filepath.split('/')[:-1])
        final_filename = dir + '/subfiles_twitter_data_' + str(year) + '/' + filename
        df_list = [df1, df2, df3, df4, df5, df6, df7, df8, df9, df10]
        for i in range(len(df_list)):
            df_list[i].to_csv(final_filename + "_part" + str(i+1) + ".csv")
    except:
        print(filename)


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


def create_dump_track_file():
    df = pd.DataFrame([], columns=['visited', 'chunk_num'])
    df.to_csv(TEMP_PATH, index=False)

# create_dump_track_file()