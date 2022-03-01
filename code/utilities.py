import re
import glob
import pandas as pd
from collections import Counter
from semantic_tree import SemanticTree
from stanza_processor import Processor
from consts import PATHS, YEARS, TEMP_PATH
from stanza_light_processor import LightProcessor

processor = Processor()
light_processor = LightProcessor()


def open_csv_files_from_path(path, verbose=False):
    all_files = glob.glob(path + "/*.csv")
    for filename in all_files:
        try:
            df_iter = pd.read_csv(filename, chunksize=500, iterator=True, encoding='utf-8')
            print('yielding! - ' + filename)
            yield df_iter, filename
        except:
            print(filename)


def generate_df_from_csv_path(path):
    all_files = glob.glob(path + "/*.csv")
    for filename in all_files:
        try:
            df = pd.read_csv(filename, encoding='utf-8')
            month = filename.split('-')[1]
            # print('yielding! - ' + filename)
            yield df, month, filename
ע        except:
            if verbose:
                print(filename)


def get_posts_from_corpus(path):
    for single_day_posts, filename in open_csv_files_from_path(path):
        yield from get_multiple_columns(single_day_posts, ['text', 'created_at'])

        
def invert_words(words):
    return [w[::-1] for w in words]


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
    """
    This function generates processed sentences from a data filepath by Stanza processor
    """
    for posts_iterator, filename in open_csv_files_from_path(path):
        dump_track_df = pd.read_csv(TEMP_PATH)
        month = filename.split('-')[1]
        chunk_num = 0
        for partial_posts in posts_iterator:
            chunk_num += 1
            if not is_chunk_visited(dump_track_df, filename, chunk_num):
                print("yield chunk " + str(chunk_num))
                posts = partial_posts["text"].dropna()
                sentences_for_partial_posts = posts.str.split(r'\.|\?|\n').explode('sentences')
                sentences_for_partial_posts = sentences_for_partial_posts.replace('', float('NaN')).dropna().to_numpy()
                yield processor.get_stanza_analysis_multiple_sentences(sentences_for_partial_posts), month, filename, chunk_num
            else:
                yield None, month, filename, chunk_num


def generate_sentences_for_single_day_with_light_processor(path):
    """
    This function generates processed sentences from a data filepath by Stanza light processor - a processor without
    dependency parsing
    """
    for posts_iterator, filename in open_csv_files_from_path(path):
        month = filename.split('-')[1]
        chunk_num = 0
        for partial_posts in posts_iterator:
            chunk_num +=1
            print("yield chunk " + str(chunk_num))
            posts = partial_posts["text"].dropna()
            sentences_for_partial_posts = posts.str.split(r'\.|\?|\n').explode('sentences')
            sentences_for_partial_posts = sentences_for_partial_posts.replace('', float('NaN')).dropna().to_numpy()
            yield light_processor.get_stanza_analysis_multiple_sentences(sentences_for_partial_posts), month, filename, chunk_num


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
    return [item for sublist in map(lambda word: re.split(',| |\.|\?|\n', str(word)), array) for item
            in sublist]


def create_all_words_histogram(all_text_df):
    all_text_df.str.split().map(lambda x: len(x)).hist()


def create_dump_track_file():
    df = pd.DataFrame([], columns=['visited', 'chunk_num'])
    df.to_csv(TEMP_PATH, index=False)