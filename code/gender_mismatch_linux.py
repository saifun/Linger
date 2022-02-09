#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from collections import defaultdict
from collections import namedtuple
import stanza
import re
import glob
import pandas as pd
from collections import Counter



# stanza.download('he')
class Processor:
    def __init__(self):
        self.heb_nlp = stanza.Pipeline(lang='he', processors='tokenize,mwt,pos,lemma,depparse', verbose=False)

    def get_stanza_analysis(self, text):
        text += " XX"
        doc = self.heb_nlp(text)
        lst = []
        for sen in doc.sentences:
            for token in sen.tokens:
                for word in token.words:
                    features = [(word.text,
                                 word.lemma,
                                 word.upos,
                                 word.xpos,
                                 word.head,
                                 word.deprel,
                                 word.feats)]

                    df = pd.DataFrame(features, columns=["text", "lemma", "upos", "xpos", "head", "deprel", "feats"])
                    lst.append(df)
        tot_df = pd.concat(lst, ignore_index=True)
        tot_df = tot_df.shift(1).iloc[1:]
        tot_df["head"] = tot_df["head"].astype(int)
        return tot_df['text'], tot_df['head'], tot_df['upos'], tot_df['feats'], tot_df['deprel']

    def get_stanza_analysis_multiple_sentences(self, sentences_list):
        in_docs = [stanza.Document([], text=sent) for sent in sentences_list]
        out_docs = self.heb_nlp(in_docs)
        dfs = []
        for doc in out_docs:
            for sen in doc.sentences:
                lst = []
                for token in sen.tokens:
                    for word in token.words:
                        features = [(doc.text,
                                    word.text,
                                     word.lemma,
                                     word.upos,
                                     word.xpos,
                                     word.head,
                                     word.deprel,
                                     word.feats)]

                        df = pd.DataFrame(features, columns=["sentence", "text", "lemma", "upos", "xpos", "head", "deprel", "feats"])
                        lst.append(df)
                tot_df = pd.concat(lst, ignore_index=True)
                tot_df = tot_df.shift(1).iloc[1:]
                tot_df["head"] = tot_df["head"].astype(int)
                dfs.append(tot_df)
        return dfs


# DATA_PATH = '/Users/saifun/Documents/HUJI/3 semester/67978_Needle_in_a_Data_Haystack/final_project/twitter/hebrew_twitter/{}'
# DATA_PATH = '/Users/mariatseytlin/Documents/Msc/Needle in Data Haystack/project/{}'
# DATA_PATH = '/Users/mariatseytlin/Documents/Msc/Needle in Data Haystack/project/test_data/{}'
DATA_PATH = '/cs/labs/gabis/nfun/hebrew_twitter/{}'
# TEMP_PATH = 'temp/dump_track.csv'
TEMP_PATH = '/cs/labs/gabis/nfun/hebrew_twitter/dump_track.csv'

YEARS = list(range(2018, 2022))

# PATHS = {
#     year: DATA_PATH.format('twitter_data_' + str(year))
#     for year in YEARS
# }

PATHS = {
    year: DATA_PATH.format('data_' + str(year))
    for year in YEARS
}

SUBFILES_PATH = {
    year: DATA_PATH.format('twitter_data_' + str(year) + '/subfiles_twitter_data_' + str(year))
    for year in YEARS
}

# GENDER_MISMATCH_PATHS = {
#     year: DATA_PATH.format('twitter_data_' + str(year) + '/gender_mismatch_' + str(year))
#     for year in YEARS
# }

GENDER_MISMATCH_PATHS = {
    year: DATA_PATH.format('data_' + str(year) + '/gender_mismatch_' + str(year))
    for year in YEARS
}

# FUTURE_VERB_PATHS = {
#     year: DATA_PATH.format('twitter_data_' + str(year) + '/future_verb_' + str(year))
#     for year in YEARS
# }

FUTURE_VERB_PATHS = {
    year: DATA_PATH.format('data_' + str(year) + '/future_verb_' + str(year))
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
TENSE = 'tense'
NUMBER = 'number'
PERSON = 'person'
DEPREL = 'deprel'
Info = namedtuple('Info', [WORD, HEAD, POS, GENDER, TENSE, NUMBER, PERSON, DEPREL])
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

class SemanticTree:
    def __init__(self, text):
        self.text = text
        self.processor = Processor()

    def get_feature_dict(self, feature):
        # feature example: Gender=Masc|HebBinyan=HUFAL|Number=Sing|Person=3|Tense=Fut|Voice=Pass
        if feature:
            feature_dict = dict()
            all_features = feature.split('|')
            for f in all_features:
                split_f = f.split('=')
                feature_dict[split_f[0]] = split_f[1]
            return feature_dict
        return None

    def get_gender(self, feature_dict):
        if feature_dict and 'Gender' in feature_dict:
            return feature_dict['Gender']
        return None

    def get_tense(self, feature_dict):
        if feature_dict and 'Tense' in feature_dict:
            return feature_dict['Tense']
        return None

    def get_number(self, feature_dict):
        if feature_dict and 'Number' in feature_dict:
            return feature_dict['Number']
        return None

    def get_person(self, feature_dict):
        if feature_dict and 'Person' in feature_dict:
            return feature_dict['Person']
        return None

    def parse_text(self):
        parsed_text, tree, pos, features, deprel = self.processor.get_stanza_analysis(self.text)
        word_list = list(zip(list(parsed_text), map(lambda head: head - 1, list(tree)), list(pos),
                             map(lambda feature: self.get_feature_dict(feature), list(features)),list(deprel)))
        self.tree = {index: Info(word, head, pos, self.get_gender(feature_dict), self.get_tense(feature_dict),
                                 self.get_number(feature_dict), self.get_person(feature_dict), deprel)
                        for index, (word, head, pos, feature_dict, deprel) in enumerate(word_list)}
        self.parsed_text = parsed_text



    def parse_text_without_processing(self, parsed_text, tree, pos, features, deprel):
        word_list = list(zip(list(parsed_text), map(lambda head: head - 1, list(tree)), list(pos),
                             map(lambda feature: self.get_feature_dict(feature), list(features)), list(deprel)))
        self.tree = {index: Info(word, head, pos, self.get_gender(feature_dict), self.get_tense(feature_dict),
                                 self.get_number(feature_dict), self.get_person(feature_dict), deprel)
                     for index, (word, head, pos, feature_dict, deprel) in enumerate(word_list)}
        self.parsed_text = parsed_text

    def __str__(self):
        tree_rep = '{\n'
        for index, info in self.tree.items():
            tree_rep += '{}: {}\n'.format(index, info)
        tree_rep += '}\n'
        return tree_rep

    def is_verb(self, word_index):
        return self.tree[word_index].pos == 'VERB'

    def is_root(self, word_index):
        return self.tree[word_index].head == ROOT

    def get_word_in_index(self, index):
        return self.tree[index].word

    def find_verb_root(self, word_index):
        while (not self.is_root(word_index)):
            if self.is_verb(word_index):
                return word_index
            word_index = self.tree[word_index].head
        return word_index

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


def generate_df_from_csv_path(path):
    all_files = glob.glob(path + "/*.csv")
    for filename in all_files:
        try:
            df = pd.read_csv(filename, encoding='utf-8')
            month = filename.split('-')[1]
            print('yielding! - ' + filename)
            yield df, month, filename
        except:
            print(filename)


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

# def generate_sentences_for_single_day(path):
#     for posts_iterator, filename in open_csv_files_from_path(path):
#         dump_track_df = pd.read_csv(TEMP_PATH)
#         month = filename.split('-')[1]
#         chunk_num = 0
#         for partial_posts in posts_iterator:
#             chunk_num +=1
#             # if not is_chunk_visited(dump_track_df, filename, chunk_num):
#             if filename == "/Users/mariatseytlin/Documents/Msc/Needle in Data Haystack/project/test_data/twitter_data_2018/twitter_dump_2018-03-10.csv"\
#                     and chunk_num == 26:
#                 print("yield chunk " + str(chunk_num))
#                 posts = partial_posts["text"].dropna()
#                 sentences_for_partial_posts = posts.str.split(r'\.|\?|\n').explode('sentences')
#                 sentences_for_partial_posts = sentences_for_partial_posts.replace('', float('NaN')).dropna().to_numpy()
#                 yield processor.get_stanza_analysis_multiple_sentences(sentences_for_partial_posts), month, filename, chunk_num
#             else:
#                 yield None, month, filename, chunk_num


def separate_all_files_to_sub_files():
    for year in YEARS:
        path = PATHS[year]
        all_files = glob.glob(path + "/*.csv")
        for filepath in all_files:
            filename = filepath.split('/')[-1].split('.')[0]
            separate_file_to_sub_files(filepath, filename, year)

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

def is_verb_head_word_and_target_is_not_a_subject(head_pos, target_deprel):
    return head_pos == VERB_POS and target_deprel != SUBJECT_DEPREL


def find_mismatch_for_head(target_pos, head_idx, sent_parse_tree):
    curr_noun = sent_parse_tree[head_idx]
    target_pos_mismatches = []
    for index in sent_parse_tree:
        if sent_parse_tree[index].pos == target_pos and sent_parse_tree[index].head == head_idx:
            if is_verb_head_word_and_target_is_not_a_subject(sent_parse_tree[head_idx].pos, sent_parse_tree[index].deprel):
                continue
            if sent_parse_tree[index].gender in GENDERS\
                    and curr_noun.gender != sent_parse_tree[index].gender:
                target_pos_mismatches.append(sent_parse_tree[index])
    return target_pos_mismatches


def find_wrong_future_verb_for_head(head_idx, sent_parse_tree):
    for index in sent_parse_tree:
        if sent_parse_tree[index].pos == PRONOUN_POS and sent_parse_tree[index].head == head_idx\
                    and sent_parse_tree[index].deprel == SUBJECT_DEPREL:
            if is_pronoun_first_singular(sent_parse_tree[index]):
                return sent_parse_tree[index]
    return None


def is_pronoun_first_singular(pronoun_info):
    return pronoun_info.number == SINGULAR_NUMBER and pronoun_info.person == FIRST_PERSON


def is_verb_future_third_singular(verb_info):
    return verb_info.tense == FUTURE_TENSE and verb_info.number == SINGULAR_NUMBER\
           and verb_info.person == THIRD_PERSON and verb_info.word[0] == 'י'


def find_wrong_future_verb_for_sentence(sent_parse_tree):
    # example: {0: Info(word='אני', head=1, pos='PRON', gender='Fem,Masc', tense=None, number='Sing', person='1', deprel='nsubj'),
    #   1: Info(word='יתן', head=-1, pos='VERB', gender='Masc', tense='Fut', number='Sing', person='3', deprel='root'),
    #   2: Info(word='ל_', head=3, pos='ADP', gender=None, tense=None, number=None, person=None, deprel='case'),
    #   3: Info(word='_הוא', head=1, pos='PRON', gender='Masc', tense=None, number='Sing', person='3', deprel='obl'),
    #   4: Info(word='את', head=5, pos='ADP', gender=None, tense=None, number=None, person=None, deprel='case:acc')}
    verb_indices = [index for index in sent_parse_tree if sent_parse_tree[index].pos == VERB_POS]
    future_verb_mistakes = dict()
    for verb_idx in verb_indices:
        if is_verb_future_third_singular(sent_parse_tree[verb_idx]):
            wrong_pronoun = find_wrong_future_verb_for_head(verb_idx, sent_parse_tree)
            if wrong_pronoun:
                future_verb_mistakes[sent_parse_tree[verb_idx]] = wrong_pronoun
    return future_verb_mistakes


def find_gender_mismatches_for_sentence(sent_parse_tree, head_pos, target_pos):
    head_indices = [index for index in sent_parse_tree if sent_parse_tree[index].pos == head_pos\
                    and sent_parse_tree[index].gender in GENDERS]
    gender_mismatches = defaultdict(list)
    for head_idx in head_indices:
        mismatches = find_mismatch_for_head(target_pos, head_idx, sent_parse_tree)
        if mismatches:
            gender_mismatches[sent_parse_tree[head_idx]].extend(find_mismatch_for_head(target_pos, head_idx, sent_parse_tree))
    return gender_mismatches


def create_df_future_verb_for_sentence(sent, parse_tree, month, year):
    future_verb_mistakes_dict = find_wrong_future_verb_for_sentence(parse_tree)
    if future_verb_mistakes_dict:
        new_df = create_new_wrong_future_verb_df()
        for verb in future_verb_mistakes_dict:
            new_df['verb'] = [verb.word]
            new_df['pronoun'] = [future_verb_mistakes_dict[verb].word]
        new_df['year'] = year
        new_df['month'] = month
        new_df['sentence'] = sent
        return new_df
    return None


def create_df_gender_mismatch_for_sentence(sent, parse_tree, head_pos, target_pos, month, year):
    gender_mismatch_dict_noun_num = find_gender_mismatches_for_sentence(parse_tree, head_pos, target_pos)
    if gender_mismatch_dict_noun_num:
        new_df = create_new_gender_mismatch_df()
        for head_word in gender_mismatch_dict_noun_num:
            for word in gender_mismatch_dict_noun_num[head_word]:
                added_df = pd.DataFrame([[sent, month, year, head_word.word, head_word.gender, word.word]],
                                        columns=['sentence', 'month', 'year', 'head', 'head_gender', 'mismatch'])
                new_df = new_df.append(added_df, ignore_index=True)
        return new_df
    return None


def create_new_wrong_future_verb_df():
    new_df = pd.DataFrame([], columns=['sentence', 'month', 'year', 'verb', 'pronoun'])
    return new_df


def create_new_gender_mismatch_df():
    new_df = pd.DataFrame([], columns=['sentence', 'month', 'year', 'head', 'head_gender', 'mismatch'])
    return new_df


def create_df_gender_mismatch_for_sentence_noun_num(sent, parse_tree, month, year):
    return create_df_gender_mismatch_for_sentence(sent, parse_tree, NOUN_POS, NUM_POS, month, year)


def create_df_gender_mismatch_for_sentence_noun_adj(sent, parse_tree, month, year):
    return create_df_gender_mismatch_for_sentence(sent, parse_tree, NOUN_POS, ADJ_POS, month, year)


def create_df_gender_mismatch_for_sentence_verb_noun(sent, parse_tree, month, year):
    return create_df_gender_mismatch_for_sentence(sent, parse_tree, VERB_POS, NOUN_POS, month, year)


def write_data_to_csv(mismatch_name, mismatch_dict):
    mismatch_df = pd.DataFrame(list(mismatch_dict.items()), columns=['month', 'count'])
    mismatch_df.to_csv(f'results/gender_mismatch/{mismatch_name}_count.csv', index=False, header=True)

def get_gender_mismatch_dump_path(filename, mismatch_name, year, number):
    split_filename = filename.split('/')
    dir_path = '/'.join(split_filename[:-1]) + '/gender_mismatch_' + str(year) + '/'
    end_filename = '_'.join(split_filename[-1].split('_')[2:])
    return dir_path + 'gender_mismatch_dump_' + mismatch_name + '_chunk' + str(number) + '_' + end_filename


def get_future_verb_dump_path(filename, year, number):
    split_filename = filename.split('/')
    dir_path = '/'.join(split_filename[:-1]) + '/future_verb_' + str(year) + '/'
    end_filename = '_'.join(split_filename[-1].split('_')[2:])
    return dir_path + 'future_verb_dump_chunk' + str(number) + '_' + end_filename


def create_csv_dumps_gender_mismatch_per_year_multiple_sentences():
    for year in YEARS:
        for stanza_analysis_list, month, filename, chunk_num in generate_sentences_for_single_day(PATHS[year]):
            if stanza_analysis_list:
                dump_track_df = pd.read_csv(TEMP_PATH)
                new_df_noun_num = create_new_gender_mismatch_df()
                new_df_noun_adj = create_new_gender_mismatch_df()
                new_df_verb_noun = create_new_gender_mismatch_df()
                new_df_future_verb = create_new_wrong_future_verb_df()
                for sent_df in stanza_analysis_list:
                    semantic_tree = SemanticTree(sent_df['text'])
                    semantic_tree.parse_text_without_processing(sent_df['text'], sent_df['head'], sent_df['upos'], sent_df['feats'], sent_df['deprel'])
                    parse_tree = semantic_tree.tree
                    if not sent_df.empty:
                        sentence_text = sent_df['sentence'][1]
                        noun_num_df = create_df_gender_mismatch_for_sentence_noun_num(sentence_text, parse_tree, month, year)
                        noun_adj_df = create_df_gender_mismatch_for_sentence_noun_adj(sentence_text, parse_tree, month, year)
                        verb_noun_df = create_df_gender_mismatch_for_sentence_verb_noun(sentence_text, parse_tree, month, year)
                        future_verb_df = create_df_future_verb_for_sentence(sentence_text, parse_tree, month, year)
                        if isinstance(noun_num_df, pd.DataFrame):
                            # noun_num_df.to_csv(get_gender_mismatch_dump_path(filename, 'noun_num'))
                            new_df_noun_num = new_df_noun_num.append(noun_num_df)
                        if isinstance(noun_adj_df, pd.DataFrame):
                            # noun_adj_df.to_csv(get_gender_mismatch_dump_path(filename, 'noun_adj'))
                            new_df_noun_adj = new_df_noun_adj.append(noun_adj_df)
                        if isinstance(verb_noun_df, pd.DataFrame):
                            # verb_noun_df.to_csv(get_gender_mismatch_dump_path(filename, 'verb_noun'))
                            new_df_verb_noun = new_df_verb_noun.append(verb_noun_df)
                        if isinstance(future_verb_df, pd.DataFrame):
                            new_df_future_verb = new_df_future_verb.append(future_verb_df)
                new_df_noun_num = new_df_noun_num.drop_duplicates()
                new_df_noun_adj = new_df_noun_adj.drop_duplicates()
                new_df_verb_noun = new_df_verb_noun.drop_duplicates()
                new_df_future_verb = new_df_future_verb.drop_duplicates()
                new_df_noun_num.to_csv(get_gender_mismatch_dump_path(filename, 'noun_num', year, chunk_num))
                new_df_noun_adj.to_csv(get_gender_mismatch_dump_path(filename, 'noun_adj', year, chunk_num))
                new_df_verb_noun.to_csv(get_gender_mismatch_dump_path(filename, 'verb_noun', year, chunk_num))
                new_df_future_verb.to_csv(get_future_verb_dump_path(filename, year, chunk_num))
                dump_track_df = dump_track_df.append({'visited': filename, 'chunk_num': chunk_num}, ignore_index=True)
                dump_track_df.to_csv(TEMP_PATH, columns=['visited', 'chunk_num'])

create_dump_track_file()
create_csv_dumps_gender_mismatch_per_year_multiple_sentences()