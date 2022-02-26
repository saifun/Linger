from utilities import generate_sentences, open_csv_files_from_path, generate_sentences_for_single_day, create_dump_track_file
from consts import Info, NUM_POS, NOUN_POS, ADJ_POS, VERB_POS, GENDERS, SUBJECT_DEPREL, YEARS, PATHS, MONTHS, SUBFILES_PATH, TEMP_PATH, PRONOUN_POS, FUTURE_TENSE, SINGULAR_NUMBER, THIRD_PERSON, FIRST_PERSON
from semantic_tree import SemanticTree
from collections import defaultdict
import pandas as pd


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
                            new_df_noun_num = new_df_noun_num.append(noun_num_df)
                        if isinstance(noun_adj_df, pd.DataFrame):
                            new_df_noun_adj = new_df_noun_adj.append(noun_adj_df)
                        if isinstance(verb_noun_df, pd.DataFrame):
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