from utilities import generate_sentences, open_csv_files_from_path, generate_sentences_for_single_day
from consts import Info, NUM_POS, NOUN_POS, ADJ_POS, VERB_POS, GENDERS, SUBJECT_DEPREL, YEARS, PATHS, MONTHS, SUBFILES_PATH
from semantic_tree import SemanticTree
from collections import defaultdict
import pandas as pd


# def get_parse_tree(text, tree, pos, features, deprel):
#     word_list = list(zip(list(text), map(lambda head: head - 1, list(tree)), list(pos),
#                          map(lambda feature: get_gender(feature), list(features)), list(deprel)))
#     tree = {index: Info(word, head, pos, gender, deprel) for index, (word, head, pos, gender, deprel) in enumerate(word_list)}
#     return tree
#
#
# def get_gender(feature):
#     if feature:
#         all_features = feature.split('|')
#         for f in all_features:
#             split_f = f.split('=')
#             if split_f[0] == 'Gender':
#                 return split_f[1]
#     return None


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


def find_gender_mismatches_for_sentence(sent_parse_tree, head_pos, target_pos):
    head_indices = [index for index in sent_parse_tree if sent_parse_tree[index].pos == head_pos\
                    and sent_parse_tree[index].gender in GENDERS]
    gender_mismatches = defaultdict(list)
    for head_idx in head_indices:
        mismatches = find_mismatch_for_head(target_pos, head_idx, sent_parse_tree)
        if mismatches:
            gender_mismatches[sent_parse_tree[head_idx]].extend(find_mismatch_for_head(target_pos, head_idx, sent_parse_tree))
    return gender_mismatches


def print_gender_mismatches_per_sentence(sentence, gender_mismatch_dict):
    for head_word in gender_mismatch_dict:
        if gender_mismatch_dict[head_word]:
            print(sentence)
            print(f'{head_word.word} ({head_word.gender}): {[(word.word, word.gender) for word in gender_mismatch_dict[head_word]]}\n')


def find_num_gender_mismatch_for_sentence(parse_tree, head_pos, target_pos):
    gender_mismatch_dict_noun_num = find_gender_mismatches_for_sentence(parse_tree, head_pos, target_pos)
    num_mismatches = sum([len(gender_mismatch_dict_noun_num[head_word]) for head_word in gender_mismatch_dict_noun_num])
    return num_mismatches


def create_df_gender_mismatch_for_sentence(sent, parse_tree, head_pos, target_pos, month, year):
    gender_mismatch_dict_noun_num = find_gender_mismatches_for_sentence(parse_tree, head_pos, target_pos)
    if gender_mismatch_dict_noun_num:
        new_df = pd.DataFrame([], columns=['sentence', 'month', 'year', 'head', 'head_gender', 'mismatch'])
        for head_word in gender_mismatch_dict_noun_num:
            new_df['mismatch'] = [word.word for word in gender_mismatch_dict_noun_num[head_word]]
            new_df['head'] = len(gender_mismatch_dict_noun_num[head_word]) * [head_word.word]
            new_df['head_gender'] = len(gender_mismatch_dict_noun_num[head_word]) * [head_word.gender]
        new_df['year'] = year
        new_df['month'] = month
        new_df['sentence'] = sent
        return new_df
    return None


def create_df_gender_mismatch_for_sentence_noun_num(sent, parse_tree, month, year):
    return create_df_gender_mismatch_for_sentence(sent, parse_tree, NOUN_POS, NUM_POS, month, year)


def create_df_gender_mismatch_for_sentence_noun_adj(sent, parse_tree, month, year):
    return create_df_gender_mismatch_for_sentence(sent, parse_tree, NOUN_POS, ADJ_POS, month, year)


def create_df_gender_mismatch_for_sentence_verb_noun(sent, parse_tree, month, year):
    return create_df_gender_mismatch_for_sentence(sent, parse_tree, VERB_POS, NOUN_POS, month, year)


def find_num_gender_mismatch_for_sentence_noun_num(parse_tree):
    return find_num_gender_mismatch_for_sentence(parse_tree, NOUN_POS, NUM_POS)


def find_num_gender_mismatch_for_sentence_noun_adj(parse_tree):
    return find_num_gender_mismatch_for_sentence(parse_tree, NOUN_POS, ADJ_POS)


def find_num_gender_mismatch_for_sentence_verb_noun(parse_tree):
    return find_num_gender_mismatch_for_sentence(parse_tree, VERB_POS, NOUN_POS)


def write_data_to_csv(mismatch_name, mismatch_dict):
    mismatch_df = pd.DataFrame(list(mismatch_dict.items()), columns=['month', 'count'])
    mismatch_df.to_csv(f'results/gender_mismatch/{mismatch_name}_count.csv', index=False, header=True)


def create_df_num_gender_mismatch_per_year():
    mismatches_noun_num_per_month = {f'{year}-{month}': 0 for year in YEARS for month in MONTHS}
    mismatches_noun_adj_per_month = {f'{year}-{month}': 0 for year in YEARS for month in MONTHS}
    mismatches_verb_noun_per_month = {f'{year}-{month}': 0 for year in YEARS for month in MONTHS}
    for year in YEARS:
        for sentence, month, semantic_tree in generate_sentences(PATHS[year]):
            semantic_tree.parse_text()
            parse_tree = semantic_tree.tree
            mismatches_noun_num_per_month[f'{year}-{month}'] += find_num_gender_mismatch_for_sentence_noun_num(parse_tree)
            mismatches_noun_adj_per_month[f'{year}-{month}'] += find_num_gender_mismatch_for_sentence_noun_adj(parse_tree)
            mismatches_verb_noun_per_month[f'{year}-{month}'] += find_num_gender_mismatch_for_sentence_verb_noun(parse_tree)
    write_data_to_csv("noun_num", mismatches_noun_num_per_month)
    write_data_to_csv("noun_adj", mismatches_noun_adj_per_month)
    write_data_to_csv("verb_noun", mismatches_verb_noun_per_month)


def create_df_num_gender_mismatch_per_year_multiple_sentences():
    mismatches_noun_num_per_month = {f'{year}-{month}': 0 for year in YEARS for month in MONTHS}
    mismatches_noun_adj_per_month = {f'{year}-{month}': 0 for year in YEARS for month in MONTHS}
    mismatches_verb_noun_per_month = {f'{year}-{month}': 0 for year in YEARS for month in MONTHS}
    for year in YEARS:
        for stanza_analysis_list, month, in generate_sentences_for_single_day(PATHS[year]):
            for sent_df in stanza_analysis_list:
                semantic_tree = SemanticTree(sent_df['text'])
                semantic_tree.parse_text_without_processing(sent_df['text'], sent_df['head'], sent_df['upos'], sent_df['feats'], sent_df['deprel'])
                parse_tree = semantic_tree.tree
                mismatches_noun_num_per_month[f'{year}-{month}'] += find_num_gender_mismatch_for_sentence_noun_num(parse_tree)
                mismatches_noun_adj_per_month[f'{year}-{month}'] += find_num_gender_mismatch_for_sentence_noun_adj(parse_tree)
                mismatches_verb_noun_per_month[f'{year}-{month}'] += find_num_gender_mismatch_for_sentence_verb_noun(parse_tree)
    write_data_to_csv("noun_num", mismatches_noun_num_per_month)
    write_data_to_csv("noun_adj", mismatches_noun_adj_per_month)
    write_data_to_csv("verb_noun", mismatches_verb_noun_per_month)


def get_gender_mismatch_dump_path(filename, mismatch_name):
    split_filename = filename.split('/')
    dir_path = '/'.join(split_filename[:-1]).replace('subfiles_twitter_data', 'gender_mismatch') + '/'
    end_filename = '_'.join(split_filename[-1].split('_')[2:])
    return dir_path + 'gender_mismatch_dump_' + mismatch_name + '_' + end_filename


def create_csv_dumps_gender_mismatch_per_year_multiple_sentences():
    for year in YEARS:
        for stanza_analysis_list, month, filename in generate_sentences_for_single_day(SUBFILES_PATH[year]):
            if stanza_analysis_list:
                dump_track_df = pd.read_csv('temp/dump_track.csv')
                for sent_df in stanza_analysis_list:
                    semantic_tree = SemanticTree(sent_df['text'])
                    semantic_tree.parse_text_without_processing(sent_df['text'], sent_df['head'], sent_df['upos'], sent_df['feats'], sent_df['deprel'])
                    parse_tree = semantic_tree.tree
                    if not sent_df.empty:
                        sentence_text = sent_df['sentence'][1]
                        noun_num_df = create_df_gender_mismatch_for_sentence_noun_num(sentence_text, parse_tree, month, year)
                        noun_adj_df = create_df_gender_mismatch_for_sentence_noun_adj(sentence_text, parse_tree, month, year)
                        verb_noun_df = create_df_gender_mismatch_for_sentence_verb_noun(sentence_text, parse_tree, month, year)
                        if isinstance(noun_num_df, pd.DataFrame):
                            noun_num_df.to_csv(get_gender_mismatch_dump_path(filename, 'noun_num'))
                        if isinstance(noun_adj_df, pd.DataFrame):
                            noun_adj_df.to_csv(get_gender_mismatch_dump_path(filename, 'noun_adj'))
                        if isinstance(verb_noun_df, pd.DataFrame):
                            verb_noun_df.to_csv(get_gender_mismatch_dump_path(filename, 'verb_noun'))
                dump_track_df = dump_track_df.append({'visited': filename}, ignore_index=True)
                dump_track_df['visited'].to_csv('temp/dump_track.csv')


create_csv_dumps_gender_mismatch_per_year_multiple_sentences()
# def create_csv_gender_mismatch_per_file(filename, path, year):
#     mismatches_noun_num_df = pd.DataFrame([], columns=['month', 'year', 'head', 'head_gender', 'mismatch'])
#     mismatches_noun_adj_df = pd.DataFrame([], columns=['month', 'year', 'head', 'head_gender', 'mismatch'])
#     mismatches_verb_noun_df = pd.DataFrame([], columns=['month', 'year', 'head', 'head_gender', 'mismatch'])
#     for stanza_analysis_list, month, in generate_sentences_for_single_day(path):


def find_gender_mismatch_sentences(path):
    for sentence, semantic_tree in generate_sentences(path):
        semantic_tree.parse_text()
        parse_tree = semantic_tree.tree
        gender_mismatch_dict_noun_num = find_gender_mismatches_for_sentence(parse_tree, NOUN_POS, NUM_POS)
        gender_mismatch_dict_noun_adj = find_gender_mismatches_for_sentence(parse_tree, NOUN_POS, ADJ_POS)
        gender_mismatch_dict_verb_noun = find_gender_mismatches_for_sentence(parse_tree, VERB_POS, NOUN_POS)
        print_gender_mismatches_per_sentence(sentence, gender_mismatch_dict_noun_num)
        print_gender_mismatches_per_sentence(sentence, gender_mismatch_dict_noun_adj)
        print_gender_mismatches_per_sentence(sentence, gender_mismatch_dict_verb_noun)
