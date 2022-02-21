from utilities import generate_sentences, open_csv_files_from_path, generate_sentences_for_single_day_with_light_processor, create_dump_track_file
from consts import NUM_POS, NOUN_POS,GENDERS, SUBJECT_DEPREL, YEARS, PATHS, MONTHS
from semantic_tree import SemanticTree
import pandas as pd
from gender_mismatch import create_new_gender_mismatch_df


def find_number_mismatches_for_sentence(sent_parse_tree):
    number_indices = [index for index in sent_parse_tree if sent_parse_tree[index].pos == NUM_POS\
                    and sent_parse_tree[index].gender in GENDERS]
    gender_mismatches = dict()
    for number_idx in number_indices:
        if number_idx + 1 in sent_parse_tree:
            next_word = sent_parse_tree[number_idx + 1]
            mismatch = next_word if  next_word.pos == NOUN_POS and \
                    next_word.gender in GENDERS and next_word.gender != sent_parse_tree[number_idx].gender else None
            if mismatch:
                gender_mismatches[sent_parse_tree[number_idx]] = mismatch
    return gender_mismatches


def create_df_number_mismatch_for_sentence(sent, parse_tree, month, year):
    gender_mismatch_dict_noun_num = find_number_mismatches_for_sentence(parse_tree)
    if gender_mismatch_dict_noun_num:
        new_df = create_new_gender_mismatch_df()
        for head_word in gender_mismatch_dict_noun_num:
            word = gender_mismatch_dict_noun_num[head_word]
            added_df = pd.DataFrame([[sent, month, year, head_word.word, head_word.gender, word.word]],
                                        columns=['sentence', 'month', 'year', 'head', 'head_gender', 'mismatch'])
            new_df = new_df.append(added_df, ignore_index=True)
        return new_df
    return None

def get_number_mismatch_dump_path(filename, year, number):
    split_filename = filename.split('/')
    dir_path = '/'.join(split_filename[:-1]) + '/number_mismatch_' + str(year) + '/'
    end_filename = '_'.join(split_filename[-1].split('_')[2:])
    return dir_path + 'number_mismatch_dump_chunk' + str(number) + '_' + end_filename

def create_csv_dumps_number_mismatch_per_year_multiple_sentences():
    for year in YEARS:
        for stanza_analysis_list, month, filename, chunk_num in generate_sentences_for_single_day_with_light_processor(PATHS[year]):
            new_df_noun_num = create_new_gender_mismatch_df()
            for sent_df in stanza_analysis_list:
                semantic_tree = SemanticTree(sent_df['text'])
                semantic_tree.parse_text_without_processing_less_features(sent_df['text'], sent_df['upos'], sent_df['feats'])
                parse_tree = semantic_tree.tree
                if not sent_df.empty:
                    sentence_text = sent_df['sentence'][1]
                    noun_num_df = create_df_number_mismatch_for_sentence(sentence_text, parse_tree, month, year)
                    if isinstance(noun_num_df, pd.DataFrame):
                        # noun_num_df.to_csv(get_gender_mismatch_dump_path(filename, 'noun_num'))
                        new_df_noun_num = new_df_noun_num.append(noun_num_df)
            new_df_noun_num = new_df_noun_num.drop_duplicates()
            new_df_noun_num.to_csv(get_number_mismatch_dump_path(filename, year, chunk_num))

create_csv_dumps_number_mismatch_per_year_multiple_sentences()