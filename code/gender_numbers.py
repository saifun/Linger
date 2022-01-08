from utilities import generate_sentences
from consts import Info, NUM_POS, NOUN_POS, ADJ_POS, GENDERS
from collections import defaultdict


def get_parse_tree(text, tree, pos, features):
    word_list = list(zip(list(text), map(lambda head: head - 1, list(tree)), list(pos),
                         map(lambda feature: get_gender(feature), list(features))))
    tree = {index: Info(word, head, pos, gender) for index, (word, head, pos, gender) in enumerate(word_list)}
    return tree


def get_gender(feature):
    if feature:
        all_features = feature.split('|')
        for f in all_features:
            split_f = f.split('=')
            if split_f[0] == 'Gender':
                return split_f[1]
    return None


def find_mismatch_for_noun(target_pos, noun_idx, sent_parse_tree):
    curr_noun = sent_parse_tree[noun_idx]
    target_pos_mismatches = []
    for index in sent_parse_tree:
        if sent_parse_tree[index].pos == target_pos and sent_parse_tree[index].head == noun_idx:
            if sent_parse_tree[index].gender in GENDERS\
                    and curr_noun.gender != sent_parse_tree[index].gender:
                target_pos_mismatches.append(sent_parse_tree[index])
    return target_pos_mismatches

def find_gender_mismatches_for_sentence(sent_parse_tree):
    noun_indices = [index for index in sent_parse_tree if sent_parse_tree[index].pos == NOUN_POS\
                    and sent_parse_tree[index].gender in GENDERS]
    gender_mismatches = defaultdict(list)
    for noun_idx in noun_indices:
        gender_mismatches[sent_parse_tree[noun_idx]].extend(find_mismatch_for_noun(NUM_POS, noun_idx, sent_parse_tree))
        gender_mismatches[sent_parse_tree[noun_idx]].extend(find_mismatch_for_noun(ADJ_POS, noun_idx, sent_parse_tree))
    return gender_mismatches

def print_gender_mismatches_per_sentence(sentence, gender_mismatch_dict):
    for noun in gender_mismatch_dict:
        if gender_mismatch_dict[noun]:
            print(sentence)
            print(f'{noun.word} ({noun.gender}): {[(word.word, word.gender) for word in gender_mismatch_dict[noun]]}\n')


def find_gender_mismatch_sentences(path):
    for sentence, stanza_analysis in generate_sentences(path):
        text, tree, pos, features = stanza_analysis
        parse_tree = get_parse_tree(text, tree, pos, features)
        gender_mismatch_dict = find_gender_mismatches_for_sentence(parse_tree)
        print_gender_mismatches_per_sentence(sentence, gender_mismatch_dict)


find_gender_mismatch_sentences("./test_files")

