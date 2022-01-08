from utilities import generate_sentences
from consts import Info, NUM_POS, NOUN_POS, ADJ_POS, VERB_POS, GENDERS
from collections import defaultdict


def get_parse_tree(text, tree, pos, features, deprel):
    word_list = list(zip(list(text), map(lambda head: head - 1, list(tree)), list(pos),
                         map(lambda feature: get_gender(feature), list(features)), list(deprel)))
    tree = {index: Info(word, head, pos, gender, deprel) for index, (word, head, pos, gender, deprel) in enumerate(word_list)}
    return tree


def get_gender(feature):
    if feature:
        all_features = feature.split('|')
        for f in all_features:
            split_f = f.split('=')
            if split_f[0] == 'Gender':
                return split_f[1]
    return None


def is_verb_head_word_and_target_is_not_a_subject(head_pos, target_deprel):
    return head_pos == VERB_POS and target_deprel != 'nsubj'


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
        gender_mismatches[sent_parse_tree[head_idx]].extend(find_mismatch_for_head(target_pos, head_idx, sent_parse_tree))
    return gender_mismatches


def print_gender_mismatches_per_sentence(sentence, gender_mismatch_dict):
    for head_word in gender_mismatch_dict:
        if gender_mismatch_dict[head_word]:
            print(sentence)
            print(f'{head_word.word} ({head_word.gender}): {[(word.word, word.gender) for word in gender_mismatch_dict[head_word]]}\n')


def find_gender_mismatch_sentences(path):
    for sentence, stanza_analysis in generate_sentences(path):
        text, tree, pos, features, deprel = stanza_analysis
        parse_tree = get_parse_tree(text, tree, pos, features, deprel)
        gender_mismatch_dict_noun_num = find_gender_mismatches_for_sentence(parse_tree, NOUN_POS, NUM_POS)
        gender_mismatch_dict_noun_adj = find_gender_mismatches_for_sentence(parse_tree, NOUN_POS, ADJ_POS)
        gender_mismatch_dict_verb_noun = find_gender_mismatches_for_sentence(parse_tree, VERB_POS, NOUN_POS)
        print_gender_mismatches_per_sentence(sentence, gender_mismatch_dict_noun_num)
        print_gender_mismatches_per_sentence(sentence, gender_mismatch_dict_noun_adj)
        print_gender_mismatches_per_sentence(sentence, gender_mismatch_dict_verb_noun)


find_gender_mismatch_sentences("./test_files")

