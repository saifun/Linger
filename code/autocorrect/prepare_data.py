import pandas
import random


def build_finetuning_input():
    df = pandas.DataFrame(columns=['original', 'mistaken'])
    with open('./Hebrew_Wikipedia_Corpus.txt', 'r') as hebrew_wikipedia_corpus:
        # print(len(hebrew_wikipedia_corpus.readlines()))
        for sentence in hebrew_wikipedia_corpus.readlines():
            print('*******************')
            print(sentence.strip())
            print(insert_mistakes_to_sentence(sentence.strip()))
            df = df.append({
                'original': sentence,
                'mistaken': insert_mistakes_to_sentence(sentence)
            }, ignore_index=True)
    df.to_csv('./hebrew_corpus.csv')


def insert_mistakes_to_sentence(sentence):
    words = sentence.split()
    words_indices = [index for index in range(len(words)) if len(words[index]) > 1]
    chosen_word_index = random.choice(words_indices)
    words[chosen_word_index] = _add_mistake_to_word(words[chosen_word_index])
    return ' '.join(words)


def _add_mistake_to_word(word):
    mistake_type_operations = [_add_letter, _remove_letter, _replace_letter, _swap_letters]
    function_to_apply = random.choice(mistake_type_operations)
    list_word = function_to_apply(list(word))
    return ''.join(list_word)


def _add_letter(word):
    index_too_add = random.randint(0, len(word))
    word.insert(index_too_add, _get_random_hebrew_letter())
    return word


def _remove_letter(word):
    index_to_remove = _get_random_index_within_word(word[1:])
    word.pop(index_to_remove)
    return word


def _replace_letter(word):
    index_to_swap = _get_random_index_within_word(word)
    word[index_to_swap] = _get_random_hebrew_letter()
    return word


def _swap_letters(word):
    index_to_swap = _get_random_index_within_word(word[:-1])
    word[index_to_swap], word[index_to_swap + 1] = word[index_to_swap + 1], word[index_to_swap]
    return word


def _get_random_index_within_word(word):
    return random.randint(0, len(word) - 1)


def _get_random_hebrew_letter():
    return chr(ord('א') + random.randint(0, 26))


if __name__ == '__main__':
    build_finetuning_input()
