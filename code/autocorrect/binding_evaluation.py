from alephbert_finetune import HebrewMistakesDataset
from alephbert_binding import AlephBertPredictor

MISTAKE = 'mistaken'
CORRECT = 'correct'
MISTAKE_INDEX = 'mistake_index'


def evaluate(dataset_file, number_of_samples=101):
    dataset = HebrewMistakesDataset(dataset_file=dataset_file, column_names=['correct', 'mistaken', 'mistake_index'])
    alephbert = AlephBertPredictor()
    test_sentences = format_test_set(dataset.dataset[:number_of_samples])
    was_in_three_best = 0
    exact_match = 0
    harmonic_score = 0
    for index, sentence in enumerate(test_sentences):
        print(f'Evaluating {index}')
        suggestions = alephbert.get_autocorrect_suggestions(sentence[MISTAKE], sentence[MISTAKE_INDEX])
        three_best_suggestions = [res.value for res in suggestions[:3]]
        correct_word = sentence[CORRECT].split()[sentence[MISTAKE_INDEX]]
        current_harmonic_score = calculate_harmonic_score(suggestions, correct_word)
        print(f'Harmonic score: {current_harmonic_score}')
        harmonic_score += current_harmonic_score
        if correct_word in three_best_suggestions:
            was_in_three_best += 1
        if suggestions[0].value == correct_word:
            exact_match += 1
        else:
            print(sentence[CORRECT])
            print(sentence[MISTAKE])
            print(three_best_suggestions)
    print(f'Exact match: {exact_match}/{test_sentences}')
    print(f'Was in best three: {was_in_three_best}/{test_sentences}')
    print(f'The harmonic score was {harmonic_score}/{test_sentences}')


def format_test_set_with_no_indices(test_data):
    final_data = []
    for index, sample in test_data.iterrows():
        correct_sentence = sample[CORRECT]
        mistaken_sentence = sample[MISTAKE]
        different_index = \
            [index for index, word in enumerate(correct_sentence.split()) if word != mistaken_sentence.split()[index]][
                0]
        final_data.append({
            MISTAKE: mistaken_sentence,
            CORRECT: correct_sentence,
            MISTAKE_INDEX: different_index
        })
    return final_data


def format_test_set(test_data):
    final_data = [{
        MISTAKE: sample[MISTAKE],
        CORRECT: sample[CORRECT],
        MISTAKE_INDEX: sample[MISTAKE_INDEX]
    } for index, sample in test_data.iterrows()]
    return final_data


def calculate_harmonic_score(suggestions, correct_word):
    suggested_words = [res.value for res in suggestions]
    try:
        word_index = suggested_words.index(correct_word)
        return 1 / (word_index + 1)
    except ValueError:
        return 0


if __name__ == '__main__':
    evaluate(dataset_file='./dataset/evaluation_hebrew_corpus.csv')
    # evaluate(dataset_file='./dataset/hebrew_corpus.csv', number_of_samples=2)