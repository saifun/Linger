from alephbert_finetune import HebrewMistakesDataset
from alephbert_binding import AlephBertPredictor

MISTAKE = 'mistaken'
CORRECT = 'correct'
MISTAKE_INDEX = 'mistake_index'


def evaluate():
    dataset = HebrewMistakesDataset(dataset_file='./dataset/evaluation_hebrew_corpus.csv',
                                    column_names=['correct', 'mistaken', 'mistake_index'])
    alephbert = AlephBertPredictor()
    test_sentences = format_test_set(dataset.dataset)
    exact_match = 0
    for index, sentence in enumerate(test_sentences):
        print(f'Evaluating {index}')
        three_best = [res.value for res in
                      alephbert.get_autocorrect_suggestions(sentence[MISTAKE], sentence[MISTAKE_INDEX])[:3]]
        if sentence[CORRECT].split()[sentence[MISTAKE_INDEX]] in three_best:
            exact_match += 1
        else:
            print(sentence[CORRECT])
            print(sentence[MISTAKE])
            print(three_best)
    print(f'Exact match: {exact_match}/{len(dataset.dataset)}')


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


if __name__ == '__main__':
    evaluate()
