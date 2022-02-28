from alephbert_finetune import HebrewMistakesDataset
from alephbert_binding import AlephBertPredictor

MISTAKE = 'mistaken'
MISTAKE_INDEX = 'mistake_index'


def run_autocorrect(dataset_file, file_columns=None):
    dataset = HebrewMistakesDataset(dataset_file=dataset_file, column_names=file_columns)
    alephbert = AlephBertPredictor()
    test_sentences = format_test_set(dataset.dataset)
    print(f'Got {len(test_sentences)} test sentences')
    for index, sentence in enumerate(test_sentences):
        print(f'Evaluating {index}')
        print(sentence[MISTAKE], sentence[MISTAKE_INDEX])
        suggestions = alephbert.get_autocorrect_suggestions(sentence[MISTAKE], sentence[MISTAKE_INDEX])
        three_best_suggestions = [res.value for res in suggestions[:10]]
        print(f'The original sentence was {sentence[MISTAKE]}')
        print(f'The model suggested fixing the word {sentence[MISTAKE].split()[sentence[MISTAKE_INDEX]]} to one of these seggestions: {three_best_suggestions}')


def format_test_set(test_data):
    final_data = [{
        MISTAKE: sample[MISTAKE],
        MISTAKE_INDEX: int(sample[MISTAKE_INDEX])
    } for index, sample in test_data.iterrows()]
    return final_data


if __name__ == '__main__':
    run_autocorrect(dataset_file='./dataset/twitter_check.csv', file_columns=[MISTAKE, MISTAKE_INDEX])
