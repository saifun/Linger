from alephbert_finetune import HebrewMistakesDataset
import torch
from torch.nn.functional import softmax
from transformers import AutoTokenizer, AutoModelForCausalLM, Trainer, TrainingArguments, AutoConfig

CHECKPOINT = './writing-mistakes-finetuned-alephbert/checkpoint-1100'


def evaluate():
    dataset = HebrewMistakesDataset(dataset_file='./dataset/hebrew_corpus.csv')
    dataset.prepare()
    model = MistakeAlephBertFromPretrained()
    prediction = model.predict(dataset.tokenized_test[:5])
    formatted_sentences = [model.format(pred) for pred in prediction]
    exact_match = 0
    for index, sentence in enumerate(formatted_sentences):
        original = dataset.test['mistaken'].iloc[index]
        print(f'Original: {original} Output: {sentence}')
        if original == sentence:
            print('YAYYYYYYYYYYYYYYY')
            exact_match += 1
    print(f'Exact match: {exact_match}')


class MistakeAlephBertFromPretrained:
    def __init__(self):
        self.model = AutoModelForCausalLM.from_pretrained(CHECKPOINT)
        self.tokenizer = AutoTokenizer.from_pretrained(CHECKPOINT)
        self.trainer = Trainer(model=self.model)

    def predict(self, tokenized_text):
        token_ids = [[input_id for input_id in text['input_ids']] for text in tokenized_text]
        output = self.model(torch.tensor(token_ids))
        sentence_prediction = []
        for sentence_output in output.logits:
            per_word_prediction = []
            for index in range(len(sentence_output)):
                suggestion = self._get_masked_token_suggestion(sentence_output, index)
                if suggestion[0].startswith('['):
                    break
                per_word_prediction.append(suggestion)
            sentence_prediction.append(per_word_prediction)
        return sentence_prediction

    def _get_masked_token_suggestion(self, model_output_tensors, token_index_to_mask):
        print('Unmasking', token_index_to_mask)
        masked_token_tensor = model_output_tensors[token_index_to_mask]
        probabilities = softmax(masked_token_tensor, dim=0)
        highest_prob = max(enumerate(probabilities), key=lambda tensor_values: tensor_values[1])
        top_suggestion = self.tokenizer.convert_ids_to_tokens(highest_prob[0])
        print(top_suggestion, highest_prob)
        return (top_suggestion, highest_prob)

    def format(self, prediction):
        clean_prediction = [word_output[0] for word_output in prediction if not word_output[0].startswith('[')]
        return ' '.join(clean_prediction)


if __name__ == '__main__':
    evaluate()
