import torch
from torch.nn.functional import softmax
from transformers import AutoTokenizer, BertForMaskedLM
from dataclasses import dataclass


@dataclass
class Token:
    '''Class for keeping tokens'''
    value: str
    probability: float


class AlephBertPredictor:
    def __init__(self):
        self.alephbert_tokenizer = AutoTokenizer.from_pretrained('onlplab/alephbert-base')
        self.alephbert = BertForMaskedLM.from_pretrained('onlplab/alephbert-base')

    def evaluate(self, text, token_index_to_mask):
        masked_text = self._mask_text_at_index(text, token_index_to_mask)
        tokenized_text = self.alephbert_tokenizer.encode(masked_text)
        model_output_tensors = self.alephbert(torch.tensor([tokenized_text]))[0]
        return self._get_masked_token_suggestions(model_output_tensors, token_index_to_mask, 10)

    def _get_masked_token_suggestions(self, model_output_tensors, token_index_to_mask, amount_of_suggestions):
        masked_token_tensor = model_output_tensors[0][token_index_to_mask + 1]
        probabilities = softmax(masked_token_tensor, dim=0)
        tensor_by_order = sorted(enumerate(probabilities), key=lambda tensor_values: tensor_values[1], reverse=True)
        top_ten_indices = [index for index, prob in tensor_by_order[:amount_of_suggestions]]
        highest_probs = [float(prob) for index, prob in tensor_by_order[:amount_of_suggestions]]
        top_ten_suggestions = self.alephbert_tokenizer.convert_ids_to_tokens(top_ten_indices)
        return [Token(top_ten_suggestions[index], highest_probs[index]) for index in range(amount_of_suggestions)]

    def _get_all_tokens_argmax(self, model_output_tensors):
        return model_output_tensors[0].argmax(dim=1)

    def _mask_text_at_index(self, text, index):
        split_text = text.split()
        split_text[index] = self.alephbert_tokenizer.mask_token
        return ' '.join(split_text)


if __name__ == '__main__':
    alephbert = AlephBertPredictor()
    text = "אני רוצה לאכול מרק"
    suggested_tokens = alephbert.evaluate(text, 2)
    print(suggested_tokens)