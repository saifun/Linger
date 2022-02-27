import torch
from torch.nn.functional import softmax
from transformers import AutoTokenizer, BertForMaskedLM
from dataclasses import dataclass
from fuzzywuzzy import fuzz


@dataclass
class SuggestedToken:
    '''Class for keeping tokens'''
    value: str
    probability: float
    rank: int

    def calculate_fuzziness_for_token(self, text):
        return fuzz.ratio(self.value, text)

    def get_suggested_token_score(self, text):
        fuzziness_score = self.calculate_fuzziness_for_token(text)
        if fuzziness_score == 100:
            return 100
        return 0.7 * (fuzziness_score / 100) + (0.3 * self.probability)


class AlephBertPredictor:
    def __init__(self):
        self.alephbert_tokenizer = AutoTokenizer.from_pretrained('onlplab/alephbert-base')
        self.alephbert = BertForMaskedLM.from_pretrained('onlplab/alephbert-base')

    def get_autocorrect_suggestions(self, text: str, token_index_to_correct: int, get_whole_sentence: bool = False):
        corrected_index, corrected_sentence, model_output_tensors = \
            self.get_whole_sentence_word_correction(text, token_index_to_correct)
        if get_whole_sentence:
            return corrected_sentence
        else:
            masked_token = text.split()[token_index_to_correct]
            suggested_tokens = self._get_suggestions_by_probability(corrected_index, model_output_tensors)
            return sorted(suggested_tokens, key=lambda token: token.get_suggested_token_score(masked_token),
                          reverse=True)

    def get_whole_sentence_word_correction(self, text, index):
        suggested_sentence = []
        model_output_tensors = self._get_output_tensor_for_masked_sentence(text, index)
        for single_word_output_tensors in model_output_tensors.logits[0][1:-1]:
            suggestion = self._get_single_token_suggestion(single_word_output_tensors)
            suggested_sentence.append(suggestion)
        correct_index = self._adjust_word_index(suggested_sentence, index)
        corrected_sentence = self._build_output_sentence(suggested_sentence)
        return correct_index, corrected_sentence, model_output_tensors

    def _get_suggestions_by_probability(self, corrected_output_index, model_output_tensors):
        return self._get_masked_token_suggestions(model_output_tensors[0], corrected_output_index, 100)

    def _get_output_tensor_for_masked_sentence(self, text, token_index_to_mask):
        masked_text = self._mask_text_at_index(text, token_index_to_mask)
        tokenized_text = self.alephbert_tokenizer.encode(masked_text)
        return self.alephbert(torch.tensor([tokenized_text]))

    def _get_masked_token_suggestions(self, model_output_tensors, token_index_to_mask, amount_of_suggestions):
        masked_token_tensor = model_output_tensors[0][token_index_to_mask + 1]
        probabilities = softmax(masked_token_tensor, dim=0)
        tensor_by_order = sorted(enumerate(probabilities), key=lambda tensor_values: tensor_values[1], reverse=True)
        top_ten_indices = [index for index, prob in tensor_by_order[:amount_of_suggestions]]
        highest_probs = [float(prob) for index, prob in tensor_by_order[:amount_of_suggestions]]
        top_ten_suggestions = self.alephbert_tokenizer.convert_ids_to_tokens(top_ten_indices)
        return [SuggestedToken(top_ten_suggestions[index], highest_probs[index], index) for index in
                range(amount_of_suggestions)]

    def _get_single_token_suggestion(self, word_output_tensors):
        probabilities = softmax(word_output_tensors, dim=0)
        highest_prob_token = max(enumerate(probabilities), key=lambda tensor_values: tensor_values[1])
        top_suggestion = self.alephbert_tokenizer.convert_ids_to_tokens(highest_prob_token[0])
        return top_suggestion

    def _get_all_tokens_argmax(self, model_output_tensors):
        return model_output_tensors[0].argmax(dim=1)

    def _mask_text_at_index(self, text, index):
        split_text = text.split()
        split_text[index] = self.alephbert_tokenizer.mask_token
        return ' '.join(split_text)

    def _build_output_sentence(self, word_list):
        sentence = ''
        for word in word_list:
            if word.startswith('##'):
                sentence = sentence + word[2:]
            else:
                sentence = sentence + ' ' + word
        return sentence

    def _adjust_word_index(self, suggested_sentence, word_index):
        correct_index = word_index
        for index, word in enumerate(suggested_sentence[:word_index + 1]):
            if word.startswith('##'):
                correct_index += 1
        return correct_index


if __name__ == '__main__':
    alephbert = AlephBertPredictor()
    text = "אני רוצה לבכות שוקולד"
    text2 = "ישנו דיון לגבי המשמעות הפילוסופית שמאחורי הןרט"
    text3 = "אשאיר לאחרים שהשתתפו בדיון הקצר להסכים נלוסח"
    print(alephbert.get_autocorrect_suggestions(text2, 6))
