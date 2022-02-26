import torch
from torch.nn.functional import softmax
from transformers import AutoTokenizer, AutoModelForCausalLM, Trainer, TrainingArguments, AutoConfig

CHECKPOINT = './writing-mistakes-finetuned-alephbert/checkpoint-1100'


class MistakeAlephBertFromPretrained:
    def __init__(self):
        self.model = AutoModelForCausalLM.from_pretrained(CHECKPOINT)
        self.tokenizer = AutoTokenizer.from_pretrained(CHECKPOINT)
        self.trainer = Trainer(model=self.model)

    def predict(self, text):
        tokenized_text = self.tokenizer(text)
        print(tokenized_text)
        output = self.trainer.predict([tokenized_text])
        per_word_prediction = [self._get_masked_token_suggestions(torch.tensor(output.predictions), index, 1) for index
                               in range(len(output.predictions[0]))]
        return per_word_prediction

    def _get_masked_token_suggestions(self, model_output_tensors, token_index_to_mask, amount_of_suggestions):
        masked_token_tensor = model_output_tensors[0][token_index_to_mask]
        probabilities = softmax(masked_token_tensor, dim=0)
        tensor_by_order = sorted(enumerate(probabilities), key=lambda tensor_values: tensor_values[1], reverse=True)
        top_indices = [index for index, prob in tensor_by_order[:amount_of_suggestions]]
        highest_probs = [float(prob) for index, prob in tensor_by_order[:amount_of_suggestions]]
        top_suggestions = self.tokenizer.convert_ids_to_tokens(top_indices)
        return [(top_suggestions[index], highest_probs[index], index) for index in range(amount_of_suggestions)]

    def format(self, prediction):
        clean_prediction = [word_output[0][0] for word_output in prediction if not word_output[0][0].startswith('[')]
        return ' '.join(clean_prediction)


def run_autocorrect_on_one_sentence(sentence):
    model = MistakeAlephBertFromPretrained()
    prediction = model.predict(sentence)
    return model.format(prediction)


if __name__ == '__main__':
    text = 'ישראל היא המדיקה החמישית בגודלה'
    run_autocorrect_on_one_sentence(text)
