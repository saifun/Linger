import math
import numpy
import torch
import pandas as pd
from torch.nn.functional import softmax
from transformers import AutoTokenizer, AutoModelForCausalLM, Trainer, TrainingArguments, AutoConfig

ALEPH_BERT_BASE = 'onlplab/alephbert-base'
CHECKPOINT = './writing-mistakes-finetuned-alephbert/checkpoint-3'

class MistakeAlephBert:
    def __init__(self, dataset):
        self.model = AutoModelForCausalLM.from_pretrained(CHECKPOINT)
        self.dataset = dataset
        self.training_args = TrainingArguments(
            f"writing-mistakes-finetuned-alephbert",
            evaluation_strategy="epoch",
            learning_rate=2e-5,
            weight_decay=0.01,
            save_strategy='epoch'
        )

    def train(self):
        print('************ STARTED TRAINING ************')
        self.trainer = Trainer(
            model=self.model,
            args=self.training_args,
            train_dataset=self.dataset.tokenized_train,
            eval_dataset=self.dataset.tokenized_validation
        )
        self.trainer.train()

    def eval(self):
        eval_results = self.trainer.evaluate()
        print(f"Perplexity: {math.exp(eval_results['eval_loss']):.2f}")

class MistakeAlephBertFromPretrained:
    def __init__(self):
        self.model = AutoModelForCausalLM.from_pretrained(CHECKPOINT)
        self.tokenizer = AutoTokenizer.from_pretrained(CHECKPOINT)
        self.trainer = Trainer(model=self.model)

    def predict(self):
        tokenized_text = self.tokenizer('אני חוצה לאכול')
        print(tokenized_text)
        output = self.trainer.predict([tokenized_text])
        print(dir(output))
        print(torch.tensor(output.predictions))
        # print(self.tokenizer.decode(output.predictions))
        print(self._get_masked_token_suggestions(torch.tensor(output.predictions), 1, 5))

    def _get_masked_token_suggestions(self, model_output_tensors, token_index_to_mask, amount_of_suggestions):
        masked_token_tensor = model_output_tensors[0][token_index_to_mask + 1]
        probabilities = softmax(masked_token_tensor, dim=0)
        tensor_by_order = sorted(enumerate(probabilities), key=lambda tensor_values: tensor_values[1], reverse=True)
        top_ten_indices = [index for index, prob in tensor_by_order[:amount_of_suggestions]]
        highest_probs = [float(prob) for index, prob in tensor_by_order[:amount_of_suggestions]]
        top_ten_suggestions = self.tokenizer.convert_ids_to_tokens(top_ten_indices)
        print([(top_ten_suggestions[index], highest_probs[index], index) for index in range(amount_of_suggestions)])


class HebrewMistakesDataset:
    def __init__(self):
        self.alephbert_tokenizer = AutoTokenizer.from_pretrained(CHECKPOINT)
        self.dataset = self.load_data()

    @property
    def number_of_samples(self):
        return len(self.dataset)

    def tokenize_dataset(self, dataset):
        tokenized_input = [self.alephbert_tokenizer(sample, padding='max_length') for sample in dataset["mistaken"]]
        labels = [self.alephbert_tokenizer(sample, padding='max_length') for sample in dataset["correct"]]
        tokenized_dataset = [{'label': labels[index]['input_ids'],
                              **input_tokenization}
                             for index, input_tokenization in enumerate(tokenized_input)]
        return tokenized_dataset

    def load_data(self):
        df = pd.read_csv("./Hebrew_corpus.csv", header=0, names=['correct', 'mistaken'])
        df = df.head(10)
        print('Number of training sentences: {:,}\n'.format(df.shape[0]))
        return df

    def split_dataset(self):
        self.train, self.validation, self.test = numpy.split(self.dataset.sample(frac=1, random_state=42),
                                                             [int(0.6 * self.number_of_samples),
                                                              int(0.8 * self.number_of_samples)])

    def prepare(self):
        self.split_dataset()
        self.tokenized_train = self.tokenize_dataset(self.train)
        self.tokenized_validation = self.tokenize_dataset(self.validation)
        self.tokenized_test = self.tokenize_dataset(self.test)
        print('Train:', len(self.train))
        print('Validation:', len(self.validation))
        print('Test:', len(self.test))


if __name__ == '__main__':
    # dataset = HebrewMistakesDataset()
    # dataset.prepare()
    # alephbert = MistakeAlephBert(dataset)
    # alephbert.train()
    # alephbert.eval()
    # alephbert_tokenizer = AutoTokenizer.from_pretrained(ALEPH_BERT_BASE)
    # print(alephbert_tokenizer.decode([10664]))
    model = MistakeAlephBertFromPretrained()
    model.predict()
