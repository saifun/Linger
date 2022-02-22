import math
import numpy
import pandas as pd
from logging import getLogger
from transformers import AutoTokenizer, AutoModelForCausalLM, Trainer, TrainingArguments, AutoConfig

ALEPH_BERT_BASE = 'onlplab/alephbert-base'
logger = getLogger('correctbert')


class MistakeAlephBert:
    def __init__(self, dataset):
        self.trainer = None
        self.model = AutoModelForCausalLM.from_pretrained(ALEPH_BERT_BASE)
        self.alephbert_tokenizer = AutoTokenizer.from_pretrained(ALEPH_BERT_BASE)
        self.dataset = dataset
        self.training_args = TrainingArguments(
            f"writing-mistakes-finetuned-alephbert",
            evaluation_strategy="epoch",
            learning_rate=2e-5,
            weight_decay=0.01,
            save_strategy='epoch'
        )

    def train(self):
        logger.info('************ STARTED TRAINING ************')
        self.trainer = Trainer(
            model=self.model,
            tokenizer=self.alephbert_tokenizer,
            args=self.training_args,
            train_dataset=self.dataset.tokenized_train,
            eval_dataset=self.dataset.tokenized_validation
        )
        self.trainer.train()

    def eval(self):
        eval_results = self.trainer.evaluate()
        logger.info(f"Perplexity: {math.exp(eval_results['eval_loss']):.2f}")
        logger.info(eval_results)


class HebrewMistakesDataset:
    def __init__(self):
        self.alephbert_tokenizer = AutoTokenizer.from_pretrained(ALEPH_BERT_BASE)
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
        logger.info('Number of training sentences: {:,}\n'.format(df.shape[0]))
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
        logger.info('Train:', len(self.train))
        logger.info('Validation:', len(self.validation))
        logger.info('Test:', len(self.test))


if __name__ == '__main__':
    dataset = HebrewMistakesDataset()
    dataset.prepare()
    alephbert = MistakeAlephBert(dataset)
    alephbert.train()
    alephbert.eval()
