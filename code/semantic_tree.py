from stanza_processor import Processor
from consts import Info, ROOT


class SemanticTree:
    def __init__(self, text):
        self.text = text
        self.processor = Processor()

    def get_gender(self, feature):
        if feature:
            all_features = feature.split('|')
            for f in all_features:
                split_f = f.split('=')
                if split_f[0] == 'Gender':
                    return split_f[1]
        return None

    def parse_text(self):
        parsed_text, tree, pos, features, deprel = self.processor.get_stanza_analysis(self.text)
        word_list = list(zip(list(parsed_text), map(lambda head: head - 1, list(tree)), list(pos),
                             map(lambda feature: self.get_gender(feature), list(features)), list(deprel)))
        self.tree = {index: Info(word, head, pos, gender, deprel) for index, (word, head, pos, gender, deprel) in
                     enumerate(word_list)}
        self.parsed_text = parsed_text



    def parse_text_without_processing(self, parsed_text, tree, pos, features, deprel):
        word_list = list(zip(list(parsed_text), map(lambda head: head - 1, list(tree)), list(pos),
                             map(lambda feature: self.get_gender(feature), list(features)), list(deprel)))
        self.tree = {index: Info(word, head, pos, gender, deprel) for index, (word, head, pos, gender, deprel) in
                     enumerate(word_list)}
        self.parsed_text = parsed_text

    def __str__(self):
        tree_rep = '{\n'
        for index, info in self.tree.items():
            tree_rep += '{}: {}\n'.format(index, info)
        tree_rep += '}\n'
        return tree_rep

    def is_verb(self, word_index):
        return self.tree[word_index].pos == 'VERB'

    def is_root(self, word_index):
        return self.tree[word_index].head == ROOT

    def get_word_in_index(self, index):
        return self.tree[index].word

    def find_verb_root(self, word_index):
        while (not self.is_root(word_index)):
            if self.is_verb(word_index):
                return word_index
            word_index = self.tree[word_index].head
        return word_index
