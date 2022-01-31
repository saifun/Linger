from stanza_processor import Processor
from consts import Info, ROOT


class SemanticTree:
    def __init__(self, text):
        self.text = text
        self.processor = Processor()

    def get_feature_dict(self, feature):
        # feature example: Gender=Masc|HebBinyan=HUFAL|Number=Sing|Person=3|Tense=Fut|Voice=Pass
        if feature:
            feature_dict = dict()
            all_features = feature.split('|')
            for f in all_features:
                split_f = f.split('=')
                feature_dict[split_f[0]] = split_f[1]
            return feature_dict
        return None

    def get_gender(self, feature_dict):
        if feature_dict and 'Gender' in feature_dict:
            return feature_dict['Gender']
        return None

    def get_tense(self, feature_dict):
        if feature_dict and 'Tense' in feature_dict:
            return feature_dict['Tense']
        return None

    def get_number(self, feature_dict):
        if feature_dict and 'Number' in feature_dict:
            return feature_dict['Number']
        return None

    def get_person(self, feature_dict):
        if feature_dict and 'Person' in feature_dict:
            return feature_dict['Person']
        return None

    def parse_text(self):
        parsed_text, tree, pos, features, deprel = self.processor.get_stanza_analysis(self.text)
        word_list = list(zip(list(parsed_text), map(lambda head: head - 1, list(tree)), list(pos),
                             map(lambda feature: self.get_feature_dict(feature), list(features)),list(deprel)))
        self.tree = {index: Info(word, head, pos, self.get_gender(feature_dict), self.get_tense(feature_dict),
                                 self.get_number(feature_dict), self.get_person(feature_dict), deprel)
                        for index, (word, head, pos, feature_dict, deprel) in enumerate(word_list)}
        self.parsed_text = parsed_text



    def parse_text_without_processing(self, parsed_text, tree, pos, features, deprel):
        word_list = list(zip(list(parsed_text), map(lambda head: head - 1, list(tree)), list(pos),
                             map(lambda feature: self.get_feature_dict(feature), list(features)), list(deprel)))
        self.tree = {index: Info(word, head, pos, self.get_gender(feature_dict), self.get_tense(feature_dict),
                                 self.get_number(feature_dict), self.get_person(feature_dict), deprel)
                     for index, (word, head, pos, feature_dict, deprel) in enumerate(word_list)}
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
