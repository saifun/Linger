import stanza
import pandas as pd


class Processor:
    """
    This class is using Stanford's Stanza to process sentences.
    """
    def __init__(self):
        self.heb_nlp = stanza.Pipeline(lang='he', processors='tokenize,mwt,pos,lemma,depparse', verbose=False)

    def get_stanza_analysis(self, text):
        text += " XX"
        doc = self.heb_nlp(text)
        lst = []
        for sen in doc.sentences:
            for token in sen.tokens:
                for word in token.words:
                    features = [(word.text,
                                 word.lemma,
                                 word.upos,
                                 word.xpos,
                                 word.head,
                                 word.deprel,
                                 word.feats)]

                    df = pd.DataFrame(features, columns=["text", "lemma", "upos", "xpos", "head", "deprel", "feats"])
                    lst.append(df)
        tot_df = pd.concat(lst, ignore_index=True)
        tot_df = tot_df.shift(1).iloc[1:]
        tot_df["head"] = tot_df["head"].astype(int)
        return tot_df['text'], tot_df['head'], tot_df['upos'], tot_df['feats'], tot_df['deprel']

    def get_stanza_analysis_multiple_sentences(self, sentences_list):
        in_docs = [stanza.Document([], text=sent) for sent in sentences_list]
        out_docs = self.heb_nlp(in_docs)
        dfs = []
        for doc in out_docs:
            for sen in doc.sentences:
                lst = []
                for token in sen.tokens:
                    for word in token.words:
                        features = [(doc.text,
                                    word.text,
                                     word.lemma,
                                     word.upos,
                                     word.xpos,
                                     word.head,
                                     word.deprel,
                                     word.feats)]

                        df = pd.DataFrame(features, columns=["sentence", "text", "lemma", "upos", "xpos", "head", "deprel", "feats"])
                        lst.append(df)
                tot_df = pd.concat(lst, ignore_index=True)
                tot_df = tot_df.shift(1).iloc[1:]
                tot_df["head"] = tot_df["head"].astype(int)
                dfs.append(tot_df)
        return dfs
