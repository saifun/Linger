from stanza_processor import Processor
from semantic_tree import SemanticTree
from utilities import generate_sentences
from consts import YEARS, PATHS

processor = Processor()


def find_future_anomalies():
    sentence = 'אני יאכל ארוחת צהריים'
    tree = SemanticTree(sentence)
    tree.parse_text()
    print(tree.tree)


if __name__ == '__main__':
    find_future_anomalies()
    # for year in YEARS:
    #     generate_sentences(PATHS[year])
