# Set the backend to use mplcairo
import matplotlib

print('Default backend: ' + matplotlib.get_backend())
matplotlib.use("module://mplcairo.macosx")
print('Backend is now ' + matplotlib.get_backend())

import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
import json
import networkx as nx
from collections import defaultdict, Counter
from typing import List, Dict, Any

from emoji import UNICODE_EMOJI
from utilities import get_posts_from_corpus
from consts import PATHS, YEARS

prop = FontProperties(fname='/System/Library/Fonts/Apple Color Emoji.ttc')
EMOJIS_BY_MONTH_DUMP = './results/emoji/emojis_by_month.json'
EMOJIS_COUNT_PER_POST_DUMP = './results/emoji/total_emoji_per_post_count.json'
EMOJIS_TOTAL_COUNT_DUMP = './results/emoji/total_emoji_count.json'
EMOJIS_MOST_COMMON_PER_MONTH_COUNT_DUMP = './results/emoji/most_common_emoji_per_month.json'


class EmojiStats:
    def __init__(self):
        self.emojis = UNICODE_EMOJI['en']

    def is_emoji(self, char: str) -> bool:
        return char in self.emojis

    def get_emojis_from_text(self, text: str) -> List[str]:
        return [char for char in text if self.is_emoji(char)]

    def does_text_contain_emoji(self, text: str) -> bool:
        return not self.get_emojis_from_text(text) == []


class EmojiGraph:
    def __init__(self):
        self.graph = defaultdict(dict)
        self.more_than_one_emoji_post_count = 0
        self.only_one_emoji_post_count = 0

    def build_graph(self, emojis_by_month):
        for month, month_posts_emojis in emojis_by_month.items():
            print(f'Extracting month {month}')
            for post_emojis in month_posts_emojis:
                if len(post_emojis) > 1:
                    self.more_than_one_emoji_post_count += 1
                    for emoji in list(set(post_emojis)):
                        if self.graph[emoji].get(emoji):
                            self.graph[emoji][emoji] += post_emojis.count(emoji)
                        else:
                            self.graph[emoji][emoji] = 1
                        for neighbor in self.get_emoji_neighbors(post_emojis, emoji):
                            if self.graph[emoji].get(neighbor):
                                self.graph[emoji][neighbor] += 1
                            else:
                                self.graph[emoji][neighbor] = 1
                else:
                    self.only_one_emoji_post_count += 1
        print(self.graph['😂'])

    @staticmethod
    def get_emoji_neighbors(emoji_list, emoji):
        return list({em for em in emoji_list if em != emoji})

    def _get_graph_nodes_and_edges(self):
        per_post_stats = load_emojis(EMOJIS_COUNT_PER_POST_DUMP)
        stats_counter = Counter(per_post_stats)
        most_common_emojis = stats_counter.most_common(10)
        graph_nodes = set()
        graph_edges = set()
        for emoji, count in most_common_emojis:
            neighbors_sorted_by_commonness = sorted(self.graph[emoji].items(), key=lambda item: item[1], reverse=True)[:10]
            graph_nodes.update({related_emoji[0] for related_emoji in neighbors_sorted_by_commonness})
            graph_edges.update([(emoji, neighbor[0]) for neighbor in neighbors_sorted_by_commonness])
        most_common_emojis_no_count = set(map(lambda emoji: emoji[0], most_common_emojis))
        graph_nodes.update(most_common_emojis_no_count)
        return graph_nodes, graph_edges, most_common_emojis_no_count

    def plot(self):
        G = nx.Graph()
        nodes, edges, most_common_emojis = self._get_graph_nodes_and_edges()
        not_common_emojis = nodes - most_common_emojis
        print('*********')
        print(len(nodes))
        G.add_nodes_from(nodes)
        G.add_edges_from(edges)

        pos = nx.spring_layout(G, seed=3113794652)
        print(pos)
        print(len(pos.keys()))
        options = {"edgecolors": "tab:gray", "alpha": 1}
        nx.draw_networkx_nodes(G, pos, nodelist=not_common_emojis, node_color="#ae9dbd", node_size=1000, **options)
        nx.draw_networkx_nodes(G, pos, nodelist=most_common_emojis, node_color="#9069b3", node_size=1000, **options)
        nx.draw_networkx_edges(
            G,
            pos,
            edgelist=edges,
            width=2,
            alpha=0.4,
            edge_color="#1a1f1b"
        )
        for node_text, node_pos in pos.items():
            real_pos = [node_pos[0] - 0.02, node_pos[1] - 0.06]
            plt.annotate(text=node_text, xy=real_pos, xytext=real_pos, fontsize=20, fontproperties=prop)

        plt.show()


def extract_emojis_from_corpus() -> None:
    emoji_stats = EmojiStats()
    emojis_by_month = defaultdict(list)
    for year in YEARS:
        for post, time in get_posts_from_corpus(PATHS[year]):
            post_emojis = emoji_stats.get_emojis_from_text(post)
            if not post_emojis:
                continue
            emojis_by_month[_extract_month(time)].append(post_emojis)
    with open(EMOJIS_BY_MONTH_DUMP, 'w') as emojis_file:
        emojis_file.write(json.dumps(emojis_by_month))


def emoji_total_count(emojis_by_month: Dict[str, list]):
    total_emoji_per_post_count = defaultdict(int)
    total_emoji_count = defaultdict(int)
    for month, month_emoji_list in emojis_by_month.items():
        print(f'Processing {month}')
        for single_post_emojis in month_emoji_list:
            emoji_histogram = Counter(single_post_emojis)
            for emoji, count in emoji_histogram.items():
                total_emoji_per_post_count[emoji] += 1
                total_emoji_count[emoji] += count
    with open(EMOJIS_COUNT_PER_POST_DUMP, 'w') as per_post_file:
        per_post_file.write(json.dumps(total_emoji_per_post_count))
    with open(EMOJIS_TOTAL_COUNT_DUMP, 'w') as total_file:
        total_file.write(json.dumps(total_emoji_count))


def find_most_common_emoji_per_month(emojis_by_month: Dict[str, list]):
    most_common_emoji_per_month = {month: _find_most_common_emoji_in_month(month_emoji_list, 2)
                                   for month, month_emoji_list in emojis_by_month.items()}
    with open(EMOJIS_MOST_COMMON_PER_MONTH_COUNT_DUMP, 'w') as most_common_file:
        most_common_file.write(json.dumps(most_common_emoji_per_month))


def load_emojis(input_file: str) -> Dict[str, Any]:
    with open(input_file, 'r') as emoji_file:
        emojis = json.loads(emoji_file.read())
    return emojis


def _find_most_common_emoji_in_month(month_emoji_list: List[List[str]], n_most_common):
    emoji_count_per_month = Counter(emoji for emoji_list in month_emoji_list for emoji in emoji_list)
    return emoji_count_per_month.most_common(n_most_common)


def _extract_month(time):
    return time[:7]


if __name__ == '__main__':
    # extract_emojis_from_corpus()
    emojis_by_month = load_emojis(EMOJIS_BY_MONTH_DUMP)
    # emoji_total_count(emojis_by_month)
    # find_most_common_emoji_per_month(emojis_by_month)
    graph = EmojiGraph()
    graph.build_graph(emojis_by_month)
    graph.plot()