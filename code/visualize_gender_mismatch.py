import seaborn as sns
import pandas as pd
from consts import YEARS, GENDER_MISMATCH_PATHS, MONTHS, FUTURE_VERB_PATHS, PATHS
import os
from visualize import format_month_values
import matplotlib.pyplot as plt
from utilities import generate_df_from_csv_path, invert_words, get_all_tokens_from_array
from collections import Counter
import networkx as nx
import itertools
from wordcloud import WordCloud
from visualize import _reverse_hebrew_word
import glob


def rename_head_mismatch_columns_in_verb_noun():
    for year in YEARS:
        for df, month, filename in generate_df_from_csv_path(GENDER_MISMATCH_PATHS[year]):
            if "verb_noun" in filename:
                df.rename(columns={'head': 'mismatch', 'mismatch': 'head', 'head_gender': 'verb_gender'}, inplace=True)
                df.to_csv(filename, columns=['sentence', 'month', 'year', 'mismatch', 'verb_gender', 'head'])

# rename_head_mismatch_columns_in_verb_noun()

def write_data_to_csv(mismatch_name, mismatch_dict, dir_name):
    mismatch_df = pd.DataFrame(list(mismatch_dict.items()), columns=['month', 'count'])
    mismatch_df.to_csv(f'results/{dir_name}/{mismatch_name}_count.csv', index=False, header=True)


def create_df_num_gender_mismatch_per_year():
    mismatches_noun_num_per_month = {f'{year}-{month}': 0 for year in YEARS for month in MONTHS}
    mismatches_noun_adj_per_month = {f'{year}-{month}': 0 for year in YEARS for month in MONTHS}
    mismatches_verb_noun_per_month = {f'{year}-{month}': 0 for year in YEARS for month in MONTHS}
    for month in MONTHS:
        if int(month) > 3:
            del mismatches_noun_num_per_month[f'2021-{month}']
            del mismatches_noun_adj_per_month[f'2021-{month}']
            del mismatches_verb_noun_per_month[f'2021-{month}']
    del mismatches_noun_num_per_month[f'2018-01']
    del mismatches_noun_adj_per_month[f'2018-01']
    del mismatches_verb_noun_per_month[f'2018-01']
    del mismatches_noun_num_per_month[f'2018-02']
    del mismatches_noun_adj_per_month[f'2018-02']
    del mismatches_verb_noun_per_month[f'2018-02']
    del mismatches_noun_num_per_month[f'2019-04']
    del mismatches_noun_adj_per_month[f'2019-04']
    del mismatches_verb_noun_per_month[f'2019-04']
    gender_mismatch_dict = {"noun_num": mismatches_noun_num_per_month,
                            "noun_adj": mismatches_noun_adj_per_month,
                            "verb_noun": mismatches_verb_noun_per_month}
    for year in YEARS:
        for df, month, filename in generate_df_from_csv_path(GENDER_MISMATCH_PATHS[year]):
            if 'chunk1_' in filename:
                mismatch_name = get_mismatch_name_from_path(filename)
                gender_mismatch_dict[mismatch_name][f'{year}-{month}'] += len(df)
    write_data_to_csv("noun_num", gender_mismatch_dict["noun_num"], "gender_mismatch")
    write_data_to_csv("noun_adj", gender_mismatch_dict["noun_adj"], "gender_mismatch")
    write_data_to_csv("verb_noun", gender_mismatch_dict["verb_noun"], "gender_mismatch")


def get_corpus_for_gender_mismatch_head_words():
    counter = Counter([])
    for year in YEARS:
        for df, month, filename in generate_df_from_csv_path(GENDER_MISMATCH_PATHS[year]):
            curr_head_words = list(df['head'])
            curr_counter = Counter(curr_head_words)
            counter = sum([counter, curr_counter], Counter())
    return counter


def plot_top_gender_mismatch_words_barchart():
    # stop=set(get_hebrew_stopwords())

    # new = text.str.split()
    # new = new.values.tolist()
    # corpus = [word for i in new for word in i]
    #
    counter = get_corpus_for_gender_mismatch_head_words()
    most = counter.most_common()
    x, y = [], []
    for word, count in most[:10]:
        x.append(word)
        y.append(count)

    sns.barplot(x=y, y=invert_words(x))

    plt.show()


def create_df_num_wrong_future_verb_per_year():
    wrong_future_verb_per_month = {f'{year}-{month}': 0 for year in YEARS for month in MONTHS}
    for month in MONTHS:
        if int(month) > 3:
            del wrong_future_verb_per_month[f'2021-{month}']
    del wrong_future_verb_per_month[f'2018-01']
    del wrong_future_verb_per_month[f'2018-02']
    del wrong_future_verb_per_month[f'2019-04']
    for year in YEARS:
        for df, month, filename in generate_df_from_csv_path(FUTURE_VERB_PATHS[year]):
            if 'chunk1_' in filename:
                wrong_future_verb_per_month[f'{year}-{month}'] += len(df)
    write_data_to_csv("future_verb", wrong_future_verb_per_month, "future_verb")


def get_mismatch_name_from_path(path):
    filename = os.path.basename(path)
    return '_'.join(filename.split('_')[3:5])


def get_mismatch_name_from_count_file(path):
    filename = os.path.basename(path)
    return ' '.join(filename.split('_')[:2])


def create_gender_mismatch_graph():
    data_dir = 'results/gender_mismatch'
    sns.set_theme(style="ticks")
    dfs = []
    for filename in os.listdir(data_dir):
        df = pd.read_csv(f'{data_dir}/{filename}', encoding='utf-8')
        df = df.sort_values(by='month')
        df['mismatch name'] = get_mismatch_name_from_count_file(filename)
        dfs.append(df)
        x_values = [format_month_values(month_tag) for month_tag in df['month']]
    final_df = pd.concat(dfs, ignore_index=True)
    plot = sns.lineplot(x='month', y='count', hue='mismatch name', data=final_df, palette="pastel")
    plot.set_xticklabels(x_values)
    # plt.xticks(ticks=list(range(len(x_values))), labels=x_values)
    plt.title("Gender Mismatch Count per Month")
    plt.xlabel('Months')
    plt.ylabel('Gender Mismatch Count')

    plt.show()


def create_future_verb_graph():
    data_dir = 'results/future_verb'
    sns.set_theme(style="ticks")
    dfs = []
    for filename in os.listdir(data_dir):
        df = pd.read_csv(f'{data_dir}/{filename}', encoding='utf-8')
        df = df.sort_values(by='month')
        dfs.append(df)
        x_values = [format_month_values(month_tag) for month_tag in df['month']]
    final_df = pd.concat(dfs, ignore_index=True)
    plot = sns.barplot(x='month', y='count', data=final_df, color='#CC99FF')
    # plot = sns.lineplot(x='month', y='count', hue='mismatch name', style="mismatch name", markers=True, data=final_df, palette="pastel")
    plot.set_xticklabels(x_values)
    # plt.xticks(ticks=list(range(len(x_values))), labels=x_values)
    plt.title("Wrong Future Verb Count per Month")
    plt.xlabel('Months')
    plt.ylabel('Wrong Future Verb Count')

    plt.show()


def create_graph_for_common_gender_mismatches_wordsun():
    counter = get_corpus_for_gender_mismatch_head_words()
    most = counter.most_common()
    most_common_heads = [word for word, count in most[23:25]] + ['יומולדת']
    heads_to_mistakes_list = {head: set() for head in most_common_heads}
    for year in YEARS:
        for df, month, filename in generate_df_from_csv_path(GENDER_MISMATCH_PATHS[year]):
            for head_word in heads_to_mistakes_list:
                mistakes = df[df['head'] == head_word]
                if not mistakes.empty:
                    # print(mistakes['sentence'])
                    heads_to_mistakes_list[head_word].update(list(mistakes['mismatch']))


def create_graph_for_common_gender_mismatches_wordsun_interesting_word(word):
    most_common_heads = [word]
    heads_to_mistakes_list = {head: set() for head in most_common_heads}
    for year in YEARS:
        for df, month, filename in generate_df_from_csv_path(GENDER_MISMATCH_PATHS[year]):
            for head_word in heads_to_mistakes_list:
                mistakes = df[df['head'] == head_word]
                if not mistakes.empty:
                    # print(mistakes['sentence'])
                    heads_to_mistakes_list[head_word].update(list(mistakes['mismatch']))

    plot_gender_mismatch_word_graph(word, heads_to_mistakes_list[word])


def create_graph_for_common_gender_mismatches_wordsun_number_in_center(number):
    most_common_heads = [number]
    number_to_noun_list = {number_word: set() for number_word in most_common_heads}
    for year in YEARS:
        for df, month, filename in generate_df_from_csv_path(GENDER_MISMATCH_PATHS[year]):
            if 'noun_num' in filename:
                for number_word in number_to_noun_list:
                    mistakes = df[df['mismatch'] == number_word]
                    if not mistakes.empty:
                        # print(mistakes['sentence'])
                        number_to_noun_list[number_word].update(list(mistakes['head']))

    plot_gender_mismatch_word_graph(number, number_to_noun_list[number])

def create_graph_for_common_gender_mismatches_wordsun_number_in_center_most_common(number):
    most_common_heads = [number]
    number_to_noun_list = {number_word: list() for number_word in most_common_heads}
    for year in YEARS:
        for df, month, filename in generate_df_from_csv_path(GENDER_MISMATCH_PATHS[year]):
            if 'noun_num' in filename:
                for number_word in number_to_noun_list:
                    mistakes = df[df['mismatch'] == number_word]
                    if not mistakes.empty:
                        print(list(mistakes['sentence']))
                        number_to_noun_list[number_word].extend(list(mistakes['head']))
    count_mistakes = Counter(number_to_noun_list[number])
    print(count_mistakes)
    common = count_mistakes.most_common()[:10]

    plot_gender_mismatch_word_graph(number, set(word for word, count in common))

def plot_gender_mismatch_word_graph(head, mistakes):
    G = nx.Graph()
    head_list = invert_words([head])
    mistakes = invert_words(mistakes)
    G.add_nodes_from(head_list)
    G.add_nodes_from(mistakes)
    edges = list(itertools.product(head_list, mistakes))
    G.add_edges_from(edges)

    pos = nx.spring_layout(G, seed=3113794652)
    options = {"edgecolors": "tab:gray", "alpha": 1}
    nx.draw_networkx_nodes(G, pos, nodelist=mistakes, node_color="#539A9F", node_size=3000, **options)
    nx.draw_networkx_nodes(G, pos, nodelist=head_list, node_color="orange", node_size=8000, **options)
    nx.draw_networkx_edges(
        G,
        pos,
        edgelist=edges,
        width=4,
        alpha=0.4,
        edge_color="orange",
    )
    nx.draw_networkx_labels(G, pos, head_list.extend(mistakes), font_size=14, font_color="whitesmoke")

    # nx.draw_kamada_kawai(G, with_labels=True, font_weight='bold', font_color='w', node_color='lightblue', node_size=2000)

    plt.show()


def plot_gender_mismatch_word_graph_example():
    G = nx.Graph()
    head_list = invert_words(['בלונים'])
    mistakes = invert_words(['צהובות', 'יפות', 'עפות', 'שלוש', 'חמש', 'ירוקות'])
    G.add_nodes_from(head_list)
    G.add_nodes_from(mistakes)
    edges = list(itertools.product(head_list, mistakes))
    G.add_edges_from(edges)

    pos = nx.spring_layout(G, seed=3113794652)
    options = {"edgecolors": "tab:gray", "alpha": 1}
    nx.draw_networkx_nodes(G, pos, nodelist=mistakes, node_color="#539A9F", node_size=3000, **options)
    nx.draw_networkx_nodes(G, pos, nodelist=head_list, node_color="orange", node_size=8000, **options)
    nx.draw_networkx_edges(
        G,
        pos,
        edgelist=edges,
        width=4,
        alpha=0.4,
        edge_color="orange",
    )
    nx.draw_networkx_labels(G, pos, head_list.extend(mistakes), font_size=14, font_color="whitesmoke")

    # nx.draw_kamada_kawai(G, with_labels=True, font_weight='bold', font_color='w', node_color='lightblue', node_size=2000)

    plt.show()

def get_common_mistaken_verbs():
    counter = Counter([])
    for year in YEARS:
        for df, month, filename in generate_df_from_csv_path(FUTURE_VERB_PATHS[year]):
            curr_verbs = list(df['verb'])
            # if 'יביסט' in curr_verbs:
            #     print(df['sentence'])
            curr_counter = Counter(curr_verbs)
            counter = sum([counter, curr_counter], Counter())
    return counter


def plot_word_cloud_for_common_mistaken_verbs():
    # with open(filename, 'r') as file:
    #     word_count = json.loads(file.read())
    # hebrew_word_data = {
    #     _reverse_hebrew_word(word): count
    #     for word, count in word_count.items()
    # }
    counter = get_common_mistaken_verbs()
    hebrew_word_data = {
        _reverse_hebrew_word(word): count
        for word, count in counter.items()
    }
    word_cloud = WordCloud(font_path='./utils/Trashim-CLM-Bold.ttf')
    word_cloud.generate_from_frequencies(frequencies=hebrew_word_data)
    plt.figure()
    plt.imshow(word_cloud, interpolation="bilinear")
    plt.axis("off")
    plt.show()


def count_interesting_words_in_the_data(mistakes_list):
    count_df = pd.DataFrame([], columns=['word', 'total', 'mistake_num', 'percent'])
    # count_df['word'] = ['שלושת', 'גרביים', 'שתי']
    for word in mistakes_list:
        to_add = pd.DataFrame([[word, 0, 0, 0]], columns=['word', 'total', 'mistake_num', 'percent'])
        count_df = count_df.append(to_add)
    # count_df['word'] = mistakes_list
    # count_df = pd.DataFrame([['', 0,0,0,0]], columns=['row_name', 'birthday', 'socks', 'three_masc', 'two_fem'])
    # count_df['row_name'] = 'total'
    for year in YEARS:
        all_files = glob.glob(PATHS[year] + "/*.csv")
        for filename in all_files:
            try:
                df_iter = pd.read_csv(filename, chunksize=200, iterator=True, encoding='utf-8')
                for first_df in df_iter:
                    tokens = get_all_tokens_from_array(list(first_df['text']))
                    counts = Counter(tokens)
                    for word in mistakes_list:
                        count_df.loc[count_df['word'] == word, "total"] += counts[word]
                    # count_df.loc[count_df['word'] == 'שלושת', "total"] += counts['שלושת']
                    # count_df.loc[count_df['word'] == 'גרביים', "total"] += counts['גרביים']
                    # count_df.loc[count_df['word'] == 'שתי', "total"] += counts['שתי']
                    # count_df['three_masc'] += counts['שלושת']
                    # count_df['socks'] += counts['גרביים']
                    # count_df['two_fem'] += counts['שתי']
                    break
            except:
                print(filename)
    count_df.to_csv('./results/gender_mismatch/count_interesting_words.csv')


def count_mistaken_interesting_words_in_the_data(mistakes_list):
    count_df = pd.read_csv('./results/gender_mismatch/count_interesting_words.csv')
    # to_add = pd.DataFrame([['', 0,0,0,0]], columns=['row_name', 'birthday', 'socks', 'three_masc', 'two_fem'])
    # to_add['row_name'] = 'mistake_num'
    for year in YEARS:
        for df, month, filename in generate_df_from_csv_path(GENDER_MISMATCH_PATHS[year]):
            if 'chunk1_' in filename:
                if 'noun_num' in filename:
                    counts = Counter(list(df['mismatch']))
                    # count_df['three_masc'] += counts['שלושת']
                    # count_df['two_fem'] += counts['שתי']
                    count_df.loc[count_df['word'] == 'שלושת', "mistake_num"] += counts['שלושת']
                    count_df.loc[count_df['word'] == 'שתי', "mistake_num"] += counts['שתי']
                counts = Counter(get_all_tokens_from_array(list(df['head'])))
                # count_df['socks'] += counts['גרביים']
                for word in mistakes_list[2:]:
                    count_df.loc[count_df['word'] == word, "mistake_num"] += counts[word]
                # count_df.loc[count_df['word'] == 'גרביים', "mistake_num"] += counts['גרביים']
                # count_df.loc[count_df['word'] == 'משקפיים', "mistake_num"] += counts['משקפיים']
                # count_df.loc[count_df['word'] == 'אופניים', "mistake_num"] += counts['אופניים']
    # count_df = count_df.append(to_add, ignore_index=True)
    count_df.to_csv('./results/gender_mismatch/count_interesting_words.csv', columns=['word', 'total', 'mistake_num', 'percent'])

def calculate_percent_interesting_words_in_the_data():
    count_df = pd.read_csv('./results/gender_mismatch/count_interesting_words.csv')
    # count_df['percent'] = (count_df[["mistake_num"]] / count_df['total']) * 100
    for index, row in count_df.iterrows():
        count_df.loc[count_df['word'] == row['word'], "percent"] = (row["mistake_num"] / float(row['total'])) * 100
    count_df.to_csv('./results/gender_mismatch/count_interesting_words.csv',
                    columns=['word', 'total', 'mistake_num', 'percent'])


def plot_mistake_percent_interesting_words():
    sns.set_theme(style="ticks")
    df = pd.read_csv('results/gender_mismatch/count_interesting_words.csv', encoding='utf-8')
    plot = sns.barplot(x=invert_words(df['word']), y='percent', data=df, color='#CC99FF')
    # plot = sns.lineplot(x='month', y='count', hue='mismatch name', style="mismatch name", markers=True, data=final_df, palette="pastel")
    # plot.set_xticklabels(x_values)
    # plt.xticks(ticks=list(range(len(x_values))), labels=x_values)
    plt.title("Percent of mistaken words from the corpus")
    plt.xlabel('Words')
    plt.ylabel('Gender mismatch percent')
    plt.ylim(-0.3, 50)

    plt.show()


def create_interesting_words_graph(word_list):
    count_interesting_words_in_the_data(word_list)
    count_mistaken_interesting_words_in_the_data(word_list)
    calculate_percent_interesting_words_in_the_data()
    plot_mistake_percent_interesting_words()

# plot_gender_mismatch_word_graph_example()
# create_gender_mismatch_graph()
# create_df_num_gender_mismatch_per_year()
# create_future_verb_graph()
# plot_top_gender_mismatch_words_barchart()
# create_gender_mismatch_graph
# create_graph_for_common_gender_mismatches_wordsun()
# plot_word_cloud_for_common_mistaken_verbs()
# create_df_num_wrong_future_verb_per_year()
# create_graph_for_common_gender_mismatches_wordsun_number_in_center('שני')
# create_graph_for_common_gender_mismatches_wordsun_number_in_center_most_common('שלושת')
create_interesting_words_graph(['שלושת', 'שתי','גרביים','משקפיים', 'אופניים', 'צומת'])
