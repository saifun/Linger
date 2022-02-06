import seaborn as sns
import pandas as pd
from consts import YEARS, GENDER_MISMATCH_PATHS, MONTHS, FUTURE_VERB_PATHS
import os
from visualize import format_month_values
import matplotlib.pyplot as plt
from utilities import generate_df_from_csv_path, invert_words
from collections import  Counter


def write_data_to_csv(mismatch_name, mismatch_dict, dir_name):
    mismatch_df = pd.DataFrame(list(mismatch_dict.items()), columns=['month', 'count'])
    mismatch_df.to_csv(f'results/{dir_name}/{mismatch_name}_count.csv', index=False, header=True)


def create_df_num_gender_mismatch_per_year():
    mismatches_noun_num_per_month = {f'{year}-{month}': 0 for year in YEARS for month in MONTHS}
    mismatches_noun_adj_per_month = {f'{year}-{month}': 0 for year in YEARS for month in MONTHS}
    mismatches_verb_noun_per_month = {f'{year}-{month}': 0 for year in YEARS for month in MONTHS}
    gender_mismatch_dict = {"noun_num": mismatches_noun_num_per_month,
                            "noun_adj": mismatches_noun_adj_per_month,
                            "verb_noun": mismatches_verb_noun_per_month}
    for year in YEARS:
        for df, month, filename in generate_df_from_csv_path(GENDER_MISMATCH_PATHS[year]):
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
    for year in YEARS:
        for df, month, filename in generate_df_from_csv_path(FUTURE_VERB_PATHS[year]):
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
    plot = sns.barplot(x='month', y='count', data=final_df, palette="pastel")
    # plot = sns.lineplot(x='month', y='count', hue='mismatch name', style="mismatch name", markers=True, data=final_df, palette="pastel")
    plot.set_xticklabels(x_values)
    # plt.xticks(ticks=list(range(len(x_values))), labels=x_values)
    plt.title("Wrong Future Verb Count per Month")
    plt.xlabel('Months')
    plt.ylabel('Wrong Future Verb Count')

    plt.show()

# create_gender_mismatch_graph()
# create_future_verb_graph()
# get_corpus_for_gender_mismatch_head_words()
# plot_top_gender_mismatch_words_barchart()
# create_gender_mismatch_graph