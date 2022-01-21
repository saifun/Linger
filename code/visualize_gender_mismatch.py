import seaborn
import pandas as pd
import os
from visualize import format_month_values
import matplotlib.pyplot as plt


def get_mismatch_name_from_path(path):
    filename = os.path.basename(path)
    return ' '.join(filename.split('_')[:2])


def generate_gender_mismatch_graph():
    data_dir = 'results/gender_mismatch'
    seaborn.set_theme(style="ticks")
    dfs = []
    for filename in os.listdir(data_dir):
        df = pd.read_csv(f'{data_dir}/{filename}', encoding='utf-8')
        df = df.sort_values(by='month')
        df['name'] = get_mismatch_name_from_path(filename)
        dfs.append(df)
        x_values = [format_month_values(month_tag) for month_tag in df['month']]
    final_df = pd.concat(dfs, ignore_index=True)
    plot = seaborn.barplot(x='month', y='count', hue='name', data=final_df, palette="pastel")
    plot.set_xticklabels(x_values)
    # plt.xticks(ticks=list(range(len(x_values))), labels=x_values)
    plt.title("Gender Mismatch Count per Month")
    plt.xlabel('Months')
    plt.ylabel('Gender Mismatch Count')

    plt.show()

generate_gender_mismatch_graph()