import os
import seaborn
import matplotlib.pyplot as plt
import pandas as pd


def create_single_file_chart(filename):
    seaborn.set_theme(style="ticks")
    df = pd.read_csv(filename, encoding='utf-8')
    df = df.sort_values(by='month')

    x_values = [format_month_values(month_tag) for month_tag in df['month']]
    plot = seaborn.lineplot(x=df['month'], y=df['count'], color='purple')
    plot.set_xticklabels(x_values)
    plt.title(get_word_from_path(filename))
    plt.xlabel('Months')
    plt.ylabel('Word count')

    plt.show()


def get_word_from_path(path):
    filename = os.path.basename(path)
    return ''.join(list(reversed(filename.split('_')[0])))


def format_month_values(month_tag):
    year, month = month_tag.split('-')
    if month == '01':
        return year
    return ''


data_dir = 'results/word_count'
for filename in os.listdir(data_dir):
    create_single_file_chart(f'{data_dir}/{filename}')
