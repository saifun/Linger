import os
import seaborn
import json
import matplotlib.pyplot as plt
import pandas as pd
from wordcloud import WordCloud


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


def visualize_word_cloud(filename):
    with open(filename, 'r') as file:
        word_count = json.loads(file.read())
    hebrew_word_data = {
        _reverse_hebrew_word(word): count
        for word, count in word_count.items()
    }
    word_cloud = WordCloud(font_path='/Users/saifun/Library/Fonts/trashimclm-bold-webfont.ttf')
    word_cloud.generate_from_frequencies(frequencies=hebrew_word_data)
    plt.figure()
    plt.imshow(word_cloud, interpolation="bilinear")
    plt.axis("off")
    plt.show()


def get_word_from_path(path):
    filename = os.path.basename(path)
    return _reverse_hebrew_word(filename.split('_')[0])


def _reverse_hebrew_word(word):
    return ''.join(list(reversed(word)))


def format_month_values(month_tag):
    year, month = month_tag.split('-')
    if month == '01':
        return year
    return ''


def generate_word_count_graphs():
    data_dir = 'results/word_count'
    for filename in os.listdir(data_dir):
        create_single_file_chart(f'{data_dir}/{filename}')
