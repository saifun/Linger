import os
import seaborn
import json
import matplotlib.pyplot as plt
import pandas as pd
from consts import HEB_CAHRS_START, HEB_CAHRS_END
from sklearn import preprocessing
from wordcloud import WordCloud


def create_single_file_chart(filename, use_word_from_file=False):
    y_axis_count_parameter = "Word"
    if use_word_from_file:
        y_axis_count_parameter = get_word_from_path(filename)
    seaborn.set_theme(style="ticks")
    df = pd.read_csv(filename, encoding='utf-8')
    df = df.sort_values(by='month')

    x_values = [format_month_values(month_tag) for month_tag in df['month']]
    plot = seaborn.lineplot(x=df['month'], y=df['count'], color='purple')
    plot.set_xticklabels(x_values)
    plt.title(get_word_from_path(filename))
    plt.xlabel('Months')
    plt.ylabel(y_axis_count_parameter + ' count')

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
    if any(HEB_CAHRS_START <= c <= HEB_CAHRS_END for c in filename.split('_')[0]):
        return ''.join(list(reversed(filename.split('_')[0])))
    else:
        return ''.join(list(filename.split('_')[0]))
    return _reverse_hebrew_word(filename.split('_')[0])


def _reverse_hebrew_word(word):
    return ''.join(list(reversed(word)))


def format_month_values(month_tag):
    year, month = month_tag.split('-')
    if month == '01':
        return year
    return ''

def visualize_trends():
    month_df = pd.read_csv('results/trends_count/month.csv')
    day_df = pd.read_csv('results/trends_count/day.csv')
    # df = month_df
    plot_trend_df(month_df)
    plot_trend_df(day_df)
    # pass


def plot_trend_df(df):
    dates = df['date']
    df.set_index(['date'], inplace=True)
    df.plot()
    cols = df.columns
    x = df.values  # returns a numpy array
    min_max_scaler = preprocessing.MinMaxScaler()
    x_scaled = min_max_scaler.fit_transform(x)
    df = pd.DataFrame(x_scaled)
    df.columns = cols
    df['date'] = dates
    df.set_index(['date'], inplace=True)
    df.plot()
    plt.show()


def main():
    visualize_trends()
    # visualize_basic_files('results/word_count')
    # visualize_basic_files('results/meta_data_count', True)


def visualize_basic_files(data_dir, use_word_from_file=False):
    for filename in os.listdir(data_dir):
        create_single_file_chart(f'{data_dir}/{filename}', use_word_from_file)

def generate_word_count_graphs():
    data_dir = 'results/word_count'
    for filename in os.listdir(data_dir):
        create_single_file_chart(f'{data_dir}/{filename}')

if __name__ == '__main__':
    main()
