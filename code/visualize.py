import os
import seaborn
import matplotlib.pyplot as plt
import pandas as pd


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


def get_word_from_path(path):
    filename = os.path.basename(path)
    if any("\u0590" <= c <= "\u05EA" for c in filename.split('_')[0]):
        return ''.join(list(reversed(filename.split('_')[0])))
    else:
        return ''.join(list(filename.split('_')[0]))


def format_month_values(month_tag):
    year, month = month_tag.split('-')
    if month == '01':
        return year
    return ''


def main():
    # visualize_basic_files('results/word_count')
    visualize_basic_files('results/meta_data_count', True)


def visualize_basic_files(data_dir, use_word_from_file=False):
    for filename in os.listdir(data_dir):
        create_single_file_chart(f'{data_dir}/{filename}', use_word_from_file)


main()
