# Set the backend to use mplcairo
import matplotlib
from wordcloud import WordCloud

print('Default backend: ' + matplotlib.get_backend())
matplotlib.use("module://mplcairo.macosx")
print('Backend is now ' + matplotlib.get_backend())

import matplotlib.pyplot as plt, numpy as np
from matplotlib.font_manager import FontProperties
from collections import Counter

from consts import RECORDS_COUNT_STATS
from fun_with_emoji import load_emojis, EMOJIS_COUNT_PER_POST_DUMP, EMOJIS_MOST_COMMON_PER_MONTH_COUNT_DUMP, \
    EMOJIS_TOTAL_COUNT_DUMP

prop = FontProperties(fname='/System/Library/Fonts/Apple Color Emoji.ttc')


def _convert_post_number_to_percents(number):
    total_post_count = _get_total_post_count()
    return number / total_post_count * 100


def _convert_percents_to_post_number(number):
    total_post_count = _get_total_post_count()
    return number * total_post_count / 100


def format_month_values(month_tag):
    year, month = month_tag.split('-')
    if month in ['01', '07']:
        return month_tag
    return ''


def create_per_post_visualization():
    per_post_stats = load_emojis(EMOJIS_COUNT_PER_POST_DUMP)
    stats_counter = Counter(per_post_stats)
    most_common_emojis = stats_counter.most_common(10)
    x_values, counts = list(zip(*most_common_emojis))
    print(x_values)
    # percents = [count / total_post_count * 100 for count in counts]
    plot_emoji_bar_chart(x_values, counts)
    # matplotlib.use("module://mplcairo.macosx")
    # plt.rcParams.update({'font.size': 18})
    # prop = FontProperties(fname='/System/Library/Fonts/Apple Color Emoji.ttc')
    # plt.rcParams['font.family'] = prop.get_family()
    # seaborn.barplot(x=list(x_values), y=percents, color='purple')
    # plt.title('Emoji Commonness - Post Percentage')
    # plt.xlabel('Emoji')
    # plt.ylabel('Percent of posts')
    # plt.show()


def _get_total_post_count():
    return sum(RECORDS_COUNT_STATS.values())


def plot_emoji_bar_chart(labels, freqs):
    _, plot = plt.subplots(constrained_layout=True, figsize=(12, 7))
    p1 = plot.bar(np.arange(len(labels)), freqs, 0.8, color="lightblue")
    plot.set_ylim(0, plt.ylim()[1] + 100000)
    plot.set_title('Emoji usage percent', fontdict={'fontsize': 30})
    plot.set_xlabel('Emoji', fontdict={'fontsize': 20})
    plot.set_ylabel('Post number', fontdict={'fontsize': 20})
    sec_y_axis = plot.secondary_yaxis('right',
                                      functions=(_convert_post_number_to_percents, _convert_percents_to_post_number))
    sec_y_axis.set_ylabel('Percent of posts containing the emoji', fontdict={'fontsize': 20})

    # Make labels
    for rect1, label in zip(p1, labels):
        height = rect1.get_height()
        plot.annotate(
            label,
            (rect1.get_x() + rect1.get_width() / 2, height),
            ha="center",
            va="bottom",
            fontsize=30,
            fontproperties=prop
        )

    plt.show()


def prepare_monthly_winner_emoji():
    monthly_stats = load_emojis(EMOJIS_MOST_COMMON_PER_MONTH_COUNT_DUMP)
    print(monthly_stats)
    months = sorted(list(monthly_stats.keys()))
    x_values = [format_month_values(month_tag) for month_tag in months]
    first_place_emojis = [stat[0][0] for stat in monthly_stats.values()]
    second_place_emojis = [stat[1][0] for stat in monthly_stats.values()]
    first_place_values = [stat[0][1] for stat in monthly_stats.values()]
    second_place_values = [stat[1][1] for stat in monthly_stats.values()]
    fig, ax = plt.subplots(constrained_layout=True, figsize=(12, 7))
    ax.plot(months, first_place_values)
    plt.plot(months, second_place_values)
    ax.set_xticklabels(x_values)
    ax.set_title('Most Used Emoji Per Month', fontdict={'fontsize': 30})
    ax.set_xlabel('Month', fontdict={'fontsize': 20})
    ax.set_ylabel('Number of Usages', fontdict={'fontsize': 20})
    ax.legend(["Most common", "Second most common"])
    for x_val, y_val, label in zip(months, first_place_values, first_place_emojis):
        height = y_val
        ax.annotate(
            label,
            (x_val, height),
            ha="center",
            va="bottom",
            fontsize=15,
            fontproperties=prop
        )
    for x_val, y_val, label in zip(months, second_place_values, second_place_emojis):
        height = y_val
        ax.annotate(
            label,
            (x_val, height),
            ha="center",
            va="bottom",
            fontsize=15,
            fontproperties=prop
        )
    plt.show()


if __name__ == '__main__':
    create_per_post_visualization()
    prepare_monthly_winner_emoji()