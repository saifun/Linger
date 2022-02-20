import os
import pandas as pd
from collections import defaultdict
from utilities import get_most_common_tokens_from_column, open_csv_files_from_path, get_single_column
from consts import YEARS, PATHS
SEPERATOR = "###"


def get_month_key(filepath, year):
    filename = os.path.splitext(os.path.basename(filepath))[0]
    month = filename.split('-')[1]
    month_key = f'{year}-{month}'
    return month_key

def get_day_key(filepath, year):
    filename = os.path.splitext(os.path.basename(filepath))[0]
    month = filename.split('-')[1]
    day = filename.split('-')[2]
    month_key = f'{year}-{month}-{day}'
    return month_key

def create_keywords_dict():
    keywords_dict = {}
    with open('trend_files\\trends.txt', encoding="utf8") as trends:
        new = False
        curr = ""
        for line in trends:
            word = line[:-1]
            if word == SEPERATOR:
                new = True
            elif new:
                new = False
                curr = word
                keywords_dict[word] = []
            else:
                keywords_dict[curr].append(word)
    return keywords_dict


def main():
    keywords_dict = create_keywords_dict()
    create_trends_result_file(get_day_key, keywords_dict, "day")
    create_trends_result_file(get_month_key, keywords_dict, "month")



def create_trends_result_file(key_func, keywords_dict, name):
    results = {}
    for year in YEARS:
        for single_day_posts, filepath in open_csv_files_from_path(PATHS[year]):
            tweet_texts = set(get_single_column(single_day_posts, 'text'))
            time_key = key_func(filepath, year)
            if time_key not in results.keys():
                results[time_key] = {}
                for topic in keywords_dict.keys():
                    if topic not in results[time_key].keys():
                        results[time_key][topic] = 0
            for text in tweet_texts:
                for topic in keywords_dict.keys():
                    for keyword in keywords_dict[topic]:
                        if keyword in text:
                            results[time_key][topic] += 1
                        break
    pd.DataFrame.from_dict(results).T.to_csv(f"results\\trends_count\\{name}.csv")


main()

