import os
import pandas as pd
from collections import defaultdict
from utilities import get_most_common_tokens_from_column, open_csv_files_from_path
from consts import YEARS, PATHS


def count_tweets_by_month():
    words_count = {}
    for year in YEARS:
        for single_day_posts, filepath in open_csv_files_from_path(PATHS[year]):
            month_key = get_month_key(filepath, year)
            if month_key not in words_count:
                words_count[month_key] = 0
            words_count[month_key]+=len(list(single_day_posts["used_id"]))
    values = []
    for month_key in words_count.keys():
        values.append((month_key, words_count[month_key]))
    print(values)
    write_output(f'results/meta_data_count/tweets_count.csv', values)


def count_unique_users_by_month():
    users_count = {}
    for year in YEARS:
        for single_day_posts, filepath in open_csv_files_from_path(PATHS[year]):
            month_key = get_month_key(filepath, year)
            if month_key not in users_count:
                users_count[month_key] = {}
            for id in list(single_day_posts["used_id"]):
                if id not in users_count[month_key]:
                    users_count[month_key][id] = 0
                users_count[month_key][id]+=1
    values = []
    for month_key in users_count.keys():
        values.append((month_key, len(users_count[month_key])))
    write_output(f'results/meta_data_count/users_count.csv', values)

def count_words_by_month():
    words_count = {}
    for year in YEARS:
        for single_day_posts, filepath in open_csv_files_from_path(PATHS[year]):
            month_key = get_month_key(filepath, year)
            if month_key not in words_count:
                words_count[month_key] = 0
            for text in list(single_day_posts["text"]):
                words_count[month_key]+=(len(text.split(" ")))
    values = []
    for month_key in words_count.keys():
        values.append((month_key, words_count[month_key]))
    write_output(f'results/meta_data_count/words_count.csv', values)


def write_output(output_file_path, values):
    users_df = pd.DataFrame(values, columns=['month', 'count'])
    users_df.to_csv(output_file_path, index=False, header=True)


def get_month_key(filepath, year):
    filename = os.path.splitext(os.path.basename(filepath))[0]
    month = filename.split('-')[1]
    month_key = f'{year}-{month}'
    return month_key


# count_unique_users_by_month()
count_words_by_month()
# for year in YEARS:
#     for single_day_posts, filepath in open_csv_files_from_path(PATHS[year]):
#         pass
