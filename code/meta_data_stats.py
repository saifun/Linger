import os
import pandas as pd
from collections import defaultdict
from utilities import get_most_common_tokens_from_column, open_csv_files_from_path
from consts import YEARS, PATHS


def count_unique_users_by_month():
    users_count = {}
    for year in YEARS:
        for single_day_posts, filepath in open_csv_files_from_path(PATHS[year]):
            all_elements_by_count = get_most_common_tokens_from_column(single_day_posts, column_name='used_id')
            filename = os.path.splitext(os.path.basename(filepath))[0]
            month = filename.split('-')[1]
            month_key = f'{year}-{month}'
            print(users_count)
            # print(len(all_elements_by_count))
            # input()

    pass

count_unique_users_by_month()
# for year in YEARS:
#     for single_day_posts, filepath in open_csv_files_from_path(PATHS[year]):
#         pass
