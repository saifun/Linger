import os
import pandas as pd
from collections import defaultdict
from utilities import get_most_common_tokens_from_column, open_csv_files_from_path
from consts import YEARS, PATHS


def count_unique_users_by_month():

    users_count = {}
    for year in YEARS:
        for single_day_posts, filepath in open_csv_files_from_path(PATHS[year]):
            # all_elements_by_count = get_most_common_tokens_from_column(single_day_posts, column_name='used_id')
            filename = os.path.splitext(os.path.basename(filepath))[0]
            month = filename.split('-')[1]
            month_key = f'{year}-{month}'
            if month_key not in users_count:
                users_count[month_key] = {}
            for id in list(single_day_posts["used_id"]):
                if id not in users_count[month_key]:
                    users_count[month_key][id] = 0
                users_count[month_key][id]+=1
            # print(len(users_count[month_key]))
            # print(len(all_elements_by_count))
            # input()
    values = []
    for month_key in users_count.keys():
        values.append((month_key,len(users_count[month_key]) ))
    users_df = pd.DataFrame(values, columns=['month', 'count'])
    users_df.to_csv(f'results/meta_data_count/users_count.csv', index=False, header=True)


count_unique_users_by_month()
# for year in YEARS:
#     for single_day_posts, filepath in open_csv_files_from_path(PATHS[year]):
#         pass
