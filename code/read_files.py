import pandas as pd
import glob
from utilities import get_most_common_tokens_from_column

path_2021 = r'/Users/mariatseytlin/Documents/Msc/Needle in Data Haystack/project/twitter_data_2021'
path_2020 = r'/Users/mariatseytlin/Documents/Msc/Needle in Data Haystack/project/twitter_data_2020'
path_2019 = r'/Users/mariatseytlin/Documents/Msc/Needle in Data Haystack/project/twitter_data_2019'
path_2018 = r'/Users/mariatseytlin/Documents/Msc/Needle in Data Haystack/project/twitter_data_2018'

months = ['01', '02', '03', '04', '05', '06', '07', '08', '09', '10', '11', '12']

paths = [path_2021, path_2020, path_2019, path_2018]


def create_dataframe_from_path(path):
    all_files = glob.glob(path + "/*.csv")
    li = []
    for filename in all_files:
        try:
            df = pd.read_csv(filename, encoding='utf-8')
            li.append(df)
        except:
            print(filename)
    final_dataframe = pd.concat(li, axis=0, ignore_index=True)
    return final_dataframe


def print_boomer_by_year():
    year = 2021
    for path in paths:
        frame = create_dataframe_from_path(path)
        all_elements_by_count = get_most_common_tokens_from_column(frame, "text")
        print(str(year) + ": " + str(list(filter(lambda element: element[0] == 'בומר', all_elements_by_count))))
        year -= 1

# print_boomer_by_year()

def print_boomer_by_month():
    year = 2021
    for path in paths:
        frame = create_dataframe_from_path(path)
        frame.to_csv("/Users/mariatseytlin/Documents/Msc/Needle in Data Haystack/project/frame_" + str(year) + ".csv", index=False, encoding="utf-8")
        for month in months:
            filtered_df = frame[frame["created_at"].str.split('-').str[1] == month]
            all_elements_by_count = get_most_common_tokens_from_column(filtered_df, "text")
            print(str(year) + "-" + month + ": " + str(list(filter(lambda element: element[0] == 'בומר', all_elements_by_count))))
        year -= 1

print_boomer_by_month()

# fram = pd.DataFrame(li, columns=['text', 'tweet_id', 'created_at', 'user_name', 'user_screen_name', 'used_id'])

# print(list(filter(lambda element: 100 < element[1] < 500, all_elements_by_count)))

# text = frame["text"]
#
# for row in text:
#     print(row)
