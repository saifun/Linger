import pandas as pd
import glob
from utilities import get_most_common_tokens_from_column
import matplotlib.pyplot as plt
import statistics

path_2021 = r'/Users/mariatseytlin/Documents/Msc/Needle in Data Haystack/project/twitter_data_2021'
path_2020 = r'C:\Users\roeis\Dropbox\study\master\term1\needle\project\hebrew_tweets_2020\data_2020'
# path_2021 = r'/Users/mariatseytlin/Documents/Msc/Needle in Data Haystack/project/twitter_data_2021'
# path_2020 = r'/Users/mariatseytlin/Documents/Msc/Needle in Data Haystack/project/twitter_data_2020'
# path_2019 = r'/Users/mariatseytlin/Documents/Msc/Needle in Data Haystack/project/twitter_data_2019'
# path_2018 = r'/Users/mariatseytlin/Documents/Msc/Needle in Data Haystack/project/twitter_data_2018'
#
frame_2021 = r'C:\Users\roeis\Dropbox\study\master\term1\needle\project\frame_2021.csv'
frame_2020 = r'C:\Users\roeis\Dropbox\study\master\term1\needle\project\frame_2020.csv'
# frame_2021 = r'/Users/mariatseytlin/Documents/Msc/Needle in Data Haystack/project/frame_2021.csv'
# frame_2020 = r'/Users/mariatseytlin/Documents/Msc/Needle in Data Haystack/project/frame_2020.csv'
# frame_2019 = r'/Users/mariatseytlin/Documents/Msc/Needle in Data Haystack/project/frame_2019.csv'
# frame_2018 = r'/Users/mariatseytlin/Documents/Msc/Needle in Data Haystack/project/frame_2018.csv'

months = ['01', '02', '03', '04', '05', '06', '07', '08', '09', '10', '11', '12']

# paths = [path_2021, path_2020, path_2019, path_2018]
# frames = [frame_2021, frame_2020, frame_2019, frame_2018]
paths = [path_2021, path_2020]
frames = [frame_2021, frame_2020]


def create_dataframe_from_path(path):
    all_files = glob.glob(path + "/*.csv")
    li = []
    for filename in all_files:
        print(filename)
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
            # print(str(year) + "-" + month + ": " + str(list(filter(lambda element: element[0] == 'בומר', all_elements_by_count))))
        year -= 1

# print_boomer_by_month()

def print_word_by_month(word):
    year = 2021
    for frame in frames:
        frame_df = pd.read_csv(frame)
        for month in months:
            filtered_df = frame_df[frame_df["created_at"].str.split('-').str[1] == month]
            all_elements_by_count = get_most_common_tokens_from_column(filtered_df, "text")
            # print(str(year) + "-" + month + ": " + str(list(filter(lambda element: element[0] == word, all_elements_by_count))))
        year -= 1

# print_word_by_month("להשים")
# fram = pd.DataFrame(li, columns=['text', 'tweet_id', 'created_at', 'user_name', 'user_screen_name', 'used_id'])

# df = create_dataframe_from_path(r"C:\Users\roeis\Dropbox\study\master\term1\needle\project\hebrew_tweets_2021_upto_march20\data")
# # print(df)
# df.to_csv(frame_2021)
# print(list(filter(lambda element: 100 < element[1] < 500, all_elements_by_count)))

# text = frame["text"]
#
# for row in text:
#     print(row)

df = pd.read_csv(frame_2020,nrows=30000)
# user_dict = {}
# for user_name in list(df["user_name"]):
#     if user_name in user_dict:
#         user_dict[user_name]+=1
#     else:
#         user_dict[user_name] = 1
# print(len(user_dict.keys()))
#
words_dict = {}
for text in list(df["text"]):
    for word in text.split(" "):
        if word in words_dict:
            words_dict[word]+=1
        else:
            words_dict[word] = 1

# print(words_dict)
# print(statistics.median(words_dict.values()))
med = statistics.median(words_dict.values())
# print(med)
# print(df.keys())
# input()
common_words_dict = {key[::-1]:val for key, val in words_dict.items() if val >= 2000}
# uncommon_words_dict = {key[::-1]:val for key, val in words_dict.items() if val < 3}

plt.bar(common_words_dict.keys(), common_words_dict.values(), color='g')
plt.show()
# plt.bar(uncommon_words_dict.keys(), uncommon_words_dict.values(), color='g')
# plt.show()
