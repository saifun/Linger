import pandas as pd
import glob
from utilities import get_most_common_tokens_from_column

path = r'/Users/mariatseytlin/Documents/Msc/Needle in Data Haystack/project/twitter_data_2021'
all_files = glob.glob(path + "/*.csv")

li = []

for filename in all_files[:2]:
    try:
        df = pd.read_csv(filename, encoding='utf-8')
        li.append(df)
    except:
        print(filename)

frame = pd.concat(li, axis=0, ignore_index=True)
# fram = pd.DataFrame(li, columns=['text', 'tweet_id', 'created_at', 'user_name', 'user_screen_name', 'used_id'])

all_elements_by_count = get_most_common_tokens_from_column(frame, "text")
print(list(filter(lambda element: 10 < element[1] < 100, all_elements_by_count)))
# text = frame["text"]
#
# for row in text:
#     print(row)
