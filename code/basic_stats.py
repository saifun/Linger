import pandas as pd
from utilities import get_most_common_tokens_from_column
from consts import FRAMES, MONTHS, YEARS


def print_word_by_month(word):
    for year in YEARS:
        frame_df = pd.read_csv(FRAMES[year])
        for month in MONTHS:
            filtered_df = frame_df[frame_df["created_at"].str.split('-').str[1] == month]
            all_elements_by_count = get_most_common_tokens_from_column(filtered_df, "text")
            print(str(year) + "-" + month + ": " + str(
                list(filter(lambda element: element[0] == word, all_elements_by_count))))


print_word_by_month("להשים")
