import pandas as pd
from utilities import get_df_from_path
from consts import FRAMES, PATHS, YEARS


def create_dataframe_from_path(path):
    final_dataframe = pd.concat(get_df_from_path(path), axis=0, ignore_index=True)
    return final_dataframe


def prepare_all_data_frames():
    for year in YEARS:
        frame = create_dataframe_from_path(PATHS[year])
        frame.to_csv(FRAMES[year], index=False, encoding="utf-8")


if __name__ == '__main__':
    prepare_all_data_frames()
