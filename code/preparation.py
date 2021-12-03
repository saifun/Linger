import glob
import pandas as pd
from consts import FRAMES, PATHS, YEARS


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


def prepare_all_data_frames():
    for year in YEARS:
        frame = create_dataframe_from_path(PATHS[year])
        frame.to_csv(FRAMES[year], index=False, encoding="utf-8")


if __name__ == '__main__':
    prepare_all_data_frames()
