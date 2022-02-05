DATA_PATH = '/Users/saifun/Documents/HUJI/3 semester/67978_Needle_in_a_Data_Haystack/final_project/twitter/hebrew_twitter/{}'

YEARS = list(range(2018, 2022))

PATHS = {
    year: DATA_PATH.format('twitter_data_' + str(year))
    for year in YEARS
}

FRAMES = {
    year: DATA_PATH.format('frame_' + str(year) + '.csv')
    for year in YEARS
}

MONTHS = ['{:02d}'.format(month) for month in range(1, 13)]

RECORDS_COUNT_STATS = {
    2018: 15523736,
    2019: 23823005,
    2020: 52823817,
    2021: 13168095,
}
