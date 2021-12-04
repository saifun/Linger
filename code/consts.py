# DATA_PATH = '/Users/saifun/Documents/HUJI/3 semester/67978_Needle_in_a_Data_Haystack/final_project/twitter/hebrew_twitter/{}'
DATA_PATH = r'C:\Users\roeis\Dropbox\study\master\term1\needle\project\{}'

YEARS = list(range(2018, 2022))
# YEARS = list(range(2020, 2022))

PATHS = {
    # year: DATA_PATH.format('twitter_data_' + str(year))
    year: DATA_PATH.format('twitter_data_' + str(year) + '/data_' + str(year))
    for year in YEARS
}

FRAMES = {
    year: DATA_PATH.format('frame_' + str(year) + '.csv')
    for year in YEARS
}

MONTHS = ['{:02d}'.format(month) for month in range(1, 13)]
