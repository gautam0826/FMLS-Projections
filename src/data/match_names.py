import os
import pandas as pd

df1 = pd.read_csv(os.path.join('..', '..', 'data', 'processed', '2018_player_info.csv'), sep=',', encoding='ISO-8859-1')
df2 = pd.read_csv(os.path.join('..', '..', 'data', 'processed', '2017_player_info.csv'), sep=',', encoding='ISO-8859-1')

df3 = pd.merge(df1, df2, how='outer', on=['player name'])
df3.to_csv(os.path.join('..', '..', 'data', 'interim', 'name_matching_automated_attempt.csv'), sep=',', index=False)