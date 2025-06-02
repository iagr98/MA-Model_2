"""
extract flooding point from result subfolder
"""
import os
import pandas as pd
import numpy as np

# load all files in results subfolder
results = os.listdir('results')
flooding_points = []
# iterate over all files and extract flooding point
for result in results:
    if os.path.isdir('results/' + result):
        # load total_volume_flow_flood.csv
        df = pd.read_csv('results/' + result + '/total_volume_flow_flood.csv')
        # extract flooding point
        flooding_point = df.iloc[0,1]*3600*1000
        flooding_points.append(flooding_point)
        print(flooding_point)
        # save flooding point in csv
    else:
        print('No results subfolder found')
        break
    
# create DataFrame and save as csv
df = pd.DataFrame(flooding_points, columns=['flooding_point'])
# add filename as column
df['filename'] = results
df.to_csv('flooding_points.csv')