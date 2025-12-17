import numpy as np
import pandas as pd
import glob
import sys
from datetime import datetime

import warnings
warnings.simplefilter(action='ignore', category=pd.errors.PerformanceWarning)
warnings.simplefilter(action='ignore', category=pd.errors.DtypeWarning)

IN_PATH = "data/raw/madden/"
OUT_PATH = "data/interim/"
PBP = "/*.xlsx"

def load_files():
    try:
        dfs = []
        for file in sorted(glob.glob(IN_PATH+PBP)):
            print(f"📂 Loading: {file}")
            temp = pd.read_excel(file)
            year = file.strip().split('.')[0][-2:]
            dfs.append([year, temp])

        print("✅ All files loaded!")
        return dfs
    except: 
        print('❌ Files not loaded, please try again')
        sys.exit()

def clean_data():
    files = load_files()
    
    temp_files = []

    for item in files:
        year = item[0]
        file = item[1]

        temp_df = pd.DataFrame()
        if 'Team' in file.columns:
            temp_df['team'] = file['Team']

        if 'Full Name' in file.columns:
            temp_df['name'] = file['Full Name']
        elif 'Name' in file.columns:
            temp_df['name'] = file['Name']
        else:
            if 'First Name' in file.columns:
                temp_df['name'] = file['First Name'] + ' ' + file['Last Name']
            else:
                temp_df['name'] = file['FirstName'] + ' ' + file['LastName']
        
        if 'Position' in file.columns:
            temp_df['position'] = file['Position']

        if 'Overall Rating' in file.columns:
            temp_df['ovr'] = file['Overall Rating']
        elif 'OverallRating' in file.columns:
            temp_df['ovr'] = file['OverallRating']
        elif 'OVR' in file.columns:
            temp_df['ovr'] = file['OVR']
        else:
            temp_df['ovr'] = file['Overall']

        temp_df['year'] = '20'+str(int(year)-1)
        
        temp_files.append(temp_df[['year','name','team','position','ovr']])
    
    out = pd.DataFrame()
    for item in temp_files:
        out = pd.concat([out, item])
    file_out = OUT_PATH+f'madden_ratings.csv'
    out.to_csv(file_out, index=False)
    print(f"💾 Saved madden_ratings.csv to: {OUT_PATH}\n")
    
        
if __name__ == "__main__":
    clean_data()