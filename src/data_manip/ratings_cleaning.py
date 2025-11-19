import numpy as np
import pandas as pd
import openpyxl
import glob
import sys
from datetime import datetime

import warnings
warnings.simplefilter(action='ignore', category=pd.errors.PerformanceWarning)
warnings.simplefilter(action='ignore', category=pd.errors.DtypeWarning)

IN_PATH = "data/raw/madden/"
OUT_PATH = "data/interim/madden/"
PBP = "/*.xlsx"

def load_files():
        years = []
        dfs = []
        for file in sorted(glob.glob(IN_PATH+PBP)):
            print(f"📂 Loading: {file}")
            temp = pd.read_excel(file)
            year = file.strip().split('.')[0][-2:]
            years.append(year)
            dfs.append(temp)

        print("✅ All files loaded!")
        return dfs, years
    # except: 
    #     print('❌ Files not loaded, please try again')
    #     sys.exit()

def clean_data(files):
    temp_files = []
    for file in files:
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
        
        temp_files.append(temp_df)
    
    return temp_files
    
        
if __name__ == "__main__":
    dfs, years = load_files()
    out_dfs = clean_data(dfs)
    for idx, out in enumerate(out_dfs):
        file_out = OUT_PATH+f'madden{years[idx]}.csv'
        out.to_csv(file_out, index=False)
        print(f"💾 Saved {file_out} to: {OUT_PATH}")



    