import numpy as np
import pandas as pd
import glob
import sys
from datetime import datetime

import warnings
warnings.simplefilter(action='ignore', category=pd.errors.PerformanceWarning)
warnings.simplefilter(action='ignore', category=pd.errors.DtypeWarning)


IN_PATH = "data/raw/pbp_data/"
OUT_PATH = "data/cleaned/pbp/"
PBP = "/*.csv"

def clean_data(off_personnel=True, add_features=True, off_form=True, off_form_personnel=True, wp_trim=0, input_path=IN_PATH+PBP):
    try:
        dfs = []
        for file in sorted(glob.glob(input_path)):
            print(f"📂 Loading: {file}")
            temp = pd.read_csv(file, low_memory=False)
            dfs.append(temp)

        df = pd.concat(dfs, ignore_index=True)
        print("✅ All files loaded!")
        shape = df.shape
        print(f'\nDataFrame Size: {shape} \n')
    except:
        print('❌ Files not loaded, please try again')
        sys.exit()

    #DROP NA DATA
    formation = df['offense_formation'].notna()
    personnel = df['offense_personnel'].notna()
    no_play = df['play_type'] != 'no_play'
    kneel = df['qb_kneel'] == 0
    spike = df['qb_spike'] == 0
    pt2 = df['down'].notna()

    df = df[pt2 & formation & personnel & no_play & kneel & spike].drop(columns=['qb_kneel','qb_spike'])

    if off_personnel == True:
        df = fix_personnel(df)
        df = df.copy()
    if add_features == True:
        df = addtl_features(df)
        df = df.copy()
    if off_form_personnel == True:
        df = formation_personnel_success(df)
        df = df.copy()
    if off_form == True:
        df = formation_success(df)
        df = df.copy()
        
    df = win_trim(df)
    df = df.copy()

    try:
        timestamp = datetime.now().strftime("%m_%d")
        out_file = f"clean_{timestamp}.csv"
        output_path = OUT_PATH+out_file
        df.to_csv(output_path)
        print(f"✅ Saved {out_file} to: {OUT_PATH}")
    except:
        print("Unable to save CSV to output path specified")

    
    #FIX PERSONNEL
def fix_personnel(dataframe):
    df = dataframe.copy()

    personnel_dict = {
        'OL':'OL',
        'C':'OL',
        'G':'OL',
        'T':'OL',
        'FB':'RB',
        'RB':'RB',
        'QB':'QB',
        'WR':'WR',
        'TE':'TE',
    }

    df['personnel_dict'] = df['offense_personnel'].apply(lambda x: {personnel_dict.get(pos): int(count) for count, pos in (p.strip().split(' ') for p in x.split(','))})
    df['personnel_dict'] = df['personnel_dict'].fillna('OTHER')

    df['QB'] = 1
    df['OL'] = 5

    personnel_titles = ['RB','TE','WR','OTHER']

    for pos in personnel_titles:
        df[f'{pos}'] = 0

    df = df.apply(update_personnel_counts, axis=1)
    df = df.drop(columns = ['personnel_dict','offense_personnel'])

    df['extra_lineman_flag'] = np.where(df['OL'] > 5, 1, 0)
    df['bigs'] = df['OL']+df['TE']-5
    df['personnel_num'] = df.apply(lambda x: f'{x['bigs']}{x['RB']}', axis=1)

    return df



#Helper Function to Fix Counts
def update_personnel_counts(row):
    personnel_titles = ['RB','TE','WR','OTHER']

    for k,v in row['personnel_dict'].items():
        if k in personnel_titles:
            row[k] += v
        else:
            if k == 'QB':
                row[k] = v
            elif k == 'OL':
                row[k] = v
            else:
                row['OTHER'] += v
    return row

#Trim DataFrame by certain win_perc
def win_trim(dataframe, win_perc=0):
    df = dataframe.copy()
    return df[(win_perc < df['wp']) & (df['wp'] < (1-win_perc))]

#Add ydstoscucces
def addtl_features(dataframe):
    df = dataframe.copy()

    #Yards to success, a combination of down and yards
    df['ydstosuccess'] = np.where(df['down']==1, df['ydstogo']*.5, np.where(df['down']==2, df['ydstogo']*.7, df['ydstogo']))

    #Define previous success based on previous plays success
    df['previous_success'] = df.groupby(['nflverse_game_id','fixed_drive'])['success'].shift(1)
    df['previous_success'] = df['previous_success'].fillna(0)

    return df

def formation_personnel_success(dataframe):
    df = dataframe.copy()

    succ = (
        df.groupby(['offense_formation', 'personnel_num'])
        .agg(
            plays=('success', 'count'),
            fp_success=('success', 'mean'),
            fp_epa=('epa', 'mean')
        )
    .reset_index()
    )

    global_mean = df['success'].mean()
    succ['smoothed_fp_success'] = (
        (succ['plays'] * succ['fp_success'] + 500 * global_mean)
        / (succ['plays'] + 500)
    )

    df = df.merge(succ[['offense_formation','personnel_num','fp_success','smoothed_fp_success','fp_epa']], how='left', on=['offense_formation','personnel_num'])

    return df

def formation_success(dataframe):
    df = dataframe.copy()

    f_success = (
        df.groupby('offense_formation')
        .agg(
            plays=('success', 'count'),
            f_success=('success', 'mean'),
            f_epa=('epa', 'mean')
        )
    .reset_index()
    )   

    global_success = df['success'].mean()

    f_success['smoothed_f_success'] = (
        (f_success['plays'] * f_success['f_success'] + 500 * global_success)
        / (f_success['plays'] + 500)
    )

    df = df.merge(f_success[['offense_formation','f_success','smoothed_f_success','f_epa']], how='left', on=['offense_formation'])

    return df

if __name__ == "__main__":
    clean_data()