import numpy as np
import pandas as pd
import glob
import sys
from datetime import datetime
import yaml

import warnings
warnings.simplefilter(action='ignore', category=pd.errors.PerformanceWarning)
warnings.simplefilter(action='ignore', category=pd.errors.DtypeWarning)


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
    df = df.copy()

    df['extra_lineman_flag'] = np.where(df['OL'] > 5, 1, 0)
    df['bigs'] = df['OL']+df['TE']-5
    df['personnel_num'] = df.apply(lambda x: f'{x['bigs']}{x['RB']}', axis=1)
    df = df.copy()

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
    df = df[(win_perc < df['wp']) & (df['wp'] < (1-win_perc))]
    df = df.copy()

#Add ydstoscucces
def addtl_features(dataframe):
    df = dataframe.copy()

    #Yards to success, a combination of down and yards
    df['ydstosuccess'] = np.where(df['down']==1, df['ydstogo']*.5, np.where(df['down']==2, df['ydstogo']*.7, df['ydstogo']))

    #Define previous success based on previous plays success
    df['previous_success'] = df.groupby(['nflverse_game_id','fixed_drive'])['success'].shift(1)
    df['previous_success'] = df['previous_success'].fillna(0)
    df = df.copy()

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
    df = df.copy()

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
    df = df.copy()

    return df

def engineer_features(cfg):
    toggles = cfg["features"]["toggles"]
    vals = cfg["features"]["values"]

    filename = cfg['file']
    input_dir = cfg['input_dir']
    output_dir = cfg['output_dir']
    log_dir = cfg['log_dir']

    read_file = input_dir+filename
    try:
        print(f"📂 Loading: {read_file}")
        df = pd.read_csv(read_file, low_memory=False)
        print("✅ All files loaded!")
    except:
        print('❌ Files not loaded, please try again')
        sys.exit()

    if toggles.get("build_off", False):
        print(f'🏈 Building Offense Personnel Counts')
        df = fix_personnel(df)
        df = df.copy()
    if toggles.get("build_addtl", False):
        print(f'🏈 Building Offense Addtl Features')
        df = addtl_features(df)
        df = df.copy()
    if toggles.get("build_fp_succ", False):
        print(f'🏈 Building Formation Personnel Success and EPA')
        df = formation_personnel_success(df)
        df = df.copy()
    if toggles.get("build_f_succ", False):
        print(f'🏈 Building Formation Success and EPA')
        df = formation_success(df)
        df = df.copy()
    if vals.get("win_trim", 0):
        print(f'🏈 Trimming plays based on WP: {vals.get("win_trim")}')
        df = win_trim(df)
        df = df.copy()

    try:
        timestamp = datetime.now().strftime("%m_%d")
        out_file = f"featured_{timestamp}.csv"
        output_path = output_dir+out_file
        df.to_csv(output_path, index=False)
        print(f"💾 Saved {out_file} to: {output_dir}")
    except:
        print("Unable to save CSV to output path specified")
    

if __name__ == "__main__":
    with open("cfg/features.yml") as f:
        cfg = yaml.safe_load(f)

    engineer_features(cfg)