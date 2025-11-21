import numpy as np
import pandas as pd
import glob
import sys
import re
from rapidfuzz import process, fuzz

import warnings
warnings.simplefilter(action='ignore', category=pd.errors.PerformanceWarning)
warnings.simplefilter(action='ignore', category=pd.errors.DtypeWarning)

MADDEN_PATH = "data/interim/madden_ratings.csv"
PLAYER_PATH = "data/raw/players/nfl_players_2016_2024.csv"
OUT_PATH = "data/processed/matched_ratings.csv"
MADDEN_TO_UNI = {
    "QB": "QB",
    "HB": "RB",
    "FB": "RB",
    "RB": "RB",
    "WR": "WR",
    "TE": "TE",
    "LT": "OL",
    "LG": "OL",
    "C":  "OL",
    "RG": "OL",
    "RT": "OL",
    "LE": "DL",
    "RE": "DL",
    "DT": "DL",
    "RE LE": "DL",
    "RLE": "DL",
    "LOLB": "LB",
    "ROLB": "LB",
    "MLB": "LB",
    "LB": "LB",
    "CB": "DB",
    "SS": "DB",
    "FS": "DB"
}
NFL_TO_UNI = {
    "QB": "QB",
    "RB": "RB",
    "HB": "RB",
    "FB": "RB",
    "WR": "WR",
    "LS": "TE",
    "TE": "TE",
    "OT":"OL",
    "T": "OL",
    "G": "OL",
    "C": "OL",
    "OL": "OL",
    "DE": "DL",
    "DT": "DL",
    "NT": "DL",
    "DL": "DL",
    "LB": "LB",
    "OLB": "LB",
    "ILB": "LB",
    "MLB": "LB",
    "CB": "DB",
    "DB": "DB",
    "FS": "DB",
    "SS": "DB",
    "SAF": "DB",
    "S":"DB",
    "LS": None,
    "K": None,
    "P": None
}


def normalize_name(name: str) -> str:
    if pd.isna(name):
        return ""
    name = name.lower()
    name = re.sub(r"\b(jr|sr|ii|iii|iv|v)\b", "", name)
    name = re.sub(r"[^a-z\s-]", "", name)
    name = name.replace("-", " ").replace("'", "")
    name = re.sub(r"\s+", " ", name)
    return name.strip()

def fuzzy_match_player(row, players_df):
    """Try fuzzy matching first within same position, otherwise all players."""
    name = row["clean_name"]
    pos  = row["position"]

    # filter by position first
    candidates = players_df[players_df["position"] == pos]["clean_name"]

    # fallback: ignore position if empty
    if candidates.empty:
        candidates = players_df["clean_name"]

    match = process.extractOne(
        name,
        candidates,
        scorer=fuzz.WRatio,
        score_cutoff=75
    )

    if match:
        matched_name = match[0]
        return players_df.loc[
            players_df["clean_name"] == matched_name, "player_id"
        ].iloc[0]

    return None

def clean():
    players['clean_name'] = players['full_name'].apply(normalize_name)
    madden['clean_name'] = madden['name'].apply(normalize_name)
    madden["std_pos"] = madden["position"].map(MADDEN_TO_UNI)
    players["std_pos"] = players["position"].map(NFL_TO_UNI)

    print("🔎 MATCHING!")

    merged = madden.merge(
        players[['clean_name','std_pos','player_id']],
        how='left',
        left_on = ['clean_name','std_pos'],
        right_on = ['clean_name','std_pos']
    )
    still_missing = merged[merged["player_id"].isna()].copy()
    still_missing["player_id"] = still_missing.apply(
        lambda row: fuzzy_match_player(row, players),
        axis=1
    )
    merged.loc[merged["player_id"].isna(), "player_id"] = still_missing["player_id"]
    players_with_ratings = players.merge(merged[['player_id','year','ovr']],
                                         how='left',
                                         on='player_id').drop(columns=['std_pos','first_name','last_name','clean_name'])
    players_with_ratings['year'] = players_with_ratings['year'].astype("Int64")
    players_with_ratings['ovr'] = players_with_ratings['ovr'].astype("Int64")
    players_with_ratings['position'] = players_with_ratings['position'].map(NFL_TO_UNI)
    players_with_ratings = players_with_ratings[players_with_ratings["position"].notna()]

    print("🔎 SUMMARY AFTER MATCHING")
    print("Total Madden rows:", len(madden))
    print("Exact matches:", merged["player_id"].notna().sum())

    return players_with_ratings

if __name__ == "__main__":
    try:
        print(f"📂 Loading: {MADDEN_PATH}")
        madden = pd.read_csv(MADDEN_PATH)
        print("✅ File loaded!")
    except:
        print('❌ Madden file not loaded, please try again')
        sys.exit()
    try:
        print(f"📂 Loading: {PLAYER_PATH}")
        players = pd.read_csv(PLAYER_PATH)
        print("✅ File loaded!")
        print(len(players['player_id'].unique()))
    except:
        print('❌ Player file not loaded, please try again')
        sys.exit()

    players_with_ratings = clean()

    try:
        players_with_ratings.to_csv(OUT_PATH, index=False)
        print(f"💾 Saved players_with ratings to: {OUT_PATH}")
    except:
        print("Unable to save CSV to output path specified")


