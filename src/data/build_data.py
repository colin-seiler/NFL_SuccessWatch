import argparse

from src.data.pbp import clean_data as pbp_clean
from src.data.ratings import clean_data as ratings_clean
from src.data.combine_ratings import clean as ratings_combine
from src.data.features import engineer_features

def build_data(madden = False):
    pbp_clean()
    if madden == True:
        ratings_clean()
        ratings_combine()
    engineer_features()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Build All Data Files")

    parser.add_argument("--enable_madden", action='store_true', help='Enable building madden tables from madden spreadsheets', required=False)

    args = parser.parse_args()

    if args.enable_madden:
        print('Loading, Cleaning, and Featuring all PBP and Madden Data')
    else:
        print('Loading, Cleaning, and Featuring PBP Data only')

    build_data(args.enable_madden)
