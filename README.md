## NFL Keys to Success

This is a project focused on predicting Success on any given play given only pre-snap features

## Obtaining Data
Data should all be sourced from NFLVerse package in R using the nflreadr library. Our data will be build from the load_participation functions sourcing data from 2016 through 2024.

You can use the following SCRIPT in R to access the needed CSV files
```R
install.packages(c("nflreadr", "nflfastR", "nflplotR", "nfl4th"))

library(nflreadr)

# Check for pbp_data folder and create a folder to store the data if DNE
if (!dir.exists("pbp_data")) {
  dir.create("pbp_data")
}

years <- 2016:2023

for (year in years) {
  pbp_data <- load_participation(seasons = TRUE, include_pbp = TRUE)

  assign(paste0("pbp_", year), pbp_data)
  
  # Save as CSV file
  filename <- paste0("pbp_data/pbp_", year, ".csv")
  write.csv(pbp_data, filename, row.names = FALSE)
  
  cat("Saved:", filename, "\n\n")
}
```
Once the CSVs are downloaded, add them to the data/raw/pbp_data folder as individual files.

We have provided the remaining smaller files for madden ratings and team statistics.

## Data Cleaning and Featuring

In bash, navigate to the project folder using cd where you replace path\to\ with your relevant path.

```Bash
cd path\to\eas_508_project
```
Once you are in the correct folder, run the following code in bash to build the feature engineered csv files:

```Bash
python -m src.data.build_data
```
You may have to replace python with py, python3, or some variation that your device specificially is using.

## Training, Evaluating, and Tuning Models

From here we just have to train, tune, and evaluate our models. For tuning we've used Optuna which is located in the requirements.txt file. You can run the following code to train using the default values provided in the config

```Bash
python -m src.models.train --model MODEL --years YEARS [--save_dir SAVE_DIR]
```
Where [argument] is not required but optional. The values recommended for use in each argument are the following:

```Bash
--model -> logistic, random_forest, xgboost
--years -> 2016 2017 2018 2019 2020 2021 2022
--save_dir -> models
```

In order to evaluate our model we can run the following code:

```Bash
python -m src.models.evaluate --model_path MODEL_PATH --years YEARS [--data DATA]
```

The values recommended for use in each argument are the following:

```Bash
--model_path -> model/xgboost_optuna
--years -> 2023 2024
--data -> data/processed/processed_all.csv
```

If you would like to predict values or tune a model using optuna, changing some feature or updating some feature list, we also have the following code:

```Bash
python -m src.models.tune --model MODEL --years YEARS [--data DATA] [--trials TRIALS]
python -m src.models.predict --model_path MODEL_PATH --years YEARS [--data DATA]
```

## Recommended Code for Best Runs

Here is an easy copy past of all codes to run and saves predictions vs ground truth in outputs/predictions/

```Bash
python -m src.data.build_data
python -m src.models.tune --model xgboost --years 2016 2017 2018 2019 2020 2021 2022
python -m src.models.evaluate --model_path model/xgboost_optuna --years 2016 2017 2018 2019 2020 2021 2022
python -m src.models.evaluate --model_path model/xgboost_optuna --years 2023
python -m src.models.predict --model_path model/xgboost_optuna --years 2023 --save
```

You can then combine this output with the processed_all.csv to get predictions and see how different features affected the model output