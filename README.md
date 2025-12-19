## NFL Keys to Success

This project investigates **what drives offensive success in the NFL** by predicting whether a play will be successful using **pre-snap information only**.

By restricting the feature space to pre-snap context (formation, personnel, down, distance, field position, etc.), this work isolates the **decision-making signal** available to coaches before the ball is snapped, framing football as a strategic, information-constrained problem rather than a purely outcome-driven one.

This repository contains the **full data pipeline, feature engineering, model training, hyperparameter tuning, and evaluation framework** used to support the analysis and results presented in the accompanying research paper.

## Research Motivation

Modern football analytics often rely on post-snap outcomes or player tracking data, which obscures the strategic decisions made **before** a play begins.

This project asks:

> *How predictable is offensive success using only information known prior to the snap?*

By answering this question, we aim to:
- Quantify the value of pre-snap context
- Identify which situational factors most strongly influence success
- Provide interpretable insights that could inform play-calling decisions

## Methodology Overview

The project follows a structured machine learning workflow:

1. **Data Acquisition**  
   NFL play-by-play and participation data sourced via the `nflreadr` and `nflfastR` ecosystem (2016–2023 seasons).

2. **Feature Engineering**  
   Construction of pre-snap features including:
   - Down, distance, and field position
   - Personnel groupings and formations
   - Temporal and contextual indicators
   - Team-level and player-level aggregates

3. **Modeling Approaches**
   - Logistic Regression (baseline, interpretability)
   - Random Forest
   - Gradient Boosting (XGBoost)
   - Ensemble methods

4. **Evaluation**
   - Season-based train/test splits
   - Out-of-sample evaluation on future seasons
   - Comparison across model classes

5. **Hyperparameter Optimization**
   - Bayesian optimization via Optuna

## Key Findings

- Pre-snap features alone provide meaningful predictive power for play success
- Down and distance dominate performance, but personnel and formation features add incremental value
- Tree-based models outperform linear baselines, though at the cost of interpretability
- Ensemble approaches offer marginal gains over single best models

Detailed analysis, metrics, and interpretations are presented in the accompanying research paper.

## Reproducibility & Usage
### Obtaining Data
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

### Data Cleaning and Featuring

First things first we need to install our requirements. Run the following commands:

Mac:
```Bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Windows:
```Bash
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

In bash, navigate to the project folder using cd where you replace path\to\ with your relevant path.

```Bash
cd path\to\eas_508_project
```
Once you are in the correct folder, run the following code in bash to build the feature engineered csv files:

```Bash
python -m src.data.build_data
```
You may have to replace python with py, python3, or some variation that your device specificially is using.

### Training, Evaluating, and Tuning Models

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
--model_path -> models/xgboost_optuna.joblib
--years -> 2023
```

If you would like to predict values or tune a model using optuna, changing some feature or updating some feature list, we also have the following code:

```Bash
python -m src.models.tune --model MODEL --years YEARS [--data DATA] [--trials TRIALS]
python -m src.models.predict --model_path MODEL_PATH --years YEARS [--data DATA]
```

### Recommended Code for Best Runs

Here is an easy copy paste of all codes to run and saves predictions vs ground truth in outputs/predictions/

```Bash
python -m src.data.build_data
python -m src.models.tune --model xgboost --years 2016 2017 2018 2019 2020 2021 2022
python -m src.models.evaluate --model_path models/xgboost_optuna.joblib --years 2016 2017 2018 2019 2020 2021 2022
python -m src.models.evaluate --model_path models/xgboost_optuna.joblib --years 2023
python -m src.models.predict --model_path models/xgboost_optuna.joblib --years 2023 --save
```

You can also use this code to run the ensemble method, the final line will save outputs if you wish to merge these and compare them in our plots and feature importance notebooks.

```Bash
python -m src.data.build_data
python -m src.models.tune --model logistic --years 2016 2017 2018 2019 2020
python -m src.models.tune --model xgboost --years 2016 2017 2018 2019 2020
python -m src.models.tune --model random_forest --years 2016 2017 2018 2019 2020
python -m src.models.tune --model ensemble --years 2021 2022
python -m src.models.evaluate --model_path models/ensemble_optuna.joblib --years 2023
python -m src.models.predict --model_path models/xgboost_optuna.joblib --years 2023 --save
```