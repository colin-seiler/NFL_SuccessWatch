Data should all be sourced from NFLVerse package in R using the nflreadr library. Our data will be build from the load_participation functions sourcing data from 2016 through 2024.

You can use the following SCRIPT in R to access the needed CSV files
```R
install.packages(c("nflreadr", "nflfastR", "nflplotR", "nfl4th"))

library(nflreadr)

# Check for pbp_data folder and create a folder to store the data if DNE
if (!dir.exists("pbp_data")) {
  dir.create("pbp_data")
}

years <- 2016:2024

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

In bash, navigate to the project folder using cd where you replace path\to\ with your relevant path.

```Bash
cd path\to\eas_508_project
```
Once you are in the correct folder, run the following code in bash:

```Bash
python src/data_manip/cleaning_pbp.py 
python src/data_manip/cleaning_ratings.py
python src/data_manip/combine_madden_ratings.py
python src/data_manip/extract_features.py
python src/data_manip/create_pipeline
```

Use pipeline.yml to update which features to use, categorize, and scale. Enter model type as well to make sure features are presented correctly for each model.

You can then load the interim dataset and perform more data engineering to prepare it for models or to add more features. Working features will be added to the feature_extraction.py in the future.

We need to use Divide and Conquer with differing models to provide a proper prediction, our current predictions have not come close enough to working