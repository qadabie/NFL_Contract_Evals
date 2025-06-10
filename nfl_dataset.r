# Install devtools first if not already installed
if (!requireNamespace("devtools", quietly = TRUE)) {
    install.packages("devtools")
}
devtools::install_github("Ginsburg1/ProSportsDraftData")

# Install arrow package if not already installed
if (!requireNamespace("arrow", quietly = TRUE)) {
    install.packages("arrow")
}

library(ProSportsDraftData)
library(arrow)

# Get the data
nfl_data <- nfl_data_walter_football()

# Save as parquet
write_parquet(nfl_data, "nfl_draft_data.parquet")