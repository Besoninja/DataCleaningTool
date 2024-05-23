# Data Cleaning Tool

This data cleaning tool is built to clean and sort messy data. The tool is modular in design, i.e., any or all parts of the tool can be run on your data. It has many different functions focusing on a particular aspect of the data cleaning process.

## Features

1. Import a messy CSV as a DataFrame called "DF".
2. Create an 'enhanced_info' table - gives a better view of the dataset and its missingness.
3. Handling columns with mixed data and Data Type Conversion.
4. Handling Missing Values:
   - Let the user decide what they want to do with the missing values.
   - Advanced imputation methods: mean, median, mode, linear regression, KNN, MICE, random forest, EM, Bayesian, hot deck, LOCF, NOCB.
5. Normalisation and scaling.
6. Filtering and Selecting Data.
7. Dealing with Duplicates.
8. Handling Outliers.
9. Data Transformation.
10. Feature Engineering.
11. Error Checking and Reporting.

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/Besoninja/DataCleaningTool.git
   cd DataCleaningTool
