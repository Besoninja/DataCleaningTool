import streamlit as st
import pandas as pd
import numpy as np

# Set page configuration
st.set_page_config(page_title="Data Cleaning Tool", layout="wide")

# Initialise session state variables for storing processed DataFrame and analysis results
if 'processed_df' not in st.session_state:
    st.session_state.processed_df = None
if 'incorrect_entries_analysis' not in st.session_state:
    st.session_state.incorrect_entries_analysis = None

# Title and description
st.title("Data Cleaning Tool")
st.markdown("""
This data cleaning tool is built to clean and sort messy data. 
It is modular in design, meaning you can choose which parts of the tool to run on your data.
""")

def generate_enhanced_information_table(dataframe):
    """
    Generate an enhanced overview table for a given DataFrame, including information such as:
    - Column Names
    - Data Types
    - Number of Unique Values
    - Null (missing) Value Counts
    - Percentage of Empty Cells per Column

    Parameters:
        dataframe (pd.DataFrame): The dataset for which to generate an information table.

    Returns:
        tuple: (info_dataframe, columns_with_missing)
               info_dataframe (pd.DataFrame): A summary table with column-level details.
               columns_with_missing (list): A list of tuples (column_name, dtype) for columns with missing values.

    Example usage:
        info_df, cols_missing = generate_enhanced_information_table(df)
    """
    info_data = {
        'ColumnName': [],
        'DataType': [], 
        'UniqueValues': [],
        'NullCount': [],
        '% EmptyCells': [],
    }
    
    columns_with_missing = []
    
    # Iterate through each column to gather statistics
    for column in dataframe.columns:
        null_count = dataframe[column].isnull().sum()
        unique_values_count = dataframe[column].nunique()
        column_type = dataframe[column].dtype.name
        empty_cells_percentage = (null_count / dataframe.shape[0]) * 100

        info_data['ColumnName'].append(column)
        info_data['DataType'].append(column_type)
        info_data['UniqueValues'].append(unique_values_count)
        info_data['NullCount'].append(null_count)
        info_data['% EmptyCells'].append(f'{empty_cells_percentage:.2f}%')
        
        # Keep track of columns that contain missing values for easy reference
        if null_count > 0:
            columns_with_missing.append((column, dataframe[column].dtype))

    info_dataframe = pd.DataFrame(info_data)
    return info_dataframe, columns_with_missing

def determine_column_data_types(dataframe, type_threshold=0.95):
    """
    Determine the predominant data type of each column in a DataFrame.

    The function attempts to classify each column as either 'numeric', 'string', or 'mixed'
    based on the proportion of entries that fit each type. The type_threshold parameter
    dictates the minimum proportion of entries that must be of a single type for the column
    to be classified as that type.

    Parameters:
        dataframe (pd.DataFrame): The dataset to classify.
        type_threshold (float): The proportion threshold for classifying a column's type.
                                Default is 0.95 (95%).

    Returns:
        dict: A dictionary mapping column names to determined data types ('numeric', 'string', 'mixed', 'empty').

    Example usage:
        column_types = determine_column_data_types(df, 0.95)
    """
    column_data_types = {}
    
    for column in dataframe.columns:
        string_count = 0
        numeric_count = 0
        other_count = 0
        
        # Assess each entry in the column
        for entry in dataframe[column]:
            if pd.isna(entry):
                # Ignore missing values since they don't help determine data type
                continue
            if isinstance(entry, str):
                # Attempt to parse the string as a number
                try:
                    float(entry)
                    numeric_count += 1
                except ValueError:
                    # If it cannot be parsed as a float, treat it as a string
                    string_count += 1
            elif isinstance(entry, (int, float, np.number)):
                numeric_count += 1
            else:
                # Other data types (e.g., datetime) could appear; treat as 'other'
                other_count += 1
        
        # Calculate the total number of non-missing entries
        total_entries = len(dataframe[column].dropna())
        
        # If no valid entries are present, treat the column as 'empty'
        if total_entries == 0:
            column_data_types[column] = 'empty'
            continue
            
        numeric_ratio = numeric_count / total_entries if total_entries > 0 else 0
        string_ratio = string_count / total_entries if total_entries > 0 else 0
        
        # Assign a predominant type if it exceeds the given threshold
        if numeric_ratio > type_threshold:
            column_data_types[column] = 'numeric'
        elif string_ratio > type_threshold:
            column_data_types[column] = 'string'
        else:
            # If no single type dominates, the column is 'mixed'
            column_data_types[column] = 'mixed'
            
    return column_data_types

def identify_incorrect_entries(dataframe):
    """
    Identify columns that contain a mixture of numeric and string entries, which may be considered incorrect.

    This function analyses each column to find those containing mixed data types (e.g., strings in a predominantly 
    numeric column or numeric values in a string column). It returns details needed to resolve these entries.

    Parameters:
        dataframe (pd.DataFrame): The dataset to analyse.

    Returns:
        dict: A dictionary where keys are column names and values are dictionaries with detailed counts and 
              indices of the incorrect entries (e.g., numeric_indices, string_indices).

    Example usage:
        analysis = identify_incorrect_entries(df)
    """
    analysis_results = {}
    
    for column in dataframe.columns:
        string_entries_indices = []
        numeric_entries_indices = []
        
        # Check each entry to classify it as numeric or string
        for idx, entry in enumerate(dataframe[column]):
            if pd.isna(entry):
                # Ignore missing values
                continue
            if isinstance(entry, str):
                # Check if the string can be interpreted as numeric
                try:
                    float(entry)
                    numeric_entries_indices.append(idx)
                except ValueError:
                    string_entries_indices.append(idx)
            elif isinstance(entry, (int, float, np.number)):
                numeric_entries_indices.append(idx)
        
        total_valid = len(string_entries_indices) + len(numeric_entries_indices)
        if total_valid == 0:
            # No usable entries to classify, skip this column
            continue
            
        string_ratio = len(string_entries_indices) / total_valid if total_valid > 0 else 0
        numeric_ratio = len(numeric_entries_indices) / total_valid if total_valid > 0 else 0
        
        # Only consider columns that have a meaningful mixture of string and numeric entries
        if 0 < string_ratio < 1 and 0 < numeric_ratio < 1:
            analysis_results[column] = {
                'string_count': len(string_entries_indices),
                'numeric_count': len(numeric_entries_indices),
                'string_indices': string_entries_indices,
                'numeric_indices': numeric_entries_indices,
                'string_ratio': string_ratio,
                'numeric_ratio': numeric_ratio
            }
    
    return analysis_results

def convert_mixed_data_types(dataframe, type_threshold=0.95):
    """
    Convert columns in the DataFrame to their predominant data type based on a threshold.

    For each column, if the majority of values are numeric or string by a certain threshold,
    convert that column to the identified type. This ensures more consistent column data.

    Parameters:
        dataframe (pd.DataFrame): The original dataset.
        type_threshold (float): The proportion threshold for classifying a column's type.
                                Default is 0.95 (95%).

    Returns:
        tuple: (cleaned_dataframe, conversion_report)
               cleaned_dataframe (pd.DataFrame): The DataFrame after conversions.
               conversion_report (list): A list of strings describing the changes made.

    Example usage:
        df_cleaned, report = convert_mixed_data_types(df, 0.95)
    """
    cleaned_dataframe = dataframe.copy()
    column_types = determine_column_data_types(dataframe, type_threshold)
    conversion_report = []
    
    for column, determined_type in column_types.items():
        original_type = dataframe[column].dtype
        if determined_type == 'numeric':
            # Attempt to coerce values to numeric; non-convertible values become NaN
            cleaned_dataframe[column] = pd.to_numeric(cleaned_dataframe[column], errors='coerce')
            if original_type != cleaned_dataframe[column].dtype:
                conversion_report.append(
                    f"Column '{column}' converted from {original_type} to numeric ({cleaned_dataframe[column].dtype})"
                )
        elif determined_type == 'string':
            # Convert the entire column to strings
            cleaned_dataframe[column] = cleaned_dataframe[column].astype(str)
            if original_type != cleaned_dataframe[column].dtype:
                conversion_report.append(
                    f"Column '{column}' converted from {original_type} to string ({cleaned_dataframe[column].dtype})"
                )
        # If the column is mixed or empty, we do not force a type here as per logic above.
    
    return cleaned_dataframe, conversion_report

def should_treat_as_categorical(column, cat_threshold=0.1):
    """
    Determine if a given column should be treated as categorical based on its unique value ratio.

    Parameters:
        column (pd.Series): The data column to examine.
        cat_threshold (float): The ratio below which a column should be considered categorical.
                               Default is 0.1 (10%).

    Returns:
        bool: True if the column should be treated as categorical, False otherwise.

    Example usage:
        if should_treat_as_categorical(df['col'], 0.1):
            df['col'] = pd.Categorical(df['col'])
    """
    unique_ratio = column.nunique() / len(column)
    # If the unique ratio is below the threshold, the column likely represents categorical data (e.g., categories)
    return unique_ratio < cat_threshold

def convert_to_categorical_types(dataframe, cat_threshold=0.1):
    """
    Convert eligible object-type columns to categorical type, based on a threshold for uniqueness.

    By reducing large text columns with few unique values to categories, memory usage and 
    some operations can be optimised.

    Parameters:
        dataframe (pd.DataFrame): The dataset to process.
        cat_threshold (float): The ratio threshold for determining categorical eligibility.
                               Default is 0.1 (10%).

    Returns:
        pd.DataFrame: A new DataFrame with converted categorical columns where appropriate.

    Example usage:
        df_categorical = convert_to_categorical_types(df, 0.1)
    """
    dataframe_converted = dataframe.copy()
    for col in dataframe_converted.select_dtypes(include=['object']).columns:
        if should_treat_as_categorical(dataframe_converted[col], cat_threshold):
            dataframe_converted[col] = pd.Categorical(dataframe_converted[col])
    return dataframe_converted

# File Upload Section
st.header("1. Upload Data")
uploaded_file = st.file_uploader("Choose a CSV file", type="csv")

if uploaded_file is not None:
    # Load the data if not already loaded
    if st.session_state.processed_df is None:
        st.session_state.processed_df = pd.read_csv(uploaded_file)
    df = st.session_state.processed_df
    st.success("File successfully uploaded!")
    
    # Data Overview Section
    st.header("2. Data Overview")
    st.write(f"Dataset Shape: {df.shape[0]} rows and {df.shape[1]} columns")
    
    # Show a preview of the data
    st.subheader("First Few Rows")
    st.dataframe(df.head())
    
    # Display Enhanced Information Table
    st.subheader("Enhanced Information Table")
    info_df, columns_with_missing = generate_enhanced_information_table(df)
    st.dataframe(info_df)
    
    # Missing Values Summary
    st.subheader("Missing Values Summary")
    total_missing = df.isnull().sum().sum() / (df.shape[0] * df.shape[1]) * 100
    st.write(f"Total Missing Value Proportion: {total_missing:.2f}%")
    
    if columns_with_missing:
        st.write("Columns with Missing Values and Their Data Types:")
        for col, dtype in columns_with_missing:
            st.write(f"- {col}: {dtype}")
    
    # Handle Mixed Data Types Section
    st.header("3. Handle Mixed Data Types")
    st.markdown("""
    Convert columns to their predominant data type if a certain percentage of values align with that type.
    High thresholds reduce misclassification, but you may lower it if you know the data has mixed types.
    """)
    
    # Slider for data type classification threshold
    st.markdown("**Data Type Threshold (Default 95%)**:")
    type_threshold = st.slider(
        "Data Type Threshold", 
        min_value=0.50, 
        max_value=1.00, 
        value=0.95,
        step=0.01,
        key="type_threshold"
    )
    
    if st.button("Convert Mixed Data Types"):
        cleaned_df, conversion_report = convert_mixed_data_types(df, type_threshold)
        st.session_state.processed_df = cleaned_df
        
        if conversion_report:
            st.success("Data types have been converted!")
            st.subheader("Data Type Conversions:")
            for change in conversion_report:
                st.write(change)
        else:
            st.write("No data type conversions were necessary.")
        
        st.subheader("Updated Data Preview:")
        st.dataframe(st.session_state.processed_df.head())
    
    # Handle Incorrect Entries Section
    st.header("4. Remove Incorrect Entries")
    st.markdown("""
    Identify and handle columns containing incorrect entries, such as numbers in predominantly string columns or
    strings in numeric columns. Choose how to clean these entries on a column-by-column basis.
    """)
    
    # Initialise session state for processed columns if not already set
    if 'processed_columns' not in st.session_state:
        st.session_state.processed_columns = set()
    
    if st.button("Analyse Incorrect Entries"):
        # Only analyse columns not yet processed
        remaining_df = df.drop(columns=list(st.session_state.processed_columns))
        analysis = identify_incorrect_entries(remaining_df)
        st.session_state.incorrect_entries_analysis = analysis
        
        if not analysis:
            if st.session_state.processed_columns:
                st.success("All mixed entry types have been handled!")
            else:
                st.write("No columns with mixed entry types were found.")
        else:
            st.success(f"Found {len(analysis)} columns with mixed entry types:")
            
            # For each column with issues, provide options to fix them
            for column, details in analysis.items():
                st.write(f"\n**Column: {column}**")
                st.write(f"- String entries: {details['string_count']} ({details['string_ratio']*100:.1f}%)")
                st.write(f"- Numeric entries: {details['numeric_count']} ({details['numeric_ratio']*100:.1f}%)")
                
                # Provide buttons for different cleaning actions
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    if st.button(f"Replace strings with NaN in {column}", key=f"str_nan_{column}"):
                        df.loc[details['string_indices'], column] = np.nan
                        st.session_state.processed_df = df
                        st.session_state.processed_columns.add(column)
                        st.success(f"✅ Replaced {details['string_count']} string entries with NaN in '{column}'")
                        st.experimental_rerun()
                
                with col2:
                    if st.button(f"Replace numbers with NaN in {column}", key=f"num_nan_{column}"):
                        df.loc[details['numeric_indices'], column] = np.nan
                        st.session_state.processed_df = df
                        st.session_state.processed_columns.add(column)
                        st.success(f"✅ Replaced {details['numeric_count']} numeric entries with NaN in '{column}'")
                        st.experimental_rerun()
                
                with col3:
                    if st.button(f"Skip {column}", key=f"skip_{column}"):
                        # Allow user to skip handling this column
                        st.session_state.processed_columns.add(column)
                        st.info(f"⏭️ Skipped handling incorrect entries in '{column}'")
                        st.experimental_rerun()
    
    # Display a processing history summary
    if st.session_state.processed_columns:
        st.markdown("---")
        st.markdown("**Processing History:**")
        processed_cols_list = list(st.session_state.processed_columns)
        if processed_cols_list:
            st.write(f"✓ Processed columns: {', '.join(processed_cols_list)}")
            
        # Provide an option to reset processing history
        if st.button("Reset Processing History"):
            st.session_state.processed_columns = set()
            st.success("Processing history has been reset.")
            st.experimental_rerun()

    # Categorical Data Handling Section
    st.header("5. Optimise Categorical Columns")
    st.markdown("""
    Identify and convert object-type columns with low unique value ratios into categorical columns.
    This can help reduce memory usage and potentially speed up certain operations.
    """)

    cat_threshold = st.slider(
        "Categorical Threshold (Default 10%)", 
        min_value=0.01, 
        max_value=0.50, 
        value=0.10,
        step=0.01,
        key="cat_threshold"
    )
    
    if st.button("Convert Object Columns to Categorical"):
        categorical_df = convert_to_categorical_types(df, cat_threshold)
        st.session_state.processed_df = categorical_df
        st.success("Appropriate columns have been converted to categorical type!")
        st.dataframe(st.session_state.processed_df.dtypes)
    
    # Download Processed Data Section
    st.header("6. Download Processed Data")
    if st.button("Show Final Data Preview"):
        st.subheader("Preview of Processed Data:")
        st.dataframe(st.session_state.processed_df.head())
        
        csv_export = st.session_state.processed_df.to_csv(index=False)
        st.download_button(
            label="Download CSV",
            data=csv_export,
            file_name="processed_data.csv",
            mime="text/csv"
        )
