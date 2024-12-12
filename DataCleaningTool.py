import streamlit as st
import pandas as pd
import numpy as np

st.set_page_config(page_title="Data Cleaning Tool", layout="wide")

# Initialise session state
if 'processed_df' not in st.session_state:
    st.session_state.processed_df = None
if 'incorrect_entries_analysis' not in st.session_state:
    st.session_state.incorrect_entries_analysis = None
if 'processed_columns' not in st.session_state:
    st.session_state.processed_columns = set()

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
        
        if null_count > 0:
            columns_with_missing.append((column, dataframe[column].dtype))

    info_dataframe = pd.DataFrame(info_data)
    return info_dataframe, columns_with_missing

def determine_column_data_types(dataframe, type_threshold=0.95):
    """
    Determine the predominant data type of each column in a DataFrame.

    Parameters:
        dataframe (pd.DataFrame): The dataset to classify.
        type_threshold (float): The proportion threshold for classifying a column's type (default 0.95).

    Returns:
        dict: A dictionary mapping column names to determined data types ('numeric', 'string', 'mixed', 'empty').

    Example usage:
        column_types = determine_column_data_types(df, 0.95)
    """
    column_data_types = {}
    
    for column in dataframe.columns:
        string_count = 0
        numeric_count = 0
        
        # Assess each entry in the column
        for entry in dataframe[column]:
            if pd.isna(entry):
                continue
            if isinstance(entry, str):
                # Try parsing as numeric
                try:
                    float(entry)
                    numeric_count += 1
                except ValueError:
                    string_count += 1
            elif isinstance(entry, (int, float, np.number)):
                numeric_count += 1
            # Other types ignored in ratio calculation

        total_entries = len(dataframe[column].dropna())
        if total_entries == 0:
            column_data_types[column] = 'empty'
            continue
            
        numeric_ratio = numeric_count / total_entries if total_entries > 0 else 0
        string_ratio = string_count / total_entries if total_entries > 0 else 0
        
        if numeric_ratio > type_threshold:
            column_data_types[column] = 'numeric'
        elif string_ratio > type_threshold:
            column_data_types[column] = 'string'
        else:
            column_data_types[column] = 'mixed'
            
    return column_data_types

def identify_incorrect_entries(dataframe):
    """
    Identify columns that contain a mixture of numeric and string entries.

    Parameters:
        dataframe (pd.DataFrame): The dataset to analyse.

    Returns:
        dict: A dictionary where keys are column names and values are dictionaries with details about mixed entries.

    Example usage:
        analysis = identify_incorrect_entries(df)
    """
    analysis_results = {}
    
    for column in dataframe.columns:
        string_entries_indices = []
        numeric_entries_indices = []
        
        for idx, entry in enumerate(dataframe[column]):
            if pd.isna(entry):
                continue
            if isinstance(entry, str):
                # Check if string is numeric
                try:
                    float(entry)
                    numeric_entries_indices.append(idx)
                except ValueError:
                    string_entries_indices.append(idx)
            elif isinstance(entry, (int, float, np.number)):
                numeric_entries_indices.append(idx)
        
        total_valid = len(string_entries_indices) + len(numeric_entries_indices)
        if total_valid == 0:
            continue
            
        string_ratio = len(string_entries_indices) / total_valid if total_valid > 0 else 0
        numeric_ratio = len(numeric_entries_indices) / total_valid if total_valid > 0 else 0
        
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
    Convert columns in the DataFrame to their predominant data type.

    Parameters:
        dataframe (pd.DataFrame): The original dataset.
        type_threshold (float): The proportion threshold for classifying a column's type (default 0.95).

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
            cleaned_dataframe[column] = pd.to_numeric(cleaned_dataframe[column], errors='coerce')
            if original_type != cleaned_dataframe[column].dtype:
                conversion_report.append(
                    f"Column '{column}' converted from {original_type} to numeric ({cleaned_dataframe[column].dtype})"
                )
        elif determined_type == 'string':
            cleaned_dataframe[column] = cleaned_dataframe[column].astype(str)
            if original_type != cleaned_dataframe[column].dtype:
                conversion_report.append(
                    f"Column '{column}' converted from {original_type} to string ({cleaned_dataframe[column].dtype})"
                )
    
    return cleaned_dataframe, conversion_report

def should_treat_as_categorical(column, cat_threshold=0.1):
    """
    Determine if a given column should be treated as categorical.

    Parameters:
        column (pd.Series): The data column to examine.
        cat_threshold (float): The ratio below which a column should be considered categorical.

    Returns:
        bool: True if the column should be treated as categorical, otherwise False.

    Example usage:
        if should_treat_as_categorical(df['col'], 0.1):
            df['col'] = pd.Categorical(df['col'])
    """
    unique_ratio = column.nunique() / len(column)
    return unique_ratio < cat_threshold

def convert_to_categorical_types(dataframe, cat_threshold=0.1):
    """
    Convert eligible object-type columns to categorical type.

    Parameters:
        dataframe (pd.DataFrame): The dataset to process.
        cat_threshold (float): The uniqueness ratio threshold for determining categorical eligibility.

    Returns:
        pd.DataFrame: A new DataFrame with converted categorical columns.

    Example usage:
        df_categorical = convert_to_categorical_types(df, 0.1)
    """
    dataframe_converted = dataframe.copy()
    for col in dataframe_converted.select_dtypes(include=['object']).columns:
        if should_treat_as_categorical(dataframe_converted[col], cat_threshold):
            dataframe_converted[col] = pd.Categorical(dataframe_converted[col])
    return dataframe_converted

# Layout: 2 columns
# Left column: continuously updated Enhanced Information table
# Right column: main workflow (file upload, cleaning steps, sliders, etc.)
left_col, right_col = st.columns([1,3])

# Left column - Enhanced Info table (always visible and updated)
with left_col:
    st.subheader("Enhanced Information Table (Live Updates)")
    if st.session_state.processed_df is not None:
        info_df, columns_with_missing = generate_enhanced_information_table(st.session_state.processed_df)
        st.dataframe(info_df)
    else:
        st.write("Upload data to see enhanced information.")

# Right column - main actions
with right_col:
    st.title("Data Cleaning Tool")
    st.markdown("""
    This data cleaning tool is built to clean and sort messy data. 
    It is modular in design, meaning you can choose which parts of the tool to run on your data.
    """)

    # 1. Upload Data
    st.header("1. Upload Data")
    uploaded_file = st.file_uploader("Choose a CSV file", type="csv")

    if uploaded_file is not None and st.session_state.processed_df is None:
        st.session_state.processed_df = pd.read_csv(uploaded_file)
        st.success("File successfully uploaded!")

    if st.session_state.processed_df is not None:
        df = st.session_state.processed_df

        # 2. Data Overview
        st.header("2. Data Overview")
        st.write(f"Dataset Shape: {df.shape[0]} rows and {df.shape[1]} columns")

        st.subheader("First Few Rows of Current Data")
        st.dataframe(df.head())

        # 3. Handle Mixed Data Types
        st.header("3. Handle Mixed Data Types")
        st.markdown("""
        Convert columns to their predominant data type if a certain percentage of values align with that type.
        High thresholds reduce misclassification, but you may lower it if you know the data has mixed types.
        """)

        # Make the sliders narrower and left-aligned by placing them in narrower columns
        col_slider_1, _ = st.columns([1,3])
        with col_slider_1:
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

        # 4. Remove Incorrect Entries
        st.header("4. Remove Incorrect Entries")
        st.markdown("""
        Identify and handle columns containing incorrect entries, such as numbers in predominantly string columns or
        strings in numeric columns. Choose how to clean these entries on a column-by-column basis.
        """)

        if st.button("Analyse Incorrect Entries"):
            # Analyse only columns not yet processed
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

                for column, details in analysis.items():
                    st.write(f"\n**Column: {column}**")
                    st.write(f"- String entries: {details['string_count']} ({details['string_ratio']*100:.1f}%)")
                    st.write(f"- Numeric entries: {details['numeric_count']} ({details['numeric_ratio']*100:.1f}%)")

                    col_a, col_b, col_c = st.columns(3)
                    with col_a:
                        if st.button(f"Replace strings with NaN in {column}", key=f"str_nan_{column}"):
                            df.loc[details['string_indices'], column] = np.nan
                            st.session_state.processed_df = df
                            st.session_state.processed_columns.add(column)
                            st.success(f"✅ Replaced {details['string_count']} string entries with NaN in '{column}'")
                            st.experimental_rerun()

                    with col_b:
                        if st.button(f"Replace numbers with NaN in {column}", key=f"num_nan_{column}"):
                            df.loc[details['numeric_indices'], column] = np.nan
                            st.session_state.processed_df = df
                            st.session_state.processed_columns.add(column)
                            st.success(f"✅ Replaced {details['numeric_count']} numeric entries with NaN in '{column}'")
                            st.experimental_rerun()

                    with col_c:
                        if st.button(f"Skip {column}", key=f"skip_{column}"):
                            st.session_state.processed_columns.add(column)
                            st.info(f"⏭️ Skipped handling incorrect entries in '{column}'")
                            st.experimental_rerun()

        # Show processing history if available
        if st.session_state.processed_columns:
            st.markdown("---")
            st.markdown("**Processing History:**")
            processed_cols_list = list(st.session_state.processed_columns)
            if processed_cols_list:
                st.write(f"✓ Processed columns: {', '.join(processed_cols_list)}")

            # Reset processing history button
            if st.button("Reset Processing History"):
                st.session_state.processed_columns = set()
                st.success("Processing history has been reset.")
                st.experimental_rerun()

        # 5. Optimise Categorical Columns
        st.header("5. Optimise Categorical Columns")
        st.markdown("""
        Identify and convert object-type columns with low unique value ratios into categorical columns.
        This can reduce memory usage and optimise some operations.
        """)

        col_slider_2, _ = st.columns([1,3])
        with col_slider_2:
            cat_threshold = st.slider(
                "Categorical Threshold", 
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

        # 6. Download Processed Data
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
