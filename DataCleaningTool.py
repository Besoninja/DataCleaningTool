import streamlit as st
import pandas as pd
import numpy as np

# Set the page configuration FIRST (must be the first Streamlit command)
st.set_page_config(page_title="Data Cleaning Tool", layout="wide")

# Initialise session state variables
if 'processed_df' not in st.session_state:
    st.session_state.processed_df = None
if 'incorrect_entries_analysis' not in st.session_state:
    st.session_state.incorrect_entries_analysis = None
if 'processed_columns' not in st.session_state:
    st.session_state.processed_columns = set()

def generate_enhanced_information_table(dataframe):
    """
    Generate an enhanced overview table for a given DataFrame:
    - ColumnName, DataType, UniqueValues, NullCount, % EmptyCells

    Parameters:
        dataframe (pd.DataFrame): The dataset for which to generate the information table.

    Returns:
        tuple: (info_dataframe, columns_with_missing)
               info_dataframe (pd.DataFrame): Summary table with column details.
               columns_with_missing (list): List of (column_name, dtype) for columns with missing values.
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
    Determine the predominant data type of each column based on a threshold.

    Parameters:
        dataframe (pd.DataFrame)
        type_threshold (float): Ratio needed to classify a column as numeric or string.

    Returns:
        dict: {column_name: 'numeric'|'string'|'mixed'|'empty'}
    """
    column_data_types = {}
    
    for column in dataframe.columns:
        string_count = 0
        numeric_count = 0
        
        for entry in dataframe[column]:
            if pd.isna(entry):
                continue
            if isinstance(entry, str):
                # Check if we can parse as numeric
                try:
                    float(entry)
                    numeric_count += 1
                except ValueError:
                    string_count += 1
            elif isinstance(entry, (int, float, np.number)):
                numeric_count += 1
        
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
        dict: {column_name: analysis_details}
    """
    analysis_results = {}
    
    for column in dataframe.columns:
        string_indices = []
        numeric_indices = []
        
        for idx, entry in enumerate(dataframe[column]):
            if pd.isna(entry):
                continue
            if isinstance(entry, str):
                try:
                    float(entry)
                    numeric_indices.append(idx)
                except ValueError:
                    string_indices.append(idx)
            elif isinstance(entry, (int, float, np.number)):
                numeric_indices.append(idx)
        
        total_valid = len(string_indices) + len(numeric_indices)
        if total_valid == 0:
            # No entries to classify
            continue
            
        string_ratio = len(string_indices) / total_valid if total_valid > 0 else 0
        numeric_ratio = len(numeric_indices) / total_valid if total_valid > 0 else 0
        
        # Only consider if there's a meaningful mixture
        if 0 < string_ratio < 1 and 0 < numeric_ratio < 1:
            analysis_results[column] = {
                'string_count': len(string_indices),
                'numeric_count': len(numeric_indices),
                'string_indices': string_indices,
                'numeric_indices': numeric_indices,
                'string_ratio': string_ratio,
                'numeric_ratio': numeric_ratio
            }
    
    return analysis_results

def convert_mixed_data_types(dataframe, type_threshold=0.95):
    """
    Convert columns to numeric or string based on a threshold.

    Parameters:
        dataframe (pd.DataFrame)
        type_threshold (float)

    Returns:
        (pd.DataFrame, list): (cleaned_df, conversion_report)
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
    Determine if a column should be categorical based on unique value ratio.
    """
    unique_ratio = column.nunique() / len(column)
    return unique_ratio < cat_threshold

def convert_to_categorical_types(dataframe, cat_threshold=0.1):
    """
    Convert suitable object-type columns to categorical.
    """
    dataframe_converted = dataframe.copy()
    for col in dataframe_converted.select_dtypes(include=['object']).columns:
        if should_treat_as_categorical(dataframe_converted[col], cat_threshold):
            dataframe_converted[col] = pd.Categorical(dataframe_converted[col])
    return dataframe_converted

# Layout: two columns
left_col, right_col = st.columns([1, 4], gap="medium")

with left_col:
    st.subheader("Enhanced Information Table (Live Updates)")
    # Display the Enhanced Information Table if data is loaded
    if st.session_state.processed_df is not None:
        info_df, columns_with_missing = generate_enhanced_information_table(st.session_state.processed_df)
        st.dataframe(info_df, height=600)
    else:
        st.write("No data loaded yet.")

with right_col:
    # File Upload Section
    st.header("1. Upload Data")
    uploaded_file = st.file_uploader("Choose a CSV file", type="csv")

    if uploaded_file is not None:
        # Load data if not already loaded
        if st.session_state.processed_df is None:
            st.session_state.processed_df = pd.read_csv(uploaded_file)
        df = st.session_state.processed_df
        st.success("File successfully uploaded!")

        # Data Overview
        st.header("2. Data Overview")
        st.write(f"Dataset Shape: {df.shape[0]} rows and {df.shape[1]} columns")
        
        st.subheader("First Few Rows")
        st.dataframe(df.head())

        # Missing Values Summary
        st.subheader("Missing Values Summary")
        total_missing = df.isnull().sum().sum() / (df.shape[0] * df.shape[1]) * 100
        st.write(f"Total Missing Value Proportion: {total_missing:.2f}%")
        
        if 'columns_with_missing' in locals() and columns_with_missing:
            st.write("Columns with Missing Values and Their Data Types:")
            for col, dtype in columns_with_missing:
                st.write(f"- {col}: {dtype}")
        
        # Handle Mixed Data Types
        st.header("3. Handle Mixed Data Types")
        st.markdown("""Convert columns to predominant data type.""")
        slider_col, _ = st.columns([0.3, 0.7])
        with slider_col:
            type_threshold = st.slider(
                "Data Type Threshold (Default 95%)",
                min_value=0.50, 
                max_value=1.00, 
                value=0.95,
                step=0.01,
                key="type_threshold"
            )

        if st.button("Convert Mixed Data Types"):
            cleaned_df, conversion_report = convert_mixed_data_types(df, type_threshold)
            st.session_state.processed_df = cleaned_df
            df = st.session_state.processed_df  # Update reference
            
            if conversion_report:
                st.success("Data types have been converted!")
                st.subheader("Data Type Conversions:")
                for change in conversion_report:
                    st.write(change)
            else:
                st.write("No data type conversions were necessary.")
            
            st.subheader("Updated Data Preview:")
            st.dataframe(df.head())

        # Handle Incorrect Entries
        st.header("4. Remove Incorrect Entries")
        st.markdown("Identify and fix columns with incorrect entries.")
        
        # Analyse Incorrect Entries
        if st.button("Analyse Incorrect Entries"):
            # Filter out columns already processed
            remaining_df = df.drop(columns=list(st.session_state.processed_columns), errors='ignore')
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
                    
                    c1, c2, c3 = st.columns(3)
                    
                    with c1:
                        if st.button(f"Replace strings with NaN in {column}", key=f"str_nan_{column}"):
                            df.loc[details['string_indices'], column] = np.nan
                            st.session_state.processed_df = df
                            st.session_state.processed_columns.add(column)
                            st.success(f"✅ Replaced string entries with NaN in '{column}'")
                            st.experimental_rerun()
                    
                    with c2:
                        if st.button(f"Replace numbers with NaN in {column}", key=f"num_nan_{column}"):
                            df.loc[details['numeric_indices'], column] = np.nan
                            st.session_state.processed_df = df
                            st.session_state.processed_columns.add(column)
                            st.success(f"✅ Replaced numeric entries with NaN in '{column}'")
                            st.experimental_rerun()
                    
                    with c3:
                        if st.button(f"Skip {column}", key=f"skip_{column}"):
                            st.session_state.processed_columns.add(column)
                            st.info(f"⏭️ Skipped handling incorrect entries in '{column}'")
                            st.experimental_rerun()

        # Processing History
        if st.session_state.processed_columns:
            st.markdown("---")
            st.markdown("**Processing History:**")
            processed_cols_list = list(st.session_state.processed_columns)
            if processed_cols_list:
                st.write(f"✓ Processed columns: {', '.join(processed_cols_list)}")
            
            # Reset processing history
            if st.button("Reset Processing History"):
                st.session_state.processed_columns = set()
                st.success("Processing history has been reset.")
                st.experimental_rerun()

        # Optimise Categorical Columns
        st.header("5. Optimise Categorical Columns")
        st.markdown("Convert object-type columns to categorical if they have a low unique value ratio.")
        cat_slider_col, _ = st.columns([0.3, 0.7])
        with cat_slider_col:
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
            df = st.session_state.processed_df
            st.success("Appropriate columns have been converted to categorical type!")
            st.dataframe(df.dtypes)

        # Download Processed Data
        st.header("6. Download Processed Data")
        if st.button("Show Final Data Preview"):
            st.subheader("Preview of Processed Data:")
            st.dataframe(df.head())
            
            csv_export = df.to_csv(index=False)
            st.download_button(
                label="Download CSV",
                data=csv_export,
                file_name="processed_data.csv",
                mime="text/csv"
            )
    else:
        st.info("Please upload a CSV file to get started.")
