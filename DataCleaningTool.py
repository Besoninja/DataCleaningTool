import streamlit as st
import pandas as pd
import numpy as np

st.set_page_config(page_title="Data Cleaning Tool", layout="wide")

# Initialise session state for the processed dataframe and analysis results
if 'processed_df' not in st.session_state:
    st.session_state.processed_df = None
if 'incorrect_entries_analysis' not in st.session_state:
    st.session_state.incorrect_entries_analysis = None

def generate_enhanced_information_table(dataframe):
    """
    Purpose:
        Generate a table with enhanced information for each column in the dataframe.

    Parameters:
        dataframe (pd.DataFrame): The dataset to analyse.

    Returns:
        tuple: (info_dataframe, columns_with_missing)
               info_dataframe: A pd.DataFrame with column-level details such as:
                               ColumnName, DataType, UniqueValues, NullCount, % EmptyCells
               columns_with_missing: A list of tuples (column_name, dtype) for columns with missing values.

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
    Purpose:
        Determine predominant data types of columns by analysing their values.

    Parameters:
        dataframe (pd.DataFrame): The dataset to analyse.
        type_threshold (float): Proportion threshold to classify a column's type (default 0.95).

    Returns:
        dict: Mapping of column names to their determined data type ('numeric', 'string', 'mixed', or 'empty').

    Example usage:
        column_types = determine_column_data_types(df, 0.95)
    """
    column_data_types = {}

    for column in dataframe.columns:
        string_count = 0
        numeric_count = 0

        for entry in dataframe[column]:
            if pd.isna(entry):
                # Ignore missing values as they don't define a type
                continue
            if isinstance(entry, str):
                # Try parsing the string as a number
                try:
                    float(entry)
                    numeric_count += 1
                except ValueError:
                    string_count += 1
            elif isinstance(entry, (int, float, np.number)):
                numeric_count += 1

        total_entries = len(dataframe[column].dropna())
        if total_entries == 0:
            # No valid entries, treat as 'empty'
            column_data_types[column] = 'empty'
            continue

        numeric_ratio = numeric_count / total_entries if total_entries > 0 else 0
        string_ratio = string_count / total_entries if total_entries > 0 else 0

        # Assign the type based on which ratio surpasses the threshold
        if numeric_ratio > type_threshold:
            column_data_types[column] = 'numeric'
        elif string_ratio > type_threshold:
            column_data_types[column] = 'string'
        else:
            column_data_types[column] = 'mixed'

    return column_data_types

def identify_incorrect_entries(dataframe):
    """
    Purpose:
        Identify columns containing mixed data types (string and numeric values intermingled).
        Returns a dictionary with details on which entries are incorrect.

    Parameters:
        dataframe (pd.DataFrame): The dataset to analyse.

    Returns:
        dict: Keys are column names with mixed entry types. Values contain counts and indices of incorrect entries.

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
    Purpose:
        Convert columns to their predominant data type based on the specified threshold.

    Parameters:
        dataframe (pd.DataFrame): Original dataset.
        type_threshold (float): Threshold for deciding predominant type (default 0.95).

    Returns:
        tuple: (cleaned_dataframe, conversion_report)
               cleaned_dataframe: Updated dataframe.
               conversion_report: A list of changes made to column data types.

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
    Purpose:
        Determine if a column should be treated as categorical based on its unique value ratio.

    Parameters:
        column (pd.Series): The column to examine.
        cat_threshold (float): Ratio threshold for deciding categoricity (default 0.1).

    Returns:
        bool: True if the column should be categorical, False otherwise.

    Example usage:
        if should_treat_as_categorical(df['col'], 0.1):
            df['col'] = pd.Categorical(df['col'])
    """
    unique_ratio = column.nunique() / len(column)
    return unique_ratio < cat_threshold

def convert_to_categorical_types(dataframe, cat_threshold=0.1):
    """
    Purpose:
        Convert object-type columns with low unique value ratios into categorical columns.

    Parameters:
        dataframe (pd.DataFrame): The dataset to process.
        cat_threshold (float): Threshold for categoricity (default 0.1).

    Returns:
        pd.DataFrame: Updated dataframe with categorical conversions applied.

    Example usage:
        df_categorical = convert_to_categorical_types(df, 0.1)
    """
    dataframe_converted = dataframe.copy()
    for col in dataframe_converted.select_dtypes(include=['object']).columns:
        if should_treat_as_categorical(dataframe_converted[col], cat_threshold):
            dataframe_converted[col] = pd.Categorical(dataframe_converted[col])
    return dataframe_converted

# Sidebar: Enhanced Information Table (Always Visible and Updated)
st.sidebar.subheader("Enhanced Information Table")
if st.session_state.processed_df is not None:
    info_df, columns_with_missing = generate_enhanced_information_table(st.session_state.processed_df)
    st.sidebar.dataframe(info_df, use_container_width=True)
else:
    st.sidebar.write("No data loaded yet.")

# Main Section
st.title("Data Cleaning Tool")
st.markdown("""
This tool allows you to clean and refine your dataset step-by-step. 
All changes made are reflected below in the Enhanced Information Table in the sidebar.
""")

# File Upload Section
st.header("1. Upload Data")
uploaded_file = st.file_uploader("Choose a CSV file", type="csv")

if uploaded_file is not None:
    if st.session_state.processed_df is None:
        st.session_state.processed_df = pd.read_csv(uploaded_file)
    df = st.session_state.processed_df
    st.success("File successfully uploaded!")

    # Data Overview Section
    st.header("2. Data Overview")
    st.write(f"Dataset Shape: {df.shape[0]} rows and {df.shape[1]} columns")

    st.subheader("First Few Rows")
    st.dataframe(df.head(), use_container_width=True)

    # Handle Mixed Data Types Section
    st.header("3. Handle Mixed Data Types")
    st.markdown("""
    Convert columns to a uniform data type if a certain proportion of their entries match that type.
    """)

    # Make sliders smaller and left-aligned by placing them in a narrow column
    col_sliders, _ = st.columns([1, 4])
    with col_sliders:
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

        # Show updated preview
        st.subheader("Updated Data Preview:")
        st.dataframe(st.session_state.processed_df.head(), use_container_width=True)

    # Remove Incorrect Entries Section
    st.header("4. Remove Incorrect Entries")
    st.markdown("""
    Identify and resolve columns containing incorrect entries (e.g., strings in numeric columns).
    """)

    if 'processed_columns' not in st.session_state:
        st.session_state.processed_columns = set()

    if st.button("Analyse Incorrect Entries"):
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
                        st.session_state.processed_columns.add(column)
                        st.info(f"⏭️ Skipped handling incorrect entries in '{column}'")
                        st.experimental_rerun()

    # Processing history
    if st.session_state.processed_columns:
        st.markdown("---")
        st.markdown("**Processing History:**")
        processed_cols_list = list(st.session_state.processed_columns)
        if processed_cols_list:
            st.write(f"✓ Processed columns: {', '.join(processed_cols_list)}")

        if st.button("Reset Processing History"):
            st.session_state.processed_columns = set()
            st.success("Processing history has been reset.")
            st.experimental_rerun()

    # Optimise Categorical Columns Section
    st.header("5. Optimise Categorical Columns")
    st.markdown("""
    Convert suitable object-type columns to categorical, reducing memory usage and improving efficiency.
    """)

    col_sliders_cat, _ = st.columns([1,4])
    with col_sliders_cat:
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

    # Download Processed Data Section
    st.header("6. Download Processed Data")
    if st.button("Show Final Data Preview"):
        st.subheader("Preview of Processed Data:")
        st.dataframe(st.session_state.processed_df.head(), use_container_width=True)

        csv_export = st.session_state.processed_df.to_csv(index=False)
        st.download_button(
            label="Download CSV",
            data=csv_export,
            file_name="processed_data.csv",
            mime="text/csv"
        )
