import streamlit as st
import pandas as pd
import numpy as np

# Set page config
st.set_page_config(page_title="Data Cleaning Tool", layout="wide")

# Initialize session state for processed dataframe
if 'processed_df' not in st.session_state:
    st.session_state.processed_df = None

# Title and description
st.title("Data Cleaning Tool")
st.markdown("""
This data cleaning tool is built to clean and sort messy data. 
The tool is modular in design, i.e. any or all parts of the tool can be run on your data.
""")

def enhanced_info(df):
    """Generate and print a summary table that provides an enhanced view of the DataFrame."""
    info_data = {
        'ColumnName': [],
        'DataType': [], 
        'UniqueValues': [],
        'NullCount': [],
        '% EmptyCells': [],
    }
    
    columns_with_missing = []
    
    for col in df.columns:
        null_count = df[col].isnull().sum()
        info_data['ColumnName'].append(col)
        info_data['DataType'].append(df[col].dtype.name)
        info_data['UniqueValues'].append(df[col].nunique())
        info_data['NullCount'].append(null_count)
        empty_cells_prop = (null_count / df.shape[0]) * 100
        info_data['% EmptyCells'].append(f'{empty_cells_prop:.2f}%')
        
        if null_count > 0:
            columns_with_missing.append((col, df[col].dtype))

    info_df = pd.DataFrame(info_data)
    return info_df, columns_with_missing

def check_column_data_types(dataframe, type_threshold=0.95):
    """
    Analyse each column to determine its predominant data type based on content.
    A data type is assigned if more than type_threshold of values belong to that type.
    """
    column_data_types = {}
    
    for column in dataframe.columns:
        string_count = 0
        numeric_count = 0
        other_count = 0
        
        # Check each entry in the column
        for entry in dataframe[column]:
            if pd.isna(entry):  # Skip NaN values
                continue
            if isinstance(entry, str):
                # Try to convert string to numeric
                try:
                    float(entry)
                    numeric_count += 1
                except ValueError:
                    string_count += 1
            elif isinstance(entry, (int, float, np.number)):
                numeric_count += 1
            else:
                other_count += 1
        
        # Calculate the ratio of each data type
        total_entries = len(dataframe[column].dropna())
        if total_entries == 0:  # Handle completely empty columns
            column_data_types[column] = 'empty'
            continue
            
        numeric_ratio = numeric_count / total_entries
        string_ratio = string_count / total_entries
        
        if numeric_ratio > type_threshold:
            column_data_types[column] = 'numeric'
        elif string_ratio > type_threshold:
            column_data_types[column] = 'string'
        else:
            column_data_types[column] = 'mixed'
            
    return column_data_types

def clean_mixed_data(df, type_threshold=0.95):
    """
    Clean columns based on their predominant data type and remove incorrect entries.
    Returns cleaned dataframe and report of changes made.
    """
    df_cleaned = df.copy()
    column_types = check_column_data_types(df, type_threshold)
    conversion_report = []
    incorrect_entries = {}
    
    for column, dtype in column_types.items():
        original_type = df[column].dtype
        if dtype == 'numeric':
            non_numeric_count = df_cleaned[column].apply(
                lambda x: not pd.isna(x) and not isinstance(x, (int, float, np.number))
            ).sum()
            
            df_cleaned[column] = pd.to_numeric(df_cleaned[column], errors='coerce')
            
            if original_type != df_cleaned[column].dtype:
                conversion_report.append(
                    f"Column '{column}' converted from {original_type} to {df_cleaned[column].dtype}"
                )
            
            if non_numeric_count > 0:
                incorrect_entries[column] = non_numeric_count
                
        elif dtype == 'string':
            numeric_count = df_cleaned[column].apply(
                lambda x: isinstance(x, (int, float, np.number))
            ).sum()
            
            mask = df_cleaned[column].apply(lambda x: isinstance(x, (int, float, np.number)))
            df_cleaned.loc[mask, column] = np.nan
            df_cleaned[column] = df_cleaned[column].astype(str)
            
            if original_type != df_cleaned[column].dtype:
                conversion_report.append(
                    f"Column '{column}' converted from {original_type} to {df_cleaned[column].dtype}"
                )
            
            if numeric_count > 0:
                incorrect_entries[column] = numeric_count
    
    return df_cleaned, conversion_report, incorrect_entries

def is_categorical(column, cat_threshold=0.1):
    """Determine if a column should be treated as categorical."""
    unique_ratio = column.nunique() / len(column)
    return unique_ratio < cat_threshold

def reassign_categorical_data_types(df, cat_threshold=0.1):
    """Reassign columns to 'category' where applicable."""
    df_cat = df.copy()
    for col in df_cat.select_dtypes(include=['object']).columns:
        if is_categorical(df_cat[col], cat_threshold):
            df_cat[col] = pd.Categorical(df_cat[col])
    return df_cat

# File upload
st.header("1. Upload Data")
uploaded_file = st.file_uploader("Choose a CSV file", type="csv")

if uploaded_file is not None:
    # Load the data
    if st.session_state.processed_df is None:
        st.session_state.processed_df = pd.read_csv(uploaded_file)
    
    df = st.session_state.processed_df  # Use the processed df for display
    st.success("File successfully uploaded!")
    
    # Data Overview Section
    st.header("2. Data Overview")
    st.write(f"Dataset Shape: {df.shape[0]} rows and {df.shape[1]} columns")
    
    # Show first few rows
    st.subheader("First Few Rows")
    st.dataframe(df.head())
    
    # Enhanced Info Table
    st.subheader("Enhanced Information Table")
    info_df, columns_with_missing = enhanced_info(df)
    st.dataframe(info_df)
    
    # Missing Values Summary
    st.subheader("Missing Values Summary")
    total_missing = df.isnull().sum().sum() / (df.shape[0] * df.shape[1]) * 100
    st.write(f"Total Missing Value Proportion: {total_missing:.2f}%")
    
    if columns_with_missing:
        st.write("Columns with Missing Values and Their Data Types:")
        for col, dtype in columns_with_missing:
            st.write(f"- {col}: {dtype}")
    
    # Mixed Data Type Handling
    st.header("3. Handle Mixed Data Types")
    st.markdown("""
    This section identifies columns with mixed data types and cleans them based on the predominant type in each column.
    """)
    
    # Add slider for data type threshold
    st.markdown("**Data Type Threshold (95%)**:")
    st.markdown("A high threshold (above 95%) reduces the chance of misclassifying columns. Lower it only if your dataset is known to have slight variations.")
    type_threshold = st.slider("Data Type Threshold", 
                             min_value=0.50, 
                             max_value=1.00, 
                             value=0.95,
                             step=0.01,
                             key="type_threshold")
    
    if st.button("Clean Mixed Data Types"):
        cleaned_df, conversion_report, incorrect_entries = clean_mixed_data(df, type_threshold)
        st.session_state.processed_df = cleaned_df  # Update the processed dataframe
        st.success("Mixed data types have been cleaned!")
        
        # Display conversion report
        if conversion_report:
            st.subheader("Data Type Conversions:")
            for change in conversion_report:
                st.write(change)
        else:
            st.write("No data type conversions were necessary.")
        
        # Display incorrect entries report
        if incorrect_entries:
            st.subheader("Incorrect Entries Converted to NaN:")
            for column, count in incorrect_entries.items():
                st.write(f"- {column}: {count} incorrect entries identified and converted to NaN")
            
            st.warning("⚠️ Incorrect entries have been converted to NaN. You will need to run the 'Handle Missing Values' module to address these missing values.")
        else:
            st.write("No incorrect entries were found in the dataset.")
        
        st.subheader("Updated Data Preview:")
        st.dataframe(st.session_state.processed_df.head())
    
    # Categorical Data Handling
    st.header("4. Optimise Categorical Columns")
    st.markdown("""
    This section identifies and converts appropriate columns to categorical data type.
    """)
    
    # Add slider for categorical threshold
    st.markdown("**Categorical Threshold (10%)**:")
    st.markdown("10% is a decent starting point for categorical detection. If too many columns are marked as categorical, try lowering this threshold.")
    cat_threshold = st.slider("Categorical Threshold", 
                            min_value=0.01, 
                            max_value=0.50, 
                            value=0.10,
                            step=0.01,
                            key="cat_threshold")
    
    if st.button("Convert Object Columns to Categorical"):
        categorical_df = reassign_categorical_data_types(df, cat_threshold)
        st.session_state.processed_df = categorical_df  # Update the processed dataframe
        st.success("Appropriate columns have been converted to categorical type!")
        st.dataframe(st.session_state.processed_df.dtypes)
    
    # Download processed data
    st.header("5. Download Processed Data")
    if st.button("Download Processed Data"):
        # Add a preview of the processed data
        st.subheader("Preview of Processed Data:")
        st.dataframe(st.session_state.processed_df.head())
        
        csv = st.session_state.processed_df.to_csv(index=False)
        st.download_button(
            label="Download CSV",
            data=csv,
            file_name="processed_data.csv",
            mime="text/csv"
        )
