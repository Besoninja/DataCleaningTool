import streamlit as st
import pandas as pd
import numpy as np

# Set page config
st.set_page_config(page_title="Data Cleaning Tool", layout="wide")

# Initialize session state for processed dataframe and analysis results
if 'processed_df' not in st.session_state:
    st.session_state.processed_df = None
if 'incorrect_entries_analysis' not in st.session_state:
    st.session_state.incorrect_entries_analysis = None

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

def analyze_incorrect_entries(df):
    """
    Analyze columns for incorrect entries (numerical in string columns and vice versa).
    Returns a dictionary with analysis for each affected column.
    """
    analysis = {}
    
    for column in df.columns:
        string_entries = []
        numeric_entries = []
        
        for idx, entry in enumerate(df[column]):
            if pd.isna(entry):
                continue
                
            if isinstance(entry, str):
                try:
                    float(entry)  # If this succeeds, it's a number in string form
                    numeric_entries.append(idx)
                except ValueError:
                    string_entries.append(idx)
            elif isinstance(entry, (int, float, np.number)):
                numeric_entries.append(idx)
        
        total_valid = len(string_entries) + len(numeric_entries)
        if total_valid == 0:
            continue
            
        str_ratio = len(string_entries) / total_valid if total_valid > 0 else 0
        num_ratio = len(numeric_entries) / total_valid if total_valid > 0 else 0
        
        # Only include columns with mixed types
        if 0 < str_ratio < 1 and 0 < num_ratio < 1:
            analysis[column] = {
                'string_count': len(string_entries),
                'numeric_count': len(numeric_entries),
                'string_indices': string_entries,
                'numeric_indices': numeric_entries,
                'string_ratio': str_ratio,
                'numeric_ratio': num_ratio
            }
    
    return analysis

def clean_mixed_data(df, type_threshold=0.95):
    """
    Convert columns to their predominant data type based on the threshold.
    """
    df_cleaned = df.copy()
    column_types = check_column_data_types(df, type_threshold)
    conversion_report = []
    
    for column, dtype in column_types.items():
        original_type = df[column].dtype
        if dtype == 'numeric':
            df_cleaned[column] = pd.to_numeric(df_cleaned[column], errors='coerce')
            if original_type != df_cleaned[column].dtype:
                conversion_report.append(
                    f"Column '{column}' converted from {original_type} to {df_cleaned[column].dtype}"
                )
        elif dtype == 'string':
            df_cleaned[column] = df_cleaned[column].astype(str)
            if original_type != df_cleaned[column].dtype:
                conversion_report.append(
                    f"Column '{column}' converted from {original_type} to {df_cleaned[column].dtype}"
                )
    
    return df_cleaned, conversion_report

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
    df = st.session_state.processed_df
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
    This section converts columns to their predominant data type based on the threshold.
    Columns will be converted to either numeric or string type if they meet the threshold criteria.
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
    
    if st.button("Convert Mixed Data Types"):
        cleaned_df, conversion_report = clean_mixed_data(df, type_threshold)
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
    
    # Handle Incorrect Entries
    st.header("4. Remove Incorrect Entries")
    st.markdown("""
    This section identifies columns containing incorrect entries (e.g., numerical values in string columns or string values in numerical columns).
    You can choose how to handle these entries on a column-by-column basis.
    """)
    
    if st.button("Analyze Incorrect Entries"):
        analysis = analyze_incorrect_entries(df)
        st.session_state.incorrect_entries_analysis = analysis
        
        if not analysis:
            st.write("No columns with mixed entry types were found.")
        else:
            st.success(f"Found {len(analysis)} columns with mixed entry types:")
            
            for column, details in analysis.items():
                st.write(f"\n**Column: {column}**")
                st.write(f"- String entries: {details['string_count']} ({details['string_ratio']*100:.1f}%)")
                st.write(f"- Numeric entries: {details['numeric_count']} ({details['numeric_ratio']*100:.1f}%)")
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    if st.button(f"Replace strings with NaN in {column}"):
                        df.loc[details['string_indices'], column] = np.nan
                        st.session_state.processed_df = df
                        st.success(f"Replaced {details['string_count']} string entries with NaN in {column}")
                
                with col2:
                    if st.button(f"Replace numbers with NaN in {column}"):
                        df.loc[details['numeric_indices'], column] = np.nan
                        st.session_state.processed_df = df
                        st.success(f"Replaced {details['numeric_count']} numeric entries with NaN in {column}")
                
                with col3:
                    if st.button(f"Skip {column}"):
                        st.info(f"No changes made to column {column}")
    
    # Categorical Data Handling
    st.header("5. Optimise Categorical Columns")
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
        st.session_state.processed_df = categorical_df
        st.success("Appropriate columns have been converted to categorical type!")
        st.dataframe(st.session_state.processed_df.dtypes)
    
    # Download processed data
    st.header("6. Download Processed Data")
    if st.button("Show Final Data Preview"):
        st.subheader("Preview of Processed Data:")
        st.dataframe(st.session_state.processed_df.head())
        
        csv = st.session_state.processed_df.to_csv(index=False)
        st.download_button(
            label="Download CSV",
            data=csv,
            file_name="processed_data.csv",
            mime="text/csv"
        )
