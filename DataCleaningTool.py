import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import io

# Set page config
st.set_page_config(page_title="Data Cleaning Tool", layout="wide")

# Title and description
st.title("Data Cleaning Tool")
st.markdown("""
This data cleaning tool is built to clean and sort messy data. 
The tool is modular in design, i.e. any or all parts of the tool can be run on your data.
""")

# Function to create enhanced info table
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

def is_categorical(column, threshold=0.1):
    """Determine if a column should be treated as categorical."""
    unique_ratio = column.nunique() / len(column)
    return unique_ratio < threshold

def reassign_categorical_data_types(df):
    """Reassign columns to 'category' where applicable."""
    for col in df.select_dtypes(include=['object']).columns:
        if is_categorical(df[col]):
            df[col] = pd.Categorical(df[col])
    return df

# File upload
st.header("1. Upload Data")
uploaded_file = st.file_uploader("Choose a CSV file", type="csv")

if uploaded_file is not None:
    # Load the data
    df = pd.read_csv(uploaded_file)
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
    
    # Data Type Handling
    st.header("3. Data Type Handling")
    if st.button("Convert Categorical Columns"):
        df = reassign_categorical_data_types(df)
        st.success("Categorical columns have been converted!")
        st.dataframe(df.dtypes)
    
    # Correlation Analysis
    st.header("4. Correlation Analysis")
    num_cols = df.select_dtypes(include=['int64', 'float64'])
    if not num_cols.empty:
        corr_matrix = num_cols.corr()
        
        # Create correlation heatmap
        fig, ax = plt.subplots(figsize=(10, 8))
        sns.heatmap(corr_matrix, annot=True, fmt=".2f", ax=ax)
        st.pyplot(fig)
        
        # Show strong correlations
        st.subheader("Strong Correlations (>0.5)")
        strong_corr = corr_matrix[(abs(corr_matrix) > 0.5) & (abs(corr_matrix) < 1.0)]
        for row in strong_corr.index:
            for col in strong_corr.columns:
                if not pd.isna(strong_corr.loc[row, col]):
                    st.write(f"{row} has a correlation of {strong_corr.loc[row, col]:.2f} with {col}")
    
    # Download processed data
    st.header("5. Download Processed Data")
    if st.button("Download Processed Data"):
        csv = df.to_csv(index=False)
        st.download_button(
            label="Download CSV",
            data=csv,
            file_name="processed_data.csv",
            mime="text/csv"
        )
