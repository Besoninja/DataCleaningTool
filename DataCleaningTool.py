import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime

# Set page config
st.set_page_config(page_title="Data Cleaning Tool", layout="wide")

# Title and description
st.title("Data Cleaning Tool")
st.markdown("""
This data cleaning tool is built to clean and sort messy data. 
It is modular in design; you can fine-tune thresholds and handle data type conversions as you see fit.

**Recommendations & Notes:**
- Keep the thresholds at default 'best practice' values if unsure.
- Adjust thresholds carefully. For example:
  - Lowering the main data type threshold (default 95%) may classify mixed data columns as numeric or string more aggressively.
  - Adjusting the categorical threshold (default 10%) may change which columns are considered categorical.
- Consider carefully how you treat 'mixed' columns. Automated steps are helpful, but sometimes manual review is necessary.
- For missing value imputation, consider robust methods like KNN, mean/median, or regression-based approaches. Best practice suggests starting simple and iterating.
""")

# Sidebar for user configuration
st.sidebar.header("User Configurable Thresholds")
type_threshold = st.sidebar.slider(
    "Data Type Determination Threshold (Default: 95%)",
    min_value=0.5, max_value=1.0, value=0.95, step=0.01,
    help="If 95% or more of the values can be assigned to one type (numeric, string, datetime, boolean), that type is chosen."
)
cat_threshold = st.sidebar.slider(
    "Categorical Threshold (Default: 10%)",
    min_value=0.01, max_value=1.0, value=0.1, step=0.01,
    help="If the ratio of unique values in a column is less than this threshold, it will be considered categorical."
)

st.sidebar.markdown("""
**Recommendations:**
- **Data Type Threshold (95%)**:  
  A high threshold (like 95%) reduces the chance of misclassifying columns. Lower it only if your dataset is known to have slight variations.
  
- **Categorical Threshold (10%)**:  
  10% is a decent starting point for categorical detection. If too many columns are marked as categorical, try lowering this threshold.
""")

# File upload
st.header("1. Upload Data")
uploaded_file = st.file_uploader("Choose a CSV file", type="csv")

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

def can_be_datetime(value):
    """Check if a value can be parsed as a datetime."""
    if pd.isna(value):
        return True
    try:
        _ = pd.to_datetime(value, errors='raise')
        return True
    except:
        return False

def can_be_boolean(value):
    """Check if a value can be considered boolean (True/False)."""
    if pd.isna(value):
        return True
    # Consider 'True', 'False', 'true', 'false', 1, 0 as booleans
    if isinstance(value, bool):
        return True
    str_val = str(value).strip().lower()
    if str_val in ['true', 'false', '1', '0']:
        return True
    return False

def check_column_data_types(dataframe, threshold=0.95):
    """
    Analyze each column to determine its predominant data type based on content.
    Data types considered: numeric, string, datetime, boolean.
    A data type is assigned if more than `threshold` of values belong to that type.
    Otherwise, the column is considered mixed.
    """
    column_data_types = {}
    
    for column in dataframe.columns:
        string_count = 0
        numeric_count = 0
        datetime_count = 0
        boolean_count = 0
        
        col_data = dataframe[column].dropna()
        total_entries = len(col_data)
        
        if total_entries == 0:  # Handle completely empty columns
            column_data_types[column] = 'empty'
            continue
        
        for entry in col_data:
            # Check boolean first
            if can_be_boolean(entry):
                boolean_count += 1
            # Check numeric
            try:
                float(entry)
                numeric_test = True
            except:
                numeric_test = False
            
            # Check datetime
            datetime_test = can_be_datetime(entry)
            
            # Check string: If it's not numeric or datetime exclusively, it can always be string
            # but we only count as string if it's not counted as numeric, boolean, or datetime specifically
            # We'll refine counting logic below.
            
            # Counting logic:
            # Because a value might be convertible to multiple formats (e.g. "2020" could be numeric and datetime),
            # we prioritize types in order: boolean < numeric < datetime < string (lowest priority).
            # Actually, let's separately track counts and decide at the end.
            # For now, we have separate counters; a value may increment more than one counter if it qualifies.
            
            # However, to avoid double counting, let's set a priority:
            # If boolean_count increments, it means it's recognized as boolean. 
            # But a value like "1" could also be numeric. Let's increment all applicable and handle after.
            
            # After loop we'll see which category surpasses threshold.
        
        # We need to recalculate because we might have ambiguous overlaps.
        # Let's do a second pass with priority: boolean > datetime > numeric > string
        # Actually, it's simpler to assign categories by dominance after counting.

        # Re-check each entry in a more exclusive way:
        boolean_count = 0
        numeric_count = 0
        datetime_count = 0
        string_count = 0
        
        for entry in col_data:
            if can_be_boolean(entry):
                # If boolean is possible, check if numeric or datetime are also possible
                # Prioritization: boolean or numeric or datetime?
                # Let's just count separately and later pick the largest?
                pass
            # Try numeric
            is_numeric = False
            try:
                float(entry)
                is_numeric = True
            except:
                pass
            
            is_datetime = can_be_datetime(entry)
            is_boolean = can_be_boolean(entry)
            
            # Decide type increment:
            # Priority: If boolean, increment boolean. Else if datetime and not boolean, increment datetime.
            # Else if numeric and not boolean/datetime, increment numeric.
            # Else increment string.
            if is_boolean and not (is_datetime or is_numeric):
                boolean_count += 1
            elif is_datetime and not is_boolean:
                datetime_count += 1
            elif is_numeric and not (is_boolean or is_datetime):
                numeric_count += 1
            else:
                # If it falls through all above conditions, count as string
                # This includes cases where it can be numeric or datetime but also boolean?
                # Actually, let's refine logic:
                # If it's boolean, we took that path above. If it's datetime and boolean, we pick boolean first.
                # If numeric and boolean, pick boolean first.
                # If datetime and numeric, pick datetime first.
                
                # Adjust priority:
                # boolean > datetime > numeric > string
                # This means we should strictly check in order.
                
                # Let's do a final check in priority order:
                # Already done above, if it didn't match boolean/datetime/numeric exclusively, it's string.
                string_count += 1

        # Calculate the ratio
        numeric_ratio = numeric_count / total_entries
        string_ratio = string_count / total_entries
        datetime_ratio = datetime_count / total_entries
        boolean_ratio = boolean_count / total_entries
        
        # Determine type based on threshold
        if numeric_ratio >= threshold:
            column_data_types[column] = 'numeric'
        elif string_ratio >= threshold:
            column_data_types[column] = 'string'
        elif datetime_ratio >= threshold:
            column_data_types[column] = 'datetime'
        elif boolean_ratio >= threshold:
            column_data_types[column] = 'boolean'
        else:
            column_data_types[column] = 'mixed'
            
    return column_data_types

def clean_data(df, column_data_types):
    """
    Clean columns based on their predominant data type and remove incorrect entries.
    Returns a cleaned dataframe, a conversion report, and a dictionary of incorrect entries.
    Does not overwrite the original df.
    """
    df_cleaned = df.copy()
    conversion_report = []
    incorrect_entries = {}
    
    for column, dtype in column_data_types.items():
        original_type = df[column].dtype
        
        if dtype == 'numeric':
            # Convert column to numeric, setting invalid entries to NaN
            # Count how many entries become NaN in the process
            before_non_na = df_cleaned[column].notna().sum()
            df_cleaned[column] = pd.to_numeric(df_cleaned[column], errors='coerce')
            after_non_na = df_cleaned[column].notna().sum()
            
            if original_type != df_cleaned[column].dtype:
                conversion_report.append(
                    f"Column '{column}' converted from {original_type} to numeric"
                )
            
            lost_entries = before_non_na - after_non_na
            if lost_entries > 0:
                incorrect_entries[column] = lost_entries
                
        elif dtype == 'string':
            # Convert numeric-like entries to NaN
            # Identify numeric-like entries
            mask = df_cleaned[column].apply(lambda x: True if not pd.isna(x) else False)
            mask = mask & df_cleaned[column].apply(lambda x: isinstance(x, (int, float, np.number)))
            df_cleaned.loc[mask, column] = np.nan
            
            # Ensure column is string
            df_cleaned[column] = df_cleaned[column].astype(str)
            if original_type != df_cleaned[column].dtype:
                conversion_report.append(
                    f"Column '{column}' converted from {original_type} to string"
                )
            
            lost_entries = mask.sum()
            if lost_entries > 0:
                incorrect_entries[column] = lost_entries
        
        elif dtype == 'datetime':
            # Convert to datetime, non-datetime entries become NaN
            before_non_na = df_cleaned[column].notna().sum()
            df_cleaned[column] = pd.to_datetime(df_cleaned[column], errors='coerce')
            after_non_na = df_cleaned[column].notna().sum()
            
            if original_type != df_cleaned[column].dtype:
                conversion_report.append(
                    f"Column '{column}' converted from {original_type} to datetime"
                )
            
            lost_entries = before_non_na - after_non_na
            if lost_entries > 0:
                incorrect_entries[column] = lost_entries
        
        elif dtype == 'boolean':
            # Convert to boolean. Values that are not True/False/1/0 become NaN
            def to_bool(val):
                if pd.isna(val):
                    return np.nan
                str_val = str(val).strip().lower()
                if str_val in ['true', '1']:
                    return True
                elif str_val in ['false', '0']:
                    return False
                return np.nan
            
            before_non_na = df_cleaned[column].notna().sum()
            df_cleaned[column] = df_cleaned[column].apply(to_bool)
            after_non_na = df_cleaned[column].notna().sum()
            
            if original_type != df_cleaned[column].dtype:
                conversion_report.append(
                    f"Column '{column}' converted from {original_type} to boolean"
                )
            
            lost_entries = before_non_na - after_non_na
            if lost_entries > 0:
                incorrect_entries[column] = lost_entries
        
        elif dtype == 'mixed':
            # Mixed columns are tricky. For now, just notify the user.
            # Future steps could allow users to choose how to handle mixed columns.
            st.warning(f"Column '{column}' is mixed. Please review this column manually or adjust thresholds.")
            # Optionally, we could skip any forced conversion here.
    
    return df_cleaned, conversion_report, incorrect_entries

def is_categorical(column, threshold=0.1):
    """Determine if a column should be treated as categorical."""
    non_na = column.dropna()
    if len(non_na) == 0:
        return False
    unique_ratio = non_na.nunique() / len(non_na)
    return unique_ratio < threshold

def reassign_categorical_data_types(df, threshold=0.1):
    """Reassign columns to 'category' where applicable."""
    df_cat = df.copy()
    for col in df_cat.columns:
        if df_cat[col].dtype == object:
            if is_categorical(df_cat[col], threshold=threshold):
                df_cat[col] = pd.Categorical(df_cat[col])
    return df_cat

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
    
    # Mixed Data Type Handling
    st.header("3. Handle Data Types")
    st.markdown(f"""
    This section identifies columns with various data types (numeric, string, datetime, boolean) 
    based on the {type_threshold*100:.0f}% threshold you chose, and cleans them accordingly:
    - Non-numeric entries in 'numeric' columns become NaN.
    - Numeric entries in 'string' columns become NaN.
    - Non-datetime entries in 'datetime' columns become NaN.
    - Non-boolean entries in 'boolean' columns become NaN.
    - 'mixed' columns are flagged and require manual review or adjusted thresholds.
    """)
    
    if st.button("Classify and Clean Data Types"):
        column_data_types = check_column_data_types(df, threshold=type_threshold)
        df_cleaned, conversion_report, incorrect_entries = clean_data(df, column_data_types)
        
        st.success("Data types have been classified and cleaned where possible!")
        
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
            st.info("You may need to handle these NaNs in a future Missing Values Imputation module.")
        else:
            st.write("No incorrect entries were found in the dataset.")
        
        st.subheader("Updated Data Preview (Cleaned):")
        st.dataframe(df_cleaned.head())
        
        # Categorical Data Handling
        st.header("4. Optimize Categorical Columns")
        st.markdown(f"""
        This section identifies and converts appropriate columns to categorical data type. 
        A column is considered categorical if it has fewer unique values than {cat_threshold*100:.0f}% of its non-missing entries.
        """)
        
        if st.button("Convert Object Columns to Categorical"):
            df_cat = reassign_categorical_data_types(df_cleaned, threshold=cat_threshold)
            st.success("Appropriate columns have been converted to categorical type!")
            st.dataframe(df_cat.dtypes)
            
            # Download processed data
            st.header("5. Download Processed Data")
            st.markdown("You can download the cleaned and optimized dataset here.")
            csv = df_cat.to_csv(index=False)
            st.download_button(
                label="Download CSV",
                data=csv,
                file_name="processed_data.csv",
                mime="text/csv"
            )
        
        # Future Step: Missing Value Imputation
        st.header("Future Step: Missing Value Imputation")
        st.markdown("""
        In the future, you can handle missing values by:
        - Simple methods: mean, median, mode imputation.
        - Advanced methods: KNN imputation, MICE, or regression-based methods.
        
        **Best Practice Recommendations:**
        - Start with simple methods on numeric columns (mean/median) and mode for categorical columns.
        - Consider advanced methods (like KNN) if your data is complex or if simple methods bias your results.
        """)
