import streamlit as st
import pandas as pd
import numpy as np
import openpyxl
from sklearn.impute import KNNImputer
from sklearn.linear_model import LinearRegression

# Handle IterativeImputer separately since it's experimental
try:
    from sklearn.experimental import enable_iterative_imputer
    from sklearn.impute import IterativeImputer
except ImportError:
    IterativeImputer = None

# For MissForest
try:
    from missingpy import MissForest
except ImportError:
    MissForest = None

# Set the page configuration at the very start
st.set_page_config(page_title="Data Cleaning Tool", layout="wide")

# Initialise session state variables
if 'processed_df' not in st.session_state:
    st.session_state.processed_df = None
if 'incorrect_entries_analysis' not in st.session_state:
    st.session_state.incorrect_entries_analysis = None
if 'processed_columns' not in st.session_state:
    st.session_state.processed_columns = set()
if 'selected_section' not in st.session_state:
    st.session_state.selected_section = 'File Upload'

#####################################################################################################################################
#####################################################################################################################################
#####################################################################################################################################
### Functions ###
#####################################################################################################################################
#####################################################################################################################################
#####################################################################################################################################

def generate_enhanced_information_table(dataframe):
    """
    Generate an enhanced overview table for a given DataFrame including:
    - ColumnName
    - DataType
    - UniqueValues
    - NullCount
    - % EmptyCells (2 decimal places)

    Parameters:
        dataframe (pd.DataFrame): The dataset to analyze.

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
    Determine predominant data type and metadata for each column.

    Returns:
        dict: {
            column_name: {
                'inferred_type': 'numeric'|'string'|'mixed'|'empty',
                'original_dtype': 'object'|'float64'|...,
                'numeric_ratio': float,
                'string_ratio': float,
                'missing_pct': float
            }
        }
    """
    results = {}

    for column in dataframe.columns:
        col_data = dataframe[column].dropna()
        total = len(col_data)

        if total == 0:
            results[column] = {
                'inferred_type': 'empty',
                'original_dtype': dataframe[column].dtype.name,
                'numeric_ratio': 0,
                'string_ratio': 0,
                'missing_pct': 100
            }
            continue

        numeric_count = 0
        string_count = 0

        for val in col_data:
            if isinstance(val, str):
                try:
                    float(val)
                    numeric_count += 1
                except ValueError:
                    string_count += 1
            elif isinstance(val, (int, float, np.number)):
                numeric_count += 1
            else:
                string_count += 1  # fallback to string-like

        numeric_ratio = numeric_count / total
        string_ratio = string_count / total
        missing_pct = dataframe[column].isnull().mean() * 100

        if numeric_ratio > type_threshold:
            inferred_type = 'numeric'
        elif string_ratio > type_threshold:
            inferred_type = 'string'
        else:
            inferred_type = 'mixed'

        results[column] = {
            'inferred_type': inferred_type,
            'original_dtype': dataframe[column].dtype.name,
            'numeric_ratio': numeric_ratio,
            'string_ratio': string_ratio,
            'missing_pct': missing_pct
        }

    return results

def identify_incorrect_entries(dataframe):
    """
    Identify columns that contain a mixture of numeric and string entries.

    Parameters:
        dataframe (pd.DataFrame): The dataset to analyze.

    Returns:
        dict: {column_name: {string_count, numeric_count, string_indices, numeric_indices, string_ratio, numeric_ratio}}

    Example usage:
        analysis = identify_incorrect_entries(df)
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
            continue
            
        string_ratio = (len(string_indices) / total_valid)*100 if total_valid > 0 else 0
        numeric_ratio = (len(numeric_indices) / total_valid)*100 if total_valid > 0 else 0
        
        # Only consider if there's a meaningful mixture
        if 0 < string_ratio < 100 and 0 < numeric_ratio < 100:
            analysis_results[column] = {
                'string_count': len(string_indices),
                'numeric_count': len(numeric_indices),
                'string_indices': string_indices,
                'numeric_indices': numeric_indices,
                'string_ratio': string_ratio,  # store as percentage directly
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
    Determine if a column should be treated as categorical based on unique value ratio.

    Parameters:
        column (pd.Series)
        cat_threshold (float)

    Returns:
        bool

    Example usage:
        if should_treat_as_categorical(df['col'], 0.1):
            df['col'] = pd.Categorical(df['col'])
    """
    unique_ratio = column.nunique() / len(column)
    return unique_ratio < cat_threshold

def convert_to_categorical_types(dataframe, cat_threshold=0.1):
    """
    Convert suitable object-type columns to categorical.

    Parameters:
        dataframe (pd.DataFrame)
        cat_threshold (float)

    Returns:
        pd.DataFrame

    Example usage:
        df_categorical = convert_to_categorical_types(df, 0.1)
    """
    dataframe_converted = dataframe.copy()
    for col in dataframe_converted.select_dtypes(include=['object']).columns:
        if should_treat_as_categorical(dataframe_converted[col], cat_threshold):
            dataframe_converted[col] = pd.Categorical(dataframe_converted[col])
    return dataframe_converted

def analyze_object_columns(df):
    """Analyze object columns and suggest conversions"""
    suggestions = {}
    
    for col in df.select_dtypes(include=['object']).columns:
        non_null_values = df[col].dropna()
        if len(non_null_values) == 0:
            continue
            
        # Test conversion success rates
        numeric_success = 0
        datetime_success = 0
        
        for val in non_null_values:
            # Test numeric conversion
            try:
                pd.to_numeric(val)
                numeric_success += 1
            except:
                pass
            
            # Test datetime conversion  
            try:
                pd.to_datetime(val)
                datetime_success += 1
            except:
                pass
        
        total = len(non_null_values)
        numeric_pct = (numeric_success / total) * 100
        datetime_pct = (datetime_success / total) * 100
        string_pct = 100 - max(numeric_pct, datetime_pct)
        
        # Determine best conversion
        best_conversion = 'string'  # default
        confidence = string_pct
        
        if numeric_pct > 85:  # high confidence threshold
            best_conversion = 'numeric'
            confidence = numeric_pct
        elif datetime_pct > 85:
            best_conversion = 'datetime' 
            confidence = datetime_pct
        
        suggestions[col] = {
            'numeric_pct': numeric_pct,
            'datetime_pct': datetime_pct, 
            'string_pct': string_pct,
            'suggested': best_conversion,
            'confidence': confidence
        }
    
    return suggestions

def apply_conversions(df, conversion_choices):
    """Apply the selected conversions to the dataframe"""
    df_copy = df.copy()
    conversion_results = []
    
    for col, conversion_type in conversion_choices.items():
        if conversion_type == 'numeric':
            try:
                df_copy[col] = pd.to_numeric(df_copy[col], errors='coerce')
                conversion_results.append(f"{col}: object → numeric")
            except Exception as e:
                st.error(f"Failed to convert {col} to numeric: {str(e)}")
        
        elif conversion_type == 'datetime':
            try:
                df_copy[col] = pd.to_datetime(df_copy[col], errors='coerce')
                conversion_results.append(f"{col}: object → datetime")
            except Exception as e:
                st.error(f"Failed to convert {col} to datetime: {str(e)}")
        
        elif conversion_type == 'string':
            try:
                df_copy[col] = df_copy[col].astype('string')
                conversion_results.append(f"{col}: object → string")
            except Exception as e:
                st.error(f"Failed to convert {col} to string: {str(e)}")
    
    return df_copy, conversion_results
    
#####################################################################################################################################
#####################################################################################################################################
#####################################################################################################################################
### Code ###
#####################################################################################################################################
#####################################################################################################################################
#####################################################################################################################################

# Initialize session state variables
if 'selected_section' not in st.session_state:
    st.session_state.selected_section = "File Upload"  # Default section

if 'processed_df' not in st.session_state:
    st.session_state.processed_df = None
    
#####################################################################################################################################
# Sidebar Navigation
with st.sidebar:
    st.title("Data Cleaning Tool")
    
    # Navigation Menu
    st.subheader("Select Section")
    
    # Define sections
    sections = [
        "File Upload",
        "Data Overview", 
        "Mixed-Type Columns",
        "Object Conversion",
        "Optimize Analysis",
        "Handle Outliers",
        "Clean Text Data",
        "Clean Column Names",
        "Impute Missing Values",
        "Download Processed Data"
    ]
    
    # Create navigation buttons
    for section in sections:
        if st.button(
            section,
            key=f"nav_{section}",
            use_container_width=True,
            type="primary" if st.session_state.selected_section == section else "secondary"
        ):
            st.session_state.selected_section = section
            st.rerun()

# Main Content Area
st.header(f"{st.session_state.selected_section}")

#####################################################################################################################################
#####################################################################################################################################
# SECTION 1: File Upload Section
if st.session_state.selected_section == "File Upload":
    st.header("1. Upload Data")
    
    # Check if data is already loaded
    if st.session_state.processed_df is not None:
        df = st.session_state.processed_df
        
        # Show current dataset info
        st.success("Dataset already loaded!")
        
        st.subheader("Current Dataset Overview")
        st.write(f"**Dataset Shape:** {df.shape[0]} rows and {df.shape[1]} columns")
        
        st.subheader("Enhanced Information Table")
        info_df, columns_with_missing = generate_enhanced_information_table(df)
        # Calculate dynamic height based on number of columns
        dynamic_height = min(38 + (len(info_df) * 35) + 10, 400)  # Cap at 400px for initial view
        st.dataframe(info_df, height=dynamic_height, use_container_width=True)
        
        # Show warning if there are columns with missing values
        if columns_with_missing:
            st.warning(f"Found {len(columns_with_missing)} column(s) with missing values:")
            
            # Create a detailed list of columns with missing values
            missing_details = []
            for col_name, col_dtype in columns_with_missing:
                null_count = df[col_name].isnull().sum()
                null_pct = (null_count / len(df)) * 100
                missing_details.append({
                    'Column': col_name,
                    'Missing Count': null_count,
                    'Missing %': f"{null_pct:.2f}%"
                })
            
            missing_df = pd.DataFrame(missing_details)
            st.dataframe(missing_df, use_container_width=True, hide_index=True)
        
        st.info("Navigate to other sections using the sidebar to start cleaning your data.")
        
        # Option to upload a new file
        st.divider()
        st.subheader("Upload New Dataset")
        
        with st.expander("Click here to upload a different CSV file"):
            st.warning("**Warning:** Uploading a new file will replace your current dataset and **all cleaning work will be lost** unless you export it first from Section 10 (Download Processed Data).")
            
            uploaded_file = st.file_uploader("Choose a CSV file", type="csv", key="file_uploader_replace")
            
            if uploaded_file is not None:
                if st.button("Confirm Replace Dataset", type="primary"):
                    # Reset all session state related to data cleaning
                    st.session_state.processed_df = pd.read_csv(uploaded_file)
                    st.session_state.incorrect_entries_analysis = None
                    st.session_state.processed_columns = set()
                    
                    # Clear other session state variables
                    if 'mixed_cols' in st.session_state:
                        del st.session_state.mixed_cols
                    if 'object_suggestions' in st.session_state:
                        del st.session_state.object_suggestions
                    if 'text_analysis_done' in st.session_state:
                        del st.session_state.text_analysis_done
                    if 'column_analysis_done' in st.session_state:
                        del st.session_state.column_analysis_done
                    if 'impute_log' in st.session_state:
                        st.session_state.impute_log = []
                    if 'df_backup' in st.session_state:
                        st.session_state.df_backup = None
                    
                    st.success("New file loaded successfully! All previous work has been cleared.")
                    st.rerun()
    
    else:
        # No data loaded yet - show file uploader
        st.info("Upload a CSV file to get started with data cleaning.")
        
        uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
        
        if uploaded_file is not None:
            st.session_state.processed_df = pd.read_csv(uploaded_file)
            df = st.session_state.processed_df
            st.success("File successfully uploaded!")
            
            # Add the Enhanced Information Table and dataset shape here
            st.subheader("Dataset Overview")
            st.write(f"**Dataset Shape:** {df.shape[0]} rows and {df.shape[1]} columns")
            
            st.subheader("Enhanced Information Table")
            info_df, columns_with_missing = generate_enhanced_information_table(df)
            # Calculate dynamic height based on number of columns
            dynamic_height = min(38 + (len(info_df) * 35) + 10, 400)  # Cap at 400px for initial view
            st.dataframe(info_df, height=dynamic_height, use_container_width=True)
            
            # Show warning if there are columns with missing values
            if columns_with_missing:
                st.warning(f"Found {len(columns_with_missing)} column(s) with missing values")
                
                # Create a detailed list of columns with missing values
                missing_details = []
                for col_name, col_dtype in columns_with_missing:
                    null_count = df[col_name].isnull().sum()
                    null_pct = (null_count / len(df)) * 100
                    missing_details.append({
                        'Column': col_name,
                        'Missing Count': null_count,
                        'Missing %': f"{null_pct:.2f}%"
                    })
                
                missing_df = pd.DataFrame(missing_details)
                st.dataframe(missing_df, use_container_width=True, hide_index=True)
            
            st.info("Please select an option on the left to start cleaning your data.")
            
            
#####################################################################################################################################
# SECTION 2: Data Overview
elif st.session_state.selected_section == "Data Overview":
    df = st.session_state.processed_df
    if df is None:
        st.error("Please upload a file first!")
        st.stop()
    st.header("2. Data Overview")
    
    # Button to refresh all data overview content
    if st.button("Refresh Data Overview", use_container_width=True):
        pass  # No action needed; button press triggers a rerun automatically
    
    if st.session_state.processed_df is not None:
        # Enhanced Information Table
        st.subheader("Enhanced Information Table")
        info_df, columns_with_missing = generate_enhanced_information_table(st.session_state.processed_df)
        # Calculate dynamic height: header + (rows * row_height) + padding
        dynamic_height = min(38 + (len(info_df) * 35) + 10, 600)  # Cap at 600px max
        st.dataframe(info_df, height=dynamic_height)
        
        # Get the dataframe for the rest of the section
        df = st.session_state.processed_df
        
        st.write(f"Dataset Shape: {df.shape[0]} rows and {df.shape[1]} columns")
        
        st.subheader("First Five Rows")
        st.dataframe(df.head())
        
        # Missing Values Summary
        st.subheader("Missing Value Summary")
        
        # Add refresh button
        col1, col2 = st.columns([3, 1])
        with col2:
            if st.button("Refresh Summary", use_container_width=True):
                st.rerun()
        
        if total_missing == 0:
            st.success("No missing values in your dataset!")
            
            # Show undo button even when no missing values
            if st.button("Undo Last Imputation"):
                if st.session_state.df_backup is not None:
                    st.session_state.processed_df = st.session_state.df_backup.copy()
                    df = st.session_state.processed_df
                    if st.session_state.impute_log:
                        undone = st.session_state.impute_log.pop()
                        st.success(f"Undid imputation: {undone[1]} on '{undone[0]}'")
                        st.rerun()
                    else:
                        st.info("No previous imputation to undo.")
                else:
                    st.warning("No backup available to undo.")
            
            # Show log even when no missing values
            with st.expander("Imputation Log"):
                if st.session_state.impute_log:
                    st.write("**Recent imputation operations:**")
                    for idx, (col, method) in enumerate(reversed(st.session_state.impute_log[-10:]), 1):
                        st.write(f"{idx}. Column: **{col}**, Method: **{method}**")
                else:
                    st.write("No imputations logged yet.")
    else:
        st.write("No data loaded yet.")
        
#####################################################################################################################################
# SECTION 3: Identify and Clean Mixed-Type Columns
elif st.session_state.selected_section == "Mixed-Type Columns":
    df = st.session_state.processed_df
    if df is None:
        st.error("Please upload a file first!")
        st.stop()
    st.header("3. Resolve Data Type Conflicts")
    st.markdown("""
    This step scans for columns that contain a mix of numeric and string values, which can break analysis or machine learning workflows.
    
    Even a single rogue entry can cause issues, so we'll detect ANY mixed-type columns and let you choose how to handle them.
    """)
    
    def detect_mixed_columns(df):
        mixed_cols = {}
    
        for col in df.columns:
            if df[col].dtype != 'object':
                continue
    
            values = df[col].dropna()
            total = len(values)
            if total == 0:
                continue
    
            num_count = 0
            str_count = 0
            numeric_examples = []
            string_examples = []
    
            for val in values:
                try:
                    float(val)
                    num_count += 1
                    if len(numeric_examples) < 3:
                        numeric_examples.append(val)
                except:
                    str_count += 1
                    if len(string_examples) < 3:
                        string_examples.append(val)
    
            numeric_ratio = num_count / total
            string_ratio = str_count / total
    
            # Flag ANY column with both types (even 1 rogue entry)
            if 0 < numeric_ratio < 1:
                mixed_cols[col] = {
                    'numeric_ratio': numeric_ratio * 100,
                    'string_ratio': string_ratio * 100,
                    'numeric_count': num_count,
                    'string_count': str_count,
                    'total_count': total,
                    'numeric_examples': numeric_examples,
                    'string_examples': string_examples
                }
    
        return mixed_cols
    
    def apply_resolution_strategy(df, col, strategy):
        """Apply the chosen resolution strategy to a mixed-type column"""
        results = []
        df_copy = df.copy()
        
        if strategy == "convert_all_to_string":
            # Convert everything to string - safest option
            df_copy[col] = df_copy[col].astype(str)
            df_copy[col] = df_copy[col].replace('nan', np.nan)  # Keep NaNs as NaN
            results.append(f"All values in '{col}' converted to string")
            
        elif strategy == "force_to_numeric":
            # Try to convert everything to numeric, set non-convertible to NaN
            original_nulls = df_copy[col].isna().sum()
            df_copy[col] = pd.to_numeric(df_copy[col], errors='coerce')
            new_nulls = df_copy[col].isna().sum()
            failed_conversions = new_nulls - original_nulls
            results.append(f"'{col}' converted to numeric ({failed_conversions} non-numeric values → NaN)")
            
        elif strategy == "remove_strings":
            # Remove string values (set to NaN)
            removed_count = 0
            for idx, val in df_copy[col].items():
                if pd.isna(val):
                    continue
                try:
                    float(val)
                except:
                    df_copy.at[idx, col] = np.nan
                    removed_count += 1
            results.append(f"String values removed from '{col}' ({removed_count} values → NaN)")
            
        elif strategy == "remove_numbers":
            # Remove numeric values (set to NaN)
            removed_count = 0
            for idx, val in df_copy[col].items():
                if pd.isna(val):
                    continue
                try:
                    float(val)
                    df_copy.at[idx, col] = np.nan
                    removed_count += 1
                except:
                    pass
            results.append(f"Numeric values removed from '{col}' ({removed_count} values → NaN)")
        
        return df_copy, results
    
    if st.button("🔍 Scan for Mixed-Type Columns"):
        st.session_state.mixed_cols = detect_mixed_columns(df)
    
    if 'mixed_cols' in st.session_state:
        mixed_cols = st.session_state.mixed_cols
        if mixed_cols:
            st.subheader("Mixed-Type Columns Found")
            
            for col, info in mixed_cols.items():
                with st.expander(f"🔧 Column: '{col}' - {info['numeric_count']} numeric, {info['string_count']} string values", expanded=True):
                    
                    col1, col2 = st.columns([1, 1])
                    
                    with col1:
                        st.write("**Breakdown:**")
                        st.write(f"Numeric: {info['numeric_ratio']:.1f}% ({info['numeric_count']} values)")
                        st.write(f"String: {info['string_ratio']:.1f}% ({info['string_count']} values)")
                        
                        st.write("**Sample numeric values:**")
                        st.code(str(info['numeric_examples']))
                        
                        st.write("**Sample string values:**")
                        st.code(str(info['string_examples']))
                    
                    with col2:
                        st.write("**Choose Resolution Strategy:**")
                        
                        # Determine default based on dominant type
                        default_strategy = "remove_strings" if info['numeric_count'] > info['string_count'] else "remove_numbers"
                        
                        strategy = st.selectbox(
                            f"How should we handle '{col}'?",
                            options=[
                                "remove_strings",
                                "remove_numbers",
                                "convert_all_to_string",
                                "force_to_numeric"
                            ],
                            format_func=lambda x: {
                                "convert_all_to_string": "Convert all to string",
                                "force_to_numeric": "Force to numeric (strings → NaN)",
                                "remove_strings": "Remove string values (→ NaN)", 
                                "remove_numbers": "Remove numeric values (→ NaN)"
                            }[x],
                            index=0 if default_strategy == "remove_strings" else 1,
                            key=f"strategy_{col}"
                        )
                        
                        # Show impact preview
                        if strategy == "convert_all_to_string":
                            st.info("All data preserved as text")
                        elif strategy == "force_to_numeric":
                            st.warning(f"Will set {info['string_count']} string values to NaN")
                        elif strategy == "remove_strings":
                            st.error(f"Will remove {info['string_count']} string values")
                        elif strategy == "remove_numbers":
                            st.error(f"Will remove {info['numeric_count']} numeric values")
                        
                        # Confirmation and apply
                        if st.button(f"Apply to '{col}'", key=f"apply_{col}", type="primary"):
                            df_updated, results = apply_resolution_strategy(df, col, strategy)
                            
                            st.success("Changes applied!")
                            for result in results:
                                st.write(f"• {result}")
                            
                            # Update the main dataframe
                            st.session_state.processed_df = df_updated
                            df = df_updated
                            
                            # Remove from mixed_cols
                            if col in st.session_state.mixed_cols:
                                del st.session_state.mixed_cols[col]
                            
                            st.rerun()
        else:
            st.success("No mixed-type columns detected!")
            
#####################################################################################################################################
#####################################################################################################################################
#####################################################################################################################################
# SECTION 4: Smart Object Column Conversion
elif st.session_state.selected_section == "Object Conversion":
    df = st.session_state.processed_df
    if df is None:
        st.error("Please upload a file first!")
        st.stop()
    st.header("4. Smart Object Column Conversion")
    
    with st.expander("Why does data type matter?"):
        st.markdown("""
        ### What are 'object' columns?
        When pandas loads your CSV file, columns containing text are labeled as 'object' type by default. 
        However, many of these columns might actually contain numbers or dates disguised as text.
        
        ### Why should you care?
        Using the wrong data type causes several problems:
        
        **Memory inefficiency:**
        - Object columns use significantly more memory than numeric or datetime types
        - A column with "1", "2", "3" as text uses ~10x more memory than true integers
        
        **Broken functionality:**
        - Mathematical operations won't work (can't calculate averages, sums, etc.)
        - Sorting produces wrong results ("10" comes before "2" when sorted as text)
        - Visualizations and plots may fail or display incorrectly
        - Machine learning models cannot process object columns directly
        
        **Analysis errors:**
        - Statistical functions return errors or unexpected results
        - Filtering and comparisons behave unexpectedly
        - Date calculations are impossible on text-based dates
        
        ### What this tool does:
        This section scans your 'object' columns and determines whether they should actually be:
        - **Numeric** (integers or decimals for math operations)
        - **Datetime** (dates/times for time-based analysis)
        - **String** (true text data that should stay as text)
        
        Converting to the correct type unlocks proper functionality and improves performance.
        """)
    
    st.markdown("Automatically detect and convert object columns to their appropriate data types.")
    
    if st.button("Analyze Object Columns"):
        st.session_state.object_suggestions = analyze_object_columns(df)
    
    if 'object_suggestions' in st.session_state:
        suggestions = st.session_state.object_suggestions
        
        if suggestions:
            st.subheader("Column Analysis Results")
            
            conversion_choices = {}
            
            for col, info in suggestions.items():
                st.markdown(f"### {col}")
                
                col1, col2, col3 = st.columns([3, 2, 2])
                
                with col1:
                    # Visual breakdown
                    st.write("**Data Type Breakdown:**")
                    st.write(f"Numeric: {info['numeric_pct']:.1f}%")
                    st.write(f"Datetime: {info['datetime_pct']:.1f}%") 
                    st.write(f"String: {info['string_pct']:.1f}%")
                    
                    # Show sample values
                    sample_values = df[col].dropna().head(3).tolist()
                    st.write(f"**Sample values:** {sample_values}")
                    
                    # Add recommendation message
                    suggested = info['suggested']
                    confidence = info['confidence']
                    if confidence > 85:
                        confidence_text = "High confidence"
                        confidence_color = "success"
                    elif confidence > 60:
                        confidence_text = "Medium confidence"
                        confidence_color = "warning"
                    else:
                        confidence_text = "Low confidence"
                        confidence_color = "error"
                    
                    st.info(f"**Recommendation:** Convert column '{col}' to **{suggested}** ({confidence_text}: {confidence:.1f}%)")
                
                with col2:
                    conversion_choice = st.selectbox(
                        "Convert Object column to:",
                        options=['string', 'numeric', 'datetime'],
                        index=['string', 'numeric', 'datetime'].index(suggested),
                        key=f"convert_{col}",
                        help=f"Currently: object | Suggested: {suggested} (confidence: {info['confidence']:.1f}%)"
                    )
                    conversion_choices[col] = conversion_choice
                    
                    # Individual apply button for this column
                    if st.button(f"Apply to '{col}'", key=f"apply_{col}", type="secondary"):
                        df_converted, results = apply_conversions(df, {col: conversion_choice})
                        
                        if results:
                            st.success(f"Conversion applied to '{col}'!")
                            for result in results:
                                st.write(f"• {result}")
                            
                            # Update the main dataframe
                            st.session_state.processed_df = df_converted
                            df = df_converted  # Update local df reference
                            
                            # Remove this column from suggestions since it's been processed
                            if col in st.session_state.object_suggestions:
                                del st.session_state.object_suggestions[col]
                            
                            st.rerun()  # Refresh to show updated state
                        else:
                            st.info(f"No conversion was applied to '{col}'.")
                
                with col3:
                    # Calculate confidence based on selected conversion type
                    if conversion_choice == 'numeric':
                        display_confidence = info['numeric_pct']
                    elif conversion_choice == 'datetime':
                        display_confidence = info['datetime_pct']
                    else:  # string
                        display_confidence = info['string_pct']
                    
                    st.metric("Confidence", f"{display_confidence:.1f}%")
                    if display_confidence > 85:
                        st.success("High confidence")
                    elif display_confidence > 60:
                        st.warning("Medium confidence")
                    else:
                        st.error("Low confidence")
                
                st.divider()
            
            # Keep the bulk apply button as well for convenience
            if len(suggestions) > 1:  # Only show if multiple columns remain
                st.markdown("### Bulk Operations")
                if st.button("Apply All Selected Conversions", type="primary"):
                    df_converted, results = apply_conversions(df, conversion_choices)
                    
                    if results:
                        st.success("All conversions applied successfully!")
                        for result in results:
                            st.write(f"• {result}")
                        
                        # Update the main dataframe
                        st.session_state.processed_df = df_converted
                        
                        # Clear all suggestions since they've been processed
                        del st.session_state.object_suggestions
                        
                        st.rerun()  # Refresh to show updated state
                    else:
                        st.info("No conversions were applied.")
        
        else:
            st.success("No object columns found to convert.")
    
    # Float to Int conversion section
    st.divider()
    st.subheader("Convert float columns to integer (if safe)")
    
    with st.expander("Why convert floats to integers?"):
        st.markdown("""
        ### Benefits of using integers instead of floats:
        
        **Memory efficiency:**
        - Integers use approximately 50% less memory than floats
        - Important for large datasets or memory-constrained environments
        
        **Precision and accuracy:**
        - Integers have exact representation (no rounding errors)
        - Floats can introduce tiny rounding errors in calculations
        - Better for counting, IDs, categorical codes, and whole number data
        
        **Performance:**
        - Integer operations are faster than float operations
        - Comparisons and sorting are more efficient
        
        **Clarity:**
        - Makes it obvious the column contains whole numbers only
        - Prevents confusion about decimal precision
        
        ### When is it safe?
        This tool only converts float columns where **all values are whole numbers** (like 1.0, 2.0, 3.0).
        If any value has a decimal component (like 1.5), the column won't be converted.
        """)
    
    enable_floatint = st.checkbox("Enable float-to-int optimization", key="enable_floatint")
    
    if enable_floatint:
        safe_int_cols = []
        for col in df.select_dtypes(include=['float64']):
            if df[col].dropna().apply(float.is_integer).all():
                safe_int_cols.append(col)
    
        if safe_int_cols:
            st.write(f"**Columns that can be safely converted:** {', '.join(safe_int_cols)}")
            st.write("**Preview of data:**")
            st.dataframe(df[safe_int_cols].head(), use_container_width=True)
            
            if st.button("Convert to Int", key="btn_floatint"):
                for col in safe_int_cols:
                    df[col] = df[col].astype('Int64')
                    st.success(f"{col}: float64 → Int64")
                st.session_state.processed_df = df
        else:
            st.info("No float columns are safely convertible to integers.")

#####################################################################################################################################
#####################################################################################################################################
# SECTION 5: Optimize for Analysis
elif st.session_state.selected_section == "Optimize Analysis":
    df = st.session_state.processed_df
    if df is None:
        st.error("Please upload a file first!")
        st.stop()
    st.header("5. Optimize for Analysis")
    
    with st.expander("What does optimization do and why does it matter?"):
        st.markdown("""
        ### Overview
        This section helps reduce memory usage and improve performance by converting your data to more efficient storage formats.
        For large datasets, these optimizations can reduce memory usage by 50-90% and speed up analysis significantly.
        
        ---
        
        ### Convert Low-Cardinality Columns to 'Category'
        
        **What are low-cardinality columns?**
        - Columns with relatively few unique values compared to total rows
        - Examples: Country names (195 countries vs millions of rows), Status fields ("Active"/"Inactive"), Product categories
        
        **Why convert to category?**
        
        **Massive memory savings:**
        - String columns store the full text for every row
        - Category columns store each unique value once, then use small integer codes
        - Example: A column with 1 million rows and 5 unique values:
          - As string: ~8 MB
          - As category: ~1 MB (87% reduction)
        
        **Faster operations:**
        - Sorting, grouping, and filtering are significantly faster
        - Comparisons use integer codes instead of string matching
        
        **Better for analysis:**
        - Many visualization libraries automatically recognize categories
        - Statistical functions work more efficiently
        - Makes it explicit which columns are categorical variables
        
        **When to use:**
        - Columns where unique values < 10% of total rows (this tool's default threshold)
        - Status fields, categories, labels, codes, country/state names
        
        ---
        
        ### Downcast Numeric Types
        
        **What is downcasting?**
        - Converting numbers to smaller storage formats when possible
        - Examples: int64 → int32, int64 → int16, float64 → float32
        
        **Why downcast?**
        
        **Memory savings:**
        - int64 uses 8 bytes per value
        - int32 uses 4 bytes (50% reduction)
        - int16 uses 2 bytes (75% reduction)
        - For millions of rows, this adds up quickly
        
        **Faster processing:**
        - Smaller data types mean more data fits in CPU cache
        - Calculations on smaller types are faster
        - File I/O (reading/writing) is faster
        
        **When to use:**
        - When you know your numbers fit in smaller ranges:
          - int16: -32,768 to 32,767
          - int32: -2.1 billion to 2.1 billion
          - uint (unsigned): Only positive numbers, doubles the positive range
        
        **When NOT to use:**
        - If numbers might grow beyond the smaller type's range
        - If you need maximum precision for scientific calculations
        - When memory isn't a concern and you want to play it safe
        
        ---
        
        ### Summary
        Use these optimizations when:
        - Working with large datasets (100,000+ rows)
        - Memory is limited
        - You need faster processing
        - Preparing data for machine learning or repeated analysis
        
        Skip these optimizations when:
        - Dataset is small (< 10,000 rows)
        - Quick one-time analysis
        - Unsure about data value ranges
        """)
    
    st.markdown("""
    Final optimization for memory and performance:
    - Convert low-cardinality string columns to `category` (huge memory savings for repeated values)
    - Optionally downcast numeric types to smaller formats (e.g., int64 → int32)
    """)
    
    optimize_cats = st.checkbox("Convert low-cardinality string columns to 'category'", value=True)
    downcast_nums = st.checkbox("Downcast numeric types to smaller types", value=False)
    
    if st.button("Run Optimization"):
        optimization_report = []
        
        if optimize_cats:
            st.subheader("Category Conversion Results")
            converted_count = 0
            for col in df.select_dtypes(include='object'):
                if should_treat_as_categorical(df[col]):
                    unique_count = df[col].nunique()
                    total_count = len(df)
                    unique_ratio = unique_count / total_count
                    
                    # Calculate memory savings
                    memory_before = df[col].memory_usage(deep=True)
                    df[col] = df[col].astype('category')
                    memory_after = df[col].memory_usage(deep=True)
                    memory_saved = memory_before - memory_after
                    percent_saved = (memory_saved / memory_before) * 100
                    
                    optimization_report.append(f"{col}: object → category ({unique_count} unique values, {percent_saved:.1f}% memory saved)")
                    converted_count += 1
            
            if converted_count > 0:
                st.success(f"Converted {converted_count} column(s) to category")
                for report in optimization_report:
                    st.write(f"• {report}")
            else:
                st.info("No columns met the criteria for category conversion")
    
        if downcast_nums:
            st.subheader("Numeric Downcasting Results")
            downcast_report = []
            for col in df.select_dtypes(include=['int64', 'float64']):
                original_dtype = df[col].dtype
                df[col] = pd.to_numeric(df[col], downcast='unsigned' if df[col].min() >= 0 else 'integer')
                new_dtype = df[col].dtype
                
                if original_dtype != new_dtype:
                    downcast_report.append(f"{col}: {original_dtype} → {new_dtype}")
            
            if downcast_report:
                st.success(f"Downcasted {len(downcast_report)} column(s)")
                for report in downcast_report:
                    st.write(f"• {report}")
            else:
                st.info("No numeric columns could be downcasted further")
    
        st.session_state.processed_df = df
        
        if optimize_cats or downcast_nums:
            st.success("Optimization complete!")

#####################################################################################################################################
#####################################################################################################################################
# SECTION 6: Detect and Handle Outliers
elif st.session_state.selected_section == "Handle Outliers":
    df = st.session_state.processed_df
    if df is None:
        st.error("Please upload a file first!")
        st.stop()
    st.header("6. Detect and Handle Outliers")
    with st.expander("What are outliers and how does this work?"):
        st.markdown("""
        ### What are outliers?
        Outliers are data points that are unusually high or low compared to the rest of your data. 
        For example, if most salaries in your dataset are between $30,000-$80,000, but you have one entry 
        of $500,000, that would be an outlier.
        
        ### How does this work?
        This tool uses a statistical method called the **IQR (Interquartile Range) method** to find outliers:
        
        1. **Q1 (25th percentile)**: The value below which 25% of your data falls
        2. **Q3 (75th percentile)**: The value below which 75% of your data falls  
        3. **IQR**: The difference between Q3 and Q1 (captures the "middle 50%" of your data)
        
        **Outliers are defined as:**
        - Values below: Q1 - (multiplier × IQR)
        - Values above: Q3 + (multiplier × IQR)
        
        ### Instructions:
        1. **Adjust the multiplier** below (1.5 is standard - lower values find more outliers, higher values find fewer)
        2. **Click "Run Outlier Detection"** to scan your numeric columns
        3. **For each column with outliers, choose:**
           - **Replace with NaN**: Keeps the rows but marks outlier values as missing
           - **Drop rows**: Completely removes rows containing outliers
           - **Skip**: Leave the outliers as-is for now
        
         **Tip**: Start with the default multiplier of 1.5. You can always run this multiple times with different settings.
        """)
    
    iqr_threshold = st.slider(
        "IQR Multiplier (default = 1.5)", min_value=1.0, max_value=5.0, value=1.5, step=0.1
    )
    
    # Initialize outlier report in session state if not exists
    if 'outlier_report' not in st.session_state:
        st.session_state.outlier_report = None
    
    if st.button("Run Outlier Detection"):
        df = st.session_state.processed_df.copy()
        outlier_report = {}
        
        for col in df.select_dtypes(include=[np.number]).columns:
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - iqr_threshold * IQR
            upper_bound = Q3 + iqr_threshold * IQR
            mask = (df[col] < lower_bound) | (df[col] > upper_bound)
            count = mask.sum()
            if count > 0:
                outlier_report[col] = {
                    "lower": lower_bound,
                    "upper": upper_bound,
                    "count": count,
                    "percent": 100 * count / len(df),
                    "mask": mask
                }
        
        st.session_state.outlier_report = outlier_report
    
    # Display outlier report if it exists
    if st.session_state.outlier_report is not None:
        outlier_report = st.session_state.outlier_report
        df = st.session_state.processed_df
        
        if outlier_report:
            st.success(f"Found {len(outlier_report)} column(s) with outliers.")
            
            for col, stats in outlier_report.items():
                st.markdown(f"**Column: `{col}`**")
                st.write(f"- Outliers: {stats['count']} ({stats['percent']:.2f}%)")
                st.write(f"- Lower bound: {stats['lower']:.2f}")
                st.write(f"- Upper bound: {stats['upper']:.2f}")
                
                # Show outlier rows
                outlier_rows = df[stats["mask"]]
                if len(outlier_rows) > 0:
                    st.markdown(f"**Sample outlier rows for `{col}`:**")
                    # Display up to 5 rows with scrolling if more exist
                    st.dataframe(
                        outlier_rows, 
                        height=min(200, len(outlier_rows) * 35 + 38),  # Dynamic height based on rows
                        use_container_width=True
                    )
                    if len(outlier_rows) > 5:
                        st.caption(f"Showing all {len(outlier_rows)} outlier rows. Scroll within the table to see more.")
                    else:
                        st.caption(f"Showing all {len(outlier_rows)} outlier rows.")
                
                c1, c2, c3 = st.columns(3)
        
                with c1:
                    if st.button(f"Replace outliers in '{col}' with NaN", key=f"nan_{col}"):
                        df_copy = st.session_state.processed_df.copy()
                        df_copy.loc[stats["mask"], col] = np.nan
                        st.session_state.processed_df = df_copy
                        
                        # Remove this column from outlier report
                        del st.session_state.outlier_report[col]
                        
                        st.success(f"Replaced {stats['count']} outliers in '{col}' with NaN")
                        st.rerun()
        
                with c2:
                    if st.button(f"Drop rows with outliers in '{col}'", key=f"drop_{col}"):
                        df_copy = st.session_state.processed_df.copy()
                        df_copy = df_copy[~stats["mask"]]
                        st.session_state.processed_df = df_copy
                        
                        # Clear entire outlier report since row indices have changed
                        st.session_state.outlier_report = None
                        
                        st.success(f"Dropped {stats['count']} rows with outliers in '{col}'")
                        st.rerun()
        
                with c3:
                    if st.button(f"Skip column '{col}'", key=f"skip_{col}"):
                        # Remove this column from outlier report
                        del st.session_state.outlier_report[col]
                        
                        st.info(f"Skipped handling outliers in '{col}'")
                        st.rerun()
        else:
            st.success("No outliers detected using the current IQR threshold.")
            st.session_state.outlier_report = None
            
#####################################################################################################################################
#####################################################################################################################################
# SECTION 7: Clean and Normalize Text Data
elif st.session_state.selected_section == "Clean Text Data":
    df = st.session_state.processed_df
    if df is None:
        st.error("Please upload a file first!")
        st.stop()
    st.header("7. Clean and Normalize Text Data")
    st.markdown("""
    Clean up string contents to reduce noise and standardize formatting:
    - Strip leading/trailing spaces
    - Convert to lowercase
    - Replace common 'null-like' values (e.g., "NA", "n/a", "--")
    - Remove invisible characters (e.g., \\xa0, \\u200b)
    """)
    
    # Initialize or get text analysis from session state
    if 'text_analysis_done' not in st.session_state:
        st.session_state.text_analysis_done = False
        st.session_state.text_columns = []
        st.session_state.text_overview = []
    
    # Text Detection Button
    col1, col2 = st.columns([1, 3])
    with col1:
        detect_button = st.button("Run Text Detection", type="secondary")
    with col2:
        if st.session_state.text_analysis_done:
            st.write(f"Analysis complete - {len(st.session_state.text_columns)} text column(s) need cleaning")
        else:
            st.write("Click to analyze text columns in your current dataset")
    
    if detect_button or not st.session_state.text_analysis_done:
        # Get text columns from the current dataframe
        df = st.session_state.processed_df
        all_text_columns = df.select_dtypes(include=['object', 'string']).columns.tolist()
        
        if not all_text_columns:
            st.info("No text columns found in your current dataset.")
            st.session_state.text_columns = []
            st.session_state.text_overview = []
            st.session_state.text_analysis_done = True
        else:
            # Filter to only columns that actually need cleaning
            text_columns_needing_cleanup = []
            text_overview = []
            
            for col in all_text_columns:
                non_null_count = df[col].notna().sum()
                unique_count = df[col].nunique()
                
                # Check for issues that need cleaning
                issues_found = False
                
                # Check for null-like strings
                null_like_patterns = ["NA", "N/A", "na", "n/a", "--", "-", "", "null", "NULL", "Null"]
                null_like_count = df[col].isin(null_like_patterns).sum()
                
                # Check for whitespace issues
                whitespace_issues = 0
                if non_null_count > 0:
                    whitespace_issues = df[col].astype(str).apply(lambda x: x != x.strip()).sum()
                
                # Check for mixed case (could benefit from lowercase conversion)
                mixed_case_count = 0
                if non_null_count > 0:
                    sample = df[col].dropna().astype(str)
                    if len(sample) > 0:
                        mixed_case_count = sum(1 for val in sample if val != val.lower() and val != val.upper())
                
                # Check for invisible characters
                invisible_chars_count = 0
                if non_null_count > 0:
                    import re
                    invisible_chars_count = df[col].astype(str).apply(
                        lambda x: bool(re.search(r'[\u200b\xa0\u2000-\u200f\u2028-\u202f]', str(x)))
                    ).sum()
                
                # Determine if this column needs cleaning
                if null_like_count > 0 or whitespace_issues > 0 or mixed_case_count > 0 or invisible_chars_count > 0:
                    issues_found = True
                    text_columns_needing_cleanup.append(col)
                    
                    # Sample values (first 3 non-null unique values)
                    sample_values = df[col].dropna().unique()[:3].tolist()
                    sample_str = ", ".join([f'"{str(val)}"' for val in sample_values])
                    if len(sample_values) == 3 and unique_count > 3:
                        sample_str += "..."
                    
                    # Build issues summary
                    issues_list = []
                    if null_like_count > 0:
                        issues_list.append(f"{null_like_count} null-like strings")
                    if whitespace_issues > 0:
                        issues_list.append(f"{whitespace_issues} whitespace issues")
                    if mixed_case_count > 0:
                        issues_list.append(f"{mixed_case_count} mixed case")
                    if invisible_chars_count > 0:
                        issues_list.append(f"{invisible_chars_count} invisible chars")
                    
                    issues_summary = ", ".join(issues_list)
                    
                    text_overview.append({
                        'Column': col,
                        'Non-null Values': non_null_count,
                        'Unique Values': unique_count,
                        'Issues Found': issues_summary,
                        'Sample Values': sample_str
                    })
            
            # Store results in session state
            st.session_state.text_columns = text_columns_needing_cleanup
            st.session_state.text_overview = text_overview
            st.session_state.text_analysis_done = True
            
            if text_columns_needing_cleanup:
                st.success(f"Text detection complete! Found {len(text_columns_needing_cleanup)} text column(s) that need cleaning")
            else:
                st.success("Text detection complete! All text columns are already clean")
    
    # Show results if analysis has been done
    if st.session_state.text_analysis_done and st.session_state.text_columns:
        st.subheader("Text Data Overview")
        
        overview_df = pd.DataFrame(st.session_state.text_overview)
        st.dataframe(overview_df, use_container_width=True)
        
        # Column selection
        st.subheader("Select Columns to Clean")
        selected_columns = st.multiselect(
            "Choose which text columns to clean:",
            options=st.session_state.text_columns,
            default=st.session_state.text_columns,
            help="Select the columns you want to apply text cleaning to"
        )
        
        if selected_columns:
            # Cleanup options
            st.subheader("Choose Cleanup Actions")
            text_opts = st.multiselect(
                "Select text cleanup actions:",
                options=[
                    "Strip leading/trailing whitespace",
                    "Convert to lowercase",
                    "Replace common null-like strings",
                    "Remove invisible characters"
                ],
                default=[
                    "Strip leading/trailing whitespace",
                    "Replace common null-like strings"
                ],
                help="Choose which cleaning operations to apply to the selected columns"
            )
            
            if text_opts:
                # Preview section
                st.subheader("Preview Changes")
                
                preview_col = st.selectbox(
                    "Select a column to preview changes:",
                    options=selected_columns,
                    help="See how the cleanup will affect this column"
                )
                
                if preview_col:
                    # Create preview of changes
                    preview_df = df[preview_col].copy().astype(str)
                    original_preview = preview_df.copy()
                    
                    # Apply selected transformations to preview
                    if "Strip leading/trailing whitespace" in text_opts:
                        preview_df = preview_df.str.strip()
                    
                    if "Convert to lowercase" in text_opts:
                        preview_df = preview_df.str.lower()
                    
                    if "Replace common null-like strings" in text_opts:
                        null_like_patterns = ["NA", "N/A", "na", "n/a", "--", "-", "", "null", "NULL", "Null"]
                        preview_df = preview_df.replace(null_like_patterns, pd.NA)
                    
                    if "Remove invisible characters" in text_opts:
                        import re
                        preview_df = preview_df.apply(lambda x: re.sub(r'[\u200b\xa0\u2000-\u200f\u2028-\u202f]', '', str(x)) if pd.notna(x) else x)
                    
                    # Show before/after comparison
                    changes_mask = (original_preview != preview_df) | (original_preview.isna() != preview_df.isna())
                    if changes_mask.any():
                        st.write(f"**Changes detected in '{preview_col}':**")
                        
                        # Show sample of changes
                        changed_indices = changes_mask[changes_mask].index[:10]  # Show up to 10 examples
                        
                        comparison_data = []
                        for idx in changed_indices:
                            comparison_data.append({
                                'Row': idx,
                                'Before': repr(original_preview.iloc[idx]) if pd.notna(original_preview.iloc[idx]) else 'NaN',
                                'After': repr(preview_df.iloc[idx]) if pd.notna(preview_df.iloc[idx]) else 'NaN'
                            })
                        
                        comparison_df = pd.DataFrame(comparison_data)
                        st.dataframe(comparison_df, use_container_width=True)
                        
                        # Summary stats
                        total_changes = changes_mask.sum()
                        total_rows = len(preview_df)
                        st.write(f"**Summary:** {total_changes:,} out of {total_rows:,} rows will be changed ({total_changes/total_rows*100:.1f}%)")
                        
                    else:
                        st.info(f"No changes will be made to '{preview_col}' with the selected options.")
                
                # Apply button
                st.subheader("Apply Changes")
                
                col1, col2 = st.columns([1, 3])
                with col1:
                    apply_button = st.button("Apply Text Cleanup", type="primary")
                with col2:
                    st.write(f"Will clean {len(selected_columns)} column(s) with {len(text_opts)} operation(s)")
                
                if apply_button:
                    df_cleaned = st.session_state.processed_df.copy()
                    changes_summary = {}
                    
                    for col in selected_columns:
                        original = df_cleaned[col].copy()
                        changes_in_col = 0
                        
                        if "Strip leading/trailing whitespace" in text_opts:
                            before_strip = df_cleaned[col].astype(str)
                            df_cleaned[col] = before_strip.str.strip()
                            changes_in_col += (before_strip != df_cleaned[col]).sum()
                        
                        if "Convert to lowercase" in text_opts:
                            before_lower = df_cleaned[col].astype(str)
                            df_cleaned[col] = before_lower.str.lower()
                            changes_in_col += (before_lower != df_cleaned[col]).sum()
                        
                        if "Replace common null-like strings" in text_opts:
                            before_null_replace = df_cleaned[col].copy()
                            null_like_patterns = ["NA", "N/A", "na", "n/a", "--", "-", "", "null", "NULL", "Null"]
                            df_cleaned[col] = df_cleaned[col].replace(null_like_patterns, np.nan)
                            changes_in_col += (before_null_replace.fillna('__NULL__') != df_cleaned[col].fillna('__NULL__')).sum()
                        
                        if "Remove invisible characters" in text_opts:
                            import re
                            before_invisible = df_cleaned[col].astype(str)
                            df_cleaned[col] = before_invisible.apply(lambda x: re.sub(r'[\u200b\xa0\u2000-\u200f\u2028-\u202f]', '', x))
                            changes_in_col += (before_invisible != df_cleaned[col]).sum()
                        
                        if changes_in_col > 0:
                            changes_summary[col] = changes_in_col
                    
                    # Update session state
                    st.session_state.processed_df = df_cleaned
                    
                    # Reset text analysis to allow re-detection
                    st.session_state.text_analysis_done = False
                    
                    # Show results
                    if changes_summary:
                        st.success(f"Text cleanup completed! Modified {len(changes_summary)} column(s)")
                        
                        # Detailed summary
                        st.write("**Changes made:**")
                        for col, change_count in changes_summary.items():
                            st.write(f"- **{col}**: {change_count:,} values modified")
                        
                        # Show sample of cleaned data
                        st.write("**Sample of cleaned data:**")
                        st.dataframe(df_cleaned[list(changes_summary.keys())].head(10), use_container_width=True)
                        
                    else:
                        st.info("No changes were made during text cleanup with the selected options.")
    elif st.session_state.text_analysis_done and not st.session_state.text_columns:
        st.success("All text columns in your dataset are already clean! No issues detected.")
        st.info("If you've made changes to your data, click 'Run Text Detection' again to re-analyze.")
    else:
        st.info("Click 'Run Text Detection' to analyze your dataset and find text columns that need cleaning.")

#####################################################################################################################################
# SECTION 8: Clean Column Names
elif st.session_state.selected_section == "Clean Column Names":
    df = st.session_state.processed_df
    if df is None:
        st.error("Please upload a file first!")
        st.stop()
    st.header("8. Clean Column Names")
    st.markdown("""
    Fix inconsistent column naming:
    - Remove leading/trailing whitespace
    - Standardize format (e.g., `snake_case`)
    - Remove special characters
    """)
    
    # Initialize or get column analysis from session state
    if 'column_analysis_done' not in st.session_state:
        st.session_state.column_analysis_done = False
        st.session_state.column_issues = {}
        st.session_state.current_columns = []
    
    # Column Analysis Button
    col1, col2 = st.columns([1, 3])
    with col1:
        analyze_button = st.button("🔍 Analyze Column Names", type="secondary")
    with col2:
        if st.session_state.column_analysis_done:
            issues_found = sum(len(issues) for issues in st.session_state.column_issues.values())
            st.write(f"Analysis complete - {issues_found} issue(s) detected across {len(st.session_state.current_columns)} columns")
        else:
            st.write("Click to analyze column names for potential cleanup issues")
    
    if analyze_button or not st.session_state.column_analysis_done:
        # Get current dataframe
        df = st.session_state.processed_df
        current_columns = df.columns.tolist()
        
        # Analyze column names for issues
        column_issues = {}
        import re
        
        for col in current_columns:
            issues = []
            
            # Check for whitespace issues
            if col != col.strip():
                issues.append("Leading/trailing whitespace")
            
            # Check for special characters
            if re.search(r'[^\w\s]', col):
                special_chars = re.findall(r'[^\w\s]', col)
                issues.append(f"Special characters: {', '.join(set(special_chars))}")
            
            # Check for inconsistent casing/spacing
            if not re.match(r'^[a-z][a-z0-9_]*$', col):
                if col != col.lower():
                    issues.append("Mixed case")
                if ' ' in col:
                    issues.append("Contains spaces")
                if col.startswith('_') or col.endswith('_'):
                    issues.append("Leading/trailing underscore")
            
            column_issues[col] = issues
        
        # Store results in session state
        st.session_state.column_analysis_done = True
        st.session_state.column_issues = column_issues
        st.session_state.current_columns = current_columns
        
        # Show analysis summary
        total_issues = sum(len(issues) for issues in column_issues.values())
        if total_issues > 0:
            st.warning(f"Found {total_issues} naming issue(s) across {len([col for col, issues in column_issues.items() if issues])} column(s)")
        else:
            st.success("All column names look clean!")
    
    # Show detailed analysis if completed
    if st.session_state.column_analysis_done:
        st.subheader("Column Name Analysis")
        
        # Show current column names with issues
        problematic_cols = [col for col, issues in st.session_state.column_issues.items() if issues]
        clean_cols = [col for col, issues in st.session_state.column_issues.items() if not issues]
        
        col2, col1 = st.columns(2)
        
        with col1:
            st.write("**Columns with issues:**")
            if problematic_cols:
                for col in problematic_cols:
                    issues = st.session_state.column_issues[col]
                    st.write(f"• `{col}` - {', '.join(issues)}")
            else:
                st.write("None")
        
        with col2:
            st.write("**Clean columns:**")
            if clean_cols:
                for col in clean_cols:
                    st.write(f"• `{col}`")
            else:
                st.write("None")
        
        # Show cleanup options only if there are issues or user wants to standardize
        total_issues = sum(len(issues) for issues in st.session_state.column_issues.values())
        
        if total_issues > 0 or st.checkbox("🔧 Show cleanup options anyway"):
            st.subheader("Cleanup Options")
            
            rename_opts = st.multiselect(
                "Choose column name cleanup actions:",
                options=[
                    "Strip whitespace",
                    "Remove special characters", 
                    "Standardize to snake_case"
                ],
                default=["Strip whitespace", "Remove special characters", "Standardize to snake_case"],
                help="Select which cleanup operations to apply to column names"
            )
            
            if rename_opts:
                # Preview cleanup
                st.subheader("Preview Changes")
                
                def clean_column_name(name):
                    import re
                    cleaned = name
                    if "Strip whitespace" in rename_opts:
                        cleaned = cleaned.strip()
                    if "Remove special characters" in rename_opts:
                        cleaned = re.sub(r'[^\w\s]', '', cleaned)
                    if "Standardize to snake_case" in rename_opts:
                        cleaned = re.sub(r'\s+', '_', cleaned).lower()
                    return cleaned
                
                # Show before/after column names
                changes_list = []
                cleaned_names = []
                
                for col in st.session_state.current_columns:
                    cleaned = clean_column_name(col)
                    cleaned_names.append(cleaned)
                    if col != cleaned:
                        changes_list.append((col, cleaned))
                
                if changes_list:
                    st.write("**Column name changes:**")
                    for original, cleaned in changes_list:
                        st.write(f"• `{original}` → `{cleaned}`")
                    
                    st.write(f"**Summary:** {len(changes_list)} out of {len(st.session_state.current_columns)} columns will be renamed")
                else:
                    st.info("No column names will be changed with the selected options.")
                
                # Check for potential duplicates after cleaning
                cleaned_counts = pd.Series(cleaned_names).value_counts()
                duplicates = cleaned_counts[cleaned_counts > 1].index.tolist()
                
                # Warning for duplicates
                if duplicates:
                    st.error(f"**Warning**: The following cleaned names would create duplicates: {', '.join(duplicates)}. Consider adjusting your cleanup options or renaming manually.")
                
                # Apply button
                st.subheader("Apply Changes")
                
                col1, col2 = st.columns([1, 3])
                with col1:
                    apply_button = st.button("Apply Column Name Cleanup", type="primary", disabled=bool(duplicates))
                with col2:
                    if duplicates:
                        st.write("Cannot apply due to duplicate names")
                    else:
                        st.write(f"Will rename {len(changes_list)} column(s)")
                
                if apply_button and not duplicates:
                    # Apply the changes
                    df = st.session_state.processed_df.copy()
                    old_names = df.columns.tolist()
                    new_names = [clean_column_name(col) for col in old_names]
                    
                    # Create rename mapping
                    rename_mapping = {old: new for old, new in zip(old_names, new_names) if old != new}
                    
                    # Apply rename
                    df = df.rename(columns=rename_mapping)
                    st.session_state.processed_df = df
                    
                    # Reset analysis state to force re-analysis
                    st.session_state.column_analysis_done = False
                    
                    # Show success message
                    if rename_mapping:
                        st.success(f"Column names cleaned! {len(rename_mapping)} column(s) renamed.")
                        
                        # Show what was changed
                        st.write("**Changes made:**")
                        for old_name, new_name in rename_mapping.items():
                            st.write(f"• `{old_name}` → `{new_name}`")
                    else:
                        st.info("No column names needed to be changed.")
                        
                    # Encourage re-analysis
                    st.info("Run 'Analyze Column Names' again to verify the cleanup results!")
        
        else:
            st.info("No issues found with your column names. They look good!")
    else:
        st.info("Click 'Analyze Column Names' to check your column names for potential issues.")


#####################################################################################################################################
#####################################################################################################################################
#####################################################################################################################################
#####################################################################################################################################
# SECTION 9: Impute Missing Values
elif st.session_state.selected_section == "Impute Missing Values":
    df = st.session_state.processed_df
    if df is None:
        st.error("Please upload a file first!")
        st.stop()
    st.header("9. Impute Missing Values")
    
    # ========== SIMPLE IMPUTATION HELPER FUNCTIONS ==========
    
    def apply_knn_imputation_simple(df, target_column, n_neighbors=5):
        """
        Apply KNN imputation to a numeric target column.
        Uses all other numeric columns as features.
        """
        try:
            from sklearn.impute import KNNImputer
            
            # Get all numeric columns
            numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            
            if target_column not in numeric_cols:
                return df, "Target column must be numeric for KNN imputation."
            
            if len(numeric_cols) < 2:
                return df, "Need at least 2 numeric columns for KNN imputation."
            
            # Check if target has any non-null values
            if df[target_column].notna().sum() < 1:
                return df, "Target column is completely empty."
            
            # Work with only numeric columns
            df_numeric = df[numeric_cols].copy()
            
            # Adjust n_neighbors based on available data
            non_null_count = df_numeric[target_column].notna().sum()
            actual_neighbors = min(n_neighbors, max(1, non_null_count - 1))
            
            # Apply KNN
            imputer = KNNImputer(n_neighbors=actual_neighbors)
            imputed_array = imputer.fit_transform(df_numeric)
            
            # Extract only the target column from imputed results
            target_col_idx = numeric_cols.index(target_column)
            imputed_target = imputed_array[:, target_col_idx]
            
            # Update only the missing values in original dataframe
            result_df = df.copy()
            missing_mask = df[target_column].isna()
            result_df.loc[missing_mask, target_column] = imputed_target[missing_mask]
            
            filled_count = missing_mask.sum()
            return result_df, f"KNN imputation successful! Filled {filled_count} missing values using {len(numeric_cols)-1} numeric features."
            
        except Exception as e:
            return df, f"KNN imputation failed: {str(e)}"
    
    def apply_linear_regression_simple(df, target_column):
        """
        Apply Linear Regression imputation to a numeric target column.
        """
        try:
            from sklearn.linear_model import LinearRegression
            
            # Get all numeric columns
            numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            
            if target_column not in numeric_cols:
                return df, "Target column must be numeric for Linear Regression."
            
            feature_cols = [col for col in numeric_cols if col != target_column]
            
            if len(feature_cols) < 1:
                return df, "Need at least 1 other numeric column for Linear Regression."
            
            # Split into training (complete target) and prediction (missing target) sets
            missing_mask = df[target_column].isna()
            train_df = df[~missing_mask].copy()
            predict_df = df[missing_mask].copy()
            
            # Find features with enough complete data
            usable_features = []
            for col in feature_cols:
                if train_df[col].notna().sum() >= 2:
                    usable_features.append(col)
            
            if len(usable_features) < 1:
                return df, "Not enough complete feature data for Linear Regression."
            
            # Get clean training data
            train_clean = train_df.dropna(subset=usable_features)
            
            if len(train_clean) < 2:
                return df, "Not enough complete training rows for Linear Regression."
            
            # Train model
            X_train = train_clean[usable_features]
            y_train = train_clean[target_column]
            
            model = LinearRegression()
            model.fit(X_train, y_train)
            
            # Predict on rows with missing target (that have complete features)
            predict_clean = predict_df.dropna(subset=usable_features)
            
            if len(predict_clean) == 0:
                return df, "No rows can be imputed (all missing target rows also have missing features)."
            
            X_predict = predict_clean[usable_features]
            predictions = model.predict(X_predict)
            
            # Update original dataframe
            result_df = df.copy()
            result_df.loc[predict_clean.index, target_column] = predictions
            
            return result_df, f"Linear Regression successful! Filled {len(predictions)} missing values using {len(usable_features)} features."
            
        except Exception as e:
            return df, f"Linear Regression failed: {str(e)}"
    
    def apply_mice_simple(df, target_column, max_iter=10):
        """
        Apply MICE imputation to a numeric target column.
        """
        try:
            if IterativeImputer is None:
                return df, "IterativeImputer not available. Update sklearn to 0.21+"
            
            # Get all numeric columns
            numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            
            if target_column not in numeric_cols:
                return df, "Target column must be numeric for MICE."
            
            if len(numeric_cols) < 2:
                return df, "Need at least 2 numeric columns for MICE."
            
            # Work with numeric columns only
            df_numeric = df[numeric_cols].copy()
            
            # Apply MICE
            imputer = IterativeImputer(max_iter=max_iter, random_state=0)
            imputed_array = imputer.fit_transform(df_numeric)
            
            # Extract target column
            target_col_idx = numeric_cols.index(target_column)
            imputed_target = imputed_array[:, target_col_idx]
            
            # Update only missing values
            result_df = df.copy()
            missing_mask = df[target_column].isna()
            result_df.loc[missing_mask, target_column] = imputed_target[missing_mask]
            
            filled_count = missing_mask.sum()
            return result_df, f"MICE imputation successful! Filled {filled_count} missing values."
            
        except Exception as e:
            return df, f"MICE imputation failed: {str(e)}"
    
    def apply_missforest_simple(df, target_column):
        """
        Apply MissForest imputation to a numeric target column.
        """
        try:
            if MissForest is None:
                return df, "MissForest not available. Install with: pip install missingpy"
            
            # Get all numeric columns
            numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            
            if target_column not in numeric_cols:
                return df, "Target column must be numeric for MissForest."
            
            if len(numeric_cols) < 2:
                return df, "Need at least 2 numeric columns for MissForest."
            
            # Work with numeric columns only
            df_numeric = df[numeric_cols].copy()
            
            # Apply MissForest
            imputer = MissForest(random_state=0)
            imputed_array = imputer.fit_transform(df_numeric)
            
            # Extract target column
            target_col_idx = numeric_cols.index(target_column)
            imputed_target = imputed_array[:, target_col_idx]
            
            # Update only missing values
            result_df = df.copy()
            missing_mask = df[target_column].isna()
            result_df.loc[missing_mask, target_column] = imputed_target[missing_mask]
            
            filled_count = missing_mask.sum()
            return result_df, f"MissForest imputation successful! Filled {filled_count} missing values."
            
        except Exception as e:
            return df, f"MissForest imputation failed: {str(e)}"
    
    def apply_interpolation_simple(df, target_column):
        """
        Apply interpolation to a numeric target column.
        """
        try:
            result_df = df.copy()
            missing_count = df[target_column].isna().sum()
            
            # Apply interpolation
            result_df[target_column] = result_df[target_column].interpolate(method='linear')
            
            filled_count = missing_count - result_df[target_column].isna().sum()
            return result_df, f"Interpolation successful! Filled {filled_count} missing values."
            
        except Exception as e:
            return df, f"Interpolation failed: {str(e)}"
    
    # ========== END OF HELPER FUNCTIONS ==========
    
    # Initialize session state
    if 'impute_log' not in st.session_state:
        st.session_state.impute_log = []
    if 'df_backup' not in st.session_state:
        st.session_state.df_backup = None
    
    df = st.session_state.processed_df
    
    if df is not None:
        missing_stats = df.isnull().sum()
        total_missing = missing_stats.sum()
        missing_cols = missing_stats[missing_stats > 0]
    
        st.subheader("Missing Value Summary")
        
        # Add refresh button
        col1, col2 = st.columns([3, 1])
        with col2:
            if st.button("Refresh Summary", use_container_width=True):
                st.rerun()
        
        if total_missing == 0:
            st.success("No missing values in your dataset!")
            
            # Show undo button even when no missing values
            if st.button("Undo Last Imputation"):
                if st.session_state.df_backup is not None:
                    st.session_state.processed_df = st.session_state.df_backup.copy()
                    df = st.session_state.processed_df
                    if st.session_state.impute_log:
                        undone = st.session_state.impute_log.pop()
                        st.success(f"Undid imputation: {undone[1]} on '{undone[0]}'")
                        st.rerun()
                    else:
                        st.info("No previous imputation to undo.")
                else:
                    st.warning("No backup available to undo.")
            
            # Show log even when no missing values
            with st.expander("Imputation Log"):
                if st.session_state.impute_log:
                    st.write("**Recent imputation operations:**")
                    for idx, (col, method) in enumerate(reversed(st.session_state.impute_log[-10:]), 1):
                        st.write(f"{idx}. Column: **{col}**, Method: **{method}**")
                else:
                    st.write("No imputations logged yet.")
        else:
            percent_missing = (total_missing / (df.shape[0] * df.shape[1])) * 100
            st.write(f"**Total Missing Cells:** `{total_missing}` ({percent_missing:.2f}%)")
            
            # Create summary dataframe with column types
            missing_summary_data = []
            for col in missing_cols.index:
                missing_summary_data.append({
                    'Column': col,
                    'Missing Count': missing_cols[col],
                    'Data Type': df[col].dtype.name
                })
            missing_summary_df = pd.DataFrame(missing_summary_data)
            st.dataframe(missing_summary_df, use_container_width=True)
    
            if st.checkbox("Drop rows with more than 50% missing values"):
                initial_rows = df.shape[0]
                df = df[df.isnull().mean(axis=1) < 0.5]
                st.session_state.processed_df = df
                st.success(f"Dropped {initial_rows - df.shape[0]} rows.")
                st.dataframe(df.head())
    
            missing_columns = df.columns[df.isnull().any()]
            selected_column = st.selectbox("Select a column to impute", missing_columns)
    
            col_dtype = df[selected_column].dtype
            is_numeric = pd.api.types.is_numeric_dtype(col_dtype)
            is_categorical = pd.api.types.is_object_dtype(col_dtype) or pd.api.types.is_categorical_dtype(col_dtype)
    
            st.markdown(f"**Detected Column Type:** `{col_dtype}`")
    
            # Mode selection
            if is_categorical:
                impute_mode = "Categorical"
                st.info("Categorical columns only support simple imputation methods.")
            elif is_numeric:
                impute_mode = st.radio("Select imputation complexity level:", ["Simple", "Advanced"], horizontal=True)
            else:
                st.error("Unsupported data type for imputation.")
                impute_mode = None
    
            # Apply to all toggle
            apply_all = st.checkbox("Apply to all columns of this type using this method")
    
            method_descriptions = {
                # Simple & categorical
                "Mean": """
                **What it does:** Replaces missing values with the average of all non-missing values in the column.
                
                **Best for:** 
                - Continuous numeric data (temperatures, prices, measurements)
                - Data that's roughly normally distributed (bell curve shaped)
                - When you have a moderate amount of missing data (<20%)
                
                **Avoid when:**
                - You have extreme outliers (they'll skew the average)
                - Data is skewed (use Median instead)
                - Missing data isn't random (e.g., all missing values are from one specific group)
                
                **Example:** If ages are [25, 30, ?, 28, 32], the missing value becomes 28.75 (the average).
                """,
                
                "Median": """
                **What it does:** Replaces missing values with the middle value when all values are sorted.
                
                **Best for:**
                - Numeric data with outliers (income, house prices, age)
                - Skewed distributions (most values cluster on one side)
                - When you want a "typical" value that isn't affected by extremes
                
                **Avoid when:**
                - You need the mathematical average for calculations
                - Data is categorical or text
                
                **Example:** If salaries are [$40k, $45k, ?, $50k, $200k], the missing value becomes $47.5k (not affected by the $200k outlier like Mean would be).
                """,
                
                "Mode": """
                **What it does:** Replaces missing values with the most frequently occurring value.
                
                **Best for:**
                - Categorical data (colors, categories, labels, yes/no)
                - Discrete numeric data (number of kids, rating scales)
                - When the most common value is meaningful
                
                **Avoid when:**
                - All values are unique (no mode exists)
                - You're dealing with continuous measurements
                
                **Example:** If colors are [Red, Blue, ?, Blue, Green, Blue], the missing value becomes Blue (most common).
                """,
                
                "Fill with 'NA' (string literal)": """
                **What it does:** Inserts the literal text "NA" into missing cells.
                
                **Best for:**
                - Text columns where you want to explicitly mark missing data
                - When "NA" or "Not Available" is meaningful in your context
                - Creating categorical labels for missing status
                
                **Avoid when:**
                - Working with numeric columns (use NaN instead)
                - You want to statistically impute values
                
                **Example:** If comments are ["Great", ?, "Good"], it becomes ["Great", "NA", "Good"].
                """,
                
                "Fill with custom value": """
                **What it does:** Lets you manually specify any value to replace missing cells.
                
                **Best for:**
                - When you know the correct default value (e.g., 0 for "number of complaints")
                - Categorical data with a specific "Unknown" or "Other" category
                - When domain knowledge suggests a specific value
                
                **Avoid when:**
                - You're guessing without domain knowledge
                - The custom value could bias your analysis
                
                **Example:** Fill missing "Country" values with "Unknown" or missing "Quantity" with 0.
                """,
                
                "Forward Fill (LOCF)": """
                **What it does:** Carries the last known value forward to fill gaps. "Last Observation Carried Forward."
                
                **Best for:**
                - Time-series data where values change slowly (stock prices, temperature readings)
                - Status fields that remain constant until changed
                - Sequential data where the previous value is a good estimate
                
                **Avoid when:**
                - Data is not ordered or sequential
                - Missing values are at the start (nothing to carry forward)
                - Values change rapidly or unpredictably
                
                **Example:** If temps are [20°C, 21°C, ?, ?, 25°C], it becomes [20°C, 21°C, 21°C, 21°C, 25°C].
                """,
                
                "Backward Fill (NOCB)": """
                **What it does:** Pulls the next valid value backward to fill gaps. "Next Observation Carried Backward."
                
                **Best for:**
                - Time-series data when future values are known
                - When forward fill would create too much lag
                - Combining with forward fill for better coverage
                
                **Avoid when:**
                - Data is not ordered or sequential
                - Missing values are at the end (nothing to carry backward)
                - Future values don't make logical sense for past data
                
                **Example:** If temps are [20°C, ?, ?, 25°C, 26°C], it becomes [20°C, 25°C, 25°C, 25°C, 26°C].
                """,
                
                # Advanced
                "KNN Imputer": """
                **What it does:** Finds the K most similar rows (neighbors) based on other columns, then uses their values to fill the missing value. Uses distance calculations to determine "similarity."
                
                **How it works:**
                1. Looks at all other numeric columns in your dataset
                2. Finds the rows most similar to the one with missing data
                3. Takes the average of those similar rows' values
                
                **Best for:**
                - When missing values relate to other columns (e.g., Age relates to Income and Education)
                - Datasets with multiple numeric columns that correlate
                - When you have enough complete rows to find similar patterns
                - Missing data is "Missing at Random" (MAR)
                
                **Requirements:**
                - At least 2 numeric columns (one to impute, one to use as a feature)
                - Enough non-missing data to find neighbors (recommended: >50 complete rows)
                - Other columns should be relevant to the missing column
                
                **Settings:**
                - K (neighbors): Default is 5. Lower K = more influenced by closest matches. Higher K = smoother, more averaged results.
                
                **Avoid when:**
                - You only have 1 numeric column
                - Other columns don't relate to the missing one
                - Dataset is very small (<20 rows)
                
                **Example:** Predicting missing "Salary" using similar people's Age, Years Experience, and Education Level.
                """,
                
                "Linear Regression": """
                **What it does:** Builds a mathematical formula (line of best fit) that predicts missing values based on other numeric columns. Like drawing a trend line through your data.
                
                **How it works:**
                1. Learns the relationship between the target column and other columns
                2. Creates a prediction formula (e.g., Salary = 30000 + 2000*YearsExperience)
                3. Uses this formula to predict missing values
                
                **Best for:**
                - When there's a clear linear relationship between columns
                - Continuous numeric data that follows trends
                - When you have at least 1 good predictor column
                - Data with moderate correlations
                
                **Requirements:**
                - At least 1 other numeric column to use as predictor
                - Enough complete rows to train (recommended: >30 rows)
                - Relationship between columns should be somewhat linear
                
                **Strengths:**
                - Fast and interpretable
                - Works well with linear trends
                - Doesn't require tons of data
                
                **Avoid when:**
                - Relationships are non-linear or complex
                - No clear correlation between columns
                - Very small datasets (<10 complete rows)
                
                **Example:** Predicting missing "House Price" based on Square Footage and Number of Bedrooms.
                """,
                
                "Iterative Imputer (MICE)": """
                **What it does:** "Multiple Imputation by Chained Equations" - repeatedly imputes values, learning from each round. Think of it as making educated guesses, then refining those guesses multiple times.
                
                **How it works:**
                1. Makes initial guesses for all missing values (using mean)
                2. For each column, builds a model using other columns
                3. Re-predicts missing values with the updated model
                4. Repeats this cycle multiple times until values stabilize
                
                **Best for:**
                - Complex datasets with missing values in multiple columns
                - When columns are interdependent (Age affects Income, Income affects Savings, etc.)
                - Larger datasets with moderate missingness (<40%)
                - When you want sophisticated, relationship-aware imputation
                
                **Requirements:**
                - At least 2 numeric columns
                - Sufficient complete data to learn patterns
                - Columns should have meaningful relationships
                
                **Settings:**
                - Max iterations: Default is 10. More iterations = more refined predictions but slower.
                
                **Strengths:**
                - Handles complex relationships between columns
                - Can impute multiple columns iteratively
                - Often more accurate than simpler methods
                
                **Avoid when:**
                - Dataset is very small (<50 rows)
                - You need fast results (this is slower)
                - Columns are completely independent
                
                **Example:** Predicting missing values across Age, Income, Education, and Years Experience where they all influence each other.
                """,
                
                "MissForest (Random Forest)": """
                **What it does:** Uses Random Forest machine learning to predict missing values. Builds multiple decision trees that "vote" on what the missing value should be.
                
                **How it works:**
                1. Creates many decision trees (a "forest")
                2. Each tree learns different patterns from your data
                3. Trees vote on predictions for missing values
                4. Takes the average/consensus of all tree predictions
                
                **Best for:**
                - Complex, non-linear relationships between columns
                - Mixed data types (numeric + categorical)
                - When simpler methods aren't accurate enough
                - Larger datasets with multiple predictor columns
                
                **Requirements:**
                - At least 2 numeric columns
                - Enough data for training (recommended: >100 rows)
                - Computational resources (slower than simple methods)
                
                **Strengths:**
                - Handles non-linear patterns extremely well
                - Robust to outliers
                - Can capture complex interactions
                - Often the most accurate method
                
                **Avoid when:**
                - Dataset is small (<50 rows)
                - You need fast results (this is the slowest)
                - Simple relationships (overkill - use KNN or Regression)
                - Limited computational resources
                
                **Example:** Predicting missing "Customer Churn" using complex patterns across Purchase History, Demographics, Website Behavior, and Support Tickets.
                """,
                
                "Interpolation": """
                **What it does:** "Connects the dots" between known values by drawing a smooth line. Estimates missing values based on their position between surrounding values.
                
                **How it works:**
                1. Looks at the values before and after the gap
                2. Draws a straight line between them
                3. Fills in missing values along that line
                
                **Best for:**
                - Time-series data with smooth trends (temperature over time, stock prices)
                - Sequential numeric data with gradual changes
                - When missing values are surrounded by known values (not at edges)
                - Data that changes predictably
                
                **Requirements:**
                - Sequential/ordered data
                - Missing values must have known values before AND after them
                - Numeric column only
                
                **Strengths:**
                - Simple and intuitive
                - Works great for smoothly changing data
                - Fast and lightweight
                
                **Avoid when:**
                - Data is not sequential or time-based
                - Missing values are at the start or end
                - Data has abrupt jumps or changes
                - Data is categorical
                
                **Example:** If temperatures are [20°C, 21°C, ?, ?, 25°C], interpolation fills with [22°C, 23°C] (evenly spaced between 21 and 25).
                """
            }
    
            if impute_mode == "Simple":
                methods = ["Mean", "Median", "Mode", "Fill with 'NA' (string literal)", "Fill with custom value", "Forward Fill (LOCF)", "Backward Fill (NOCB)"]
            elif impute_mode == "Advanced":
                methods = ["KNN Imputer", "Linear Regression", "Iterative Imputer (MICE)", "MissForest (Random Forest)", "Interpolation"]
            elif impute_mode == "Categorical":
                methods = ["Mode", "Fill with 'NA' (string literal)", "Fill with custom value"]
            else:
                methods = []
    
            selected_method = st.selectbox("Choose your imputation method", methods)
    
            with st.expander("What this method does"):
                st.markdown(method_descriptions.get(selected_method, "No description available."))
    
            # Additional parameters for advanced methods
            if selected_method == "KNN Imputer":
                n_neighbors = st.slider("Number of neighbors (k)", min_value=1, max_value=10, value=5, 
                                       help="Number of similar rows to use for imputation")
            
            if selected_method == "Fill with custom value":
                custom_value = st.text_input("Enter value to fill missing cells with:")
    
            if st.button("Apply Imputation", type="primary"):
                st.session_state.df_backup = df.copy()
                
                # Store rows with missing values for display
                if apply_all:
                    applicable_columns = [col for col in missing_columns if df[col].dtype == df[selected_column].dtype]
                    missing_mask = df[applicable_columns].isna().any(axis=1)
                else:
                    missing_mask = df[selected_column].isna()
                
                missing_indices = df[missing_mask].index.tolist()
    
                def apply_imputation(col):
                    """Apply the selected imputation method to a column"""
                    try:
                        # Simple methods
                        if selected_method == "Mean":
                            df[col] = df[col].fillna(df[col].mean())
                            return f"Applied Mean to '{col}'"
                            
                        elif selected_method == "Median":
                            df[col] = df[col].fillna(df[col].median())
                            return f"Applied Median to '{col}'"
                            
                        elif selected_method == "Mode":
                            mode_val = df[col].mode()
                            if len(mode_val) > 0:
                                df[col] = df[col].fillna(mode_val.iloc[0])
                                return f"Applied Mode to '{col}'"
                            else:
                                return f"No mode found for '{col}'"
                                
                        elif selected_method == "Fill with 'NA' (string literal)":
                            df[col] = df[col].fillna("NA")
                            return f"Filled '{col}' with 'NA'"
                            
                        elif selected_method == "Fill with custom value":
                            df[col] = df[col].fillna(custom_value)
                            return f"Filled '{col}' with custom value"
                            
                        elif selected_method == "Forward Fill (LOCF)":
                            df[col] = df[col].fillna(method="ffill")
                            return f"Applied Forward Fill to '{col}'"
                            
                        elif selected_method == "Backward Fill (NOCB)":
                            df[col] = df[col].fillna(method="bfill")
                            return f"Applied Backward Fill to '{col}'"
                        
                        # Advanced methods
                        elif selected_method == "KNN Imputer":
                            result_df, message = apply_knn_imputation_simple(df, col, n_neighbors=n_neighbors)
                            if "successful" in message:
                                df[col] = result_df[col]
                            return message
                            
                        elif selected_method == "Linear Regression":
                            result_df, message = apply_linear_regression_simple(df, col)
                            if "successful" in message:
                                df[col] = result_df[col]
                            return message
                            
                        elif selected_method == "Iterative Imputer (MICE)":
                            result_df, message = apply_mice_simple(df, col)
                            if "successful" in message:
                                df[col] = result_df[col]
                            return message
                            
                        elif selected_method == "MissForest (Random Forest)":
                            result_df, message = apply_missforest_simple(df, col)
                            if "successful" in message:
                                df[col] = result_df[col]
                            return message
                            
                        elif selected_method == "Interpolation":
                            result_df, message = apply_interpolation_simple(df, col)
                            if "successful" in message:
                                df[col] = result_df[col]
                            return message
                            
                        else:
                            return f"Unknown method: {selected_method}"
                            
                    except Exception as e:
                        return f"Failed to apply {selected_method} to '{col}': {str(e)}"
    
                # Apply imputation
                if apply_all:
                    applicable_columns = [col for col in missing_columns if df[col].dtype == df[selected_column].dtype]
                    results = []
                    for col in applicable_columns:
                        result = apply_imputation(col)
                        results.append(result)
                        if "successful" in result.lower() or result.startswith("Applied") or result.startswith("Filled"):
                            st.session_state.impute_log.append((col, selected_method))
                    
                    # Update session state FIRST
                    st.session_state.processed_df = df
                    
                    # Show results
                    for result in results:
                        if "successful" in result.lower() or result.startswith("Applied") or result.startswith("Filled"):
                            st.success(result)
                        elif "not enough" in result.lower() or "no mode" in result.lower() or "must be numeric" in result.lower():
                            st.warning(result)
                        else:
                            st.error(result)
                    
                    st.info(f"Applied {selected_method} to {len(applicable_columns)} column(s).")
                else:
                    result = apply_imputation(selected_column)
                    
                    # Update session state FIRST
                    st.session_state.processed_df = df
                    
                    if "successful" in result.lower() or result.startswith("Applied") or result.startswith("Filled"):
                        st.session_state.impute_log.append((selected_column, selected_method))
                        st.success(result)
                    elif "not enough" in result.lower() or "no mode" in result.lower() or "must be numeric" in result.lower():
                        st.warning(result)
                    else:
                        st.error(result)

                # Force the page to update
                st.rerun()
                
                # Show imputed rows with context
                st.subheader("Imputation Results")
                
                if len(missing_indices) > 0:
                    # Determine which columns to show
                    if apply_all:
                        cols_to_show = applicable_columns
                    else:
                        cols_to_show = [selected_column]
                    
                    # Show comparison
                    num_rows_to_show = min(10, len(missing_indices))
                    st.write(f"Showing {num_rows_to_show} of {len(missing_indices)} imputed row(s) with context:")
                    
                    for i, idx in enumerate(missing_indices[:num_rows_to_show]):
                        st.markdown(f"**Imputed Row {idx}** (showing ±2 rows for context):")
                        
                        # Get 2 rows above and 2 rows below for context
                        start_idx = max(0, df.index.get_loc(idx) - 2)
                        end_idx = min(len(df) - 1, df.index.get_loc(idx) + 2)
                        
                        # Get the context window
                        context_df = df.iloc[start_idx:end_idx + 1][cols_to_show].copy()
                        
                        # Add an indicator column to show which row was imputed
                        context_df.insert(0, 'Status', '')
                        context_df.loc[idx, 'Status'] = '← IMPUTED'
                        
                        st.dataframe(context_df, use_container_width=True)
                    
                    if len(missing_indices) > num_rows_to_show:
                        st.info(f"Showing first {num_rows_to_show} rows. {len(missing_indices) - num_rows_to_show} more rows were also imputed.")
                else:
                    st.info("No missing values were found to impute.")
    
            if st.button("Undo Last Imputation"):
                if st.session_state.df_backup is not None:
                    st.session_state.processed_df = st.session_state.df_backup.copy()
                    df = st.session_state.processed_df
                    if st.session_state.impute_log:
                        undone = st.session_state.impute_log.pop()
                        st.success(f"Undid imputation: {undone[1]} on '{undone[0]}'")
                    else:
                        st.info("No previous imputation to undo.")
                else:
                    st.warning("No backup available to undo.")
    
            with st.expander("Imputation Log"):
                if st.session_state.impute_log:
                    st.write("**Recent imputation operations:**")
                    for idx, (col, method) in enumerate(reversed(st.session_state.impute_log[-10:]), 1):
                        st.write(f"{idx}. Column: **{col}**, Method: **{method}**")
                else:
                    st.write("No imputations logged yet.")
    else:
        st.info("Please upload a CSV file to begin.")



#####################################################################################################################################            
# SECTION 10: Download Processed Data
elif st.session_state.selected_section == "Download Processed Data":
    df = st.session_state.processed_df
    if df is None:
        st.error("Please upload a file first!")
        st.stop()
    st.header("10. Download Processed Data")
    
    df = st.session_state.processed_df
    
    if df is not None:
        st.markdown(f"Your cleaned dataset has **{df.shape[0]} rows** and **{df.shape[1]} columns**.")
        
        if st.button("Show Final Data Preview"):
            st.subheader("Final Data Preview")
            
            # Use the existing Enhanced Information Table function
            info_df, columns_with_missing = generate_enhanced_information_table(df)
            
            # Calculate dynamic height
            dynamic_height = min(38 + (len(info_df) * 35) + 10, 600)
            st.dataframe(info_df, height=dynamic_height, use_container_width=True)
            
            # Show warning if there are still columns with missing values
            if columns_with_missing:
                st.warning(f"Note: {len(columns_with_missing)} column(s) still have missing values:")
                
                missing_details = []
                for col_name, col_dtype in columns_with_missing:
                    null_count = df[col_name].isnull().sum()
                    null_pct = (null_count / len(df)) * 100
                    missing_details.append({
                        'Column': col_name,
                        'Missing Count': null_count,
                        'Missing %': f"{null_pct:.2f}%"
                    })
                
                missing_df = pd.DataFrame(missing_details)
                st.dataframe(missing_df, use_container_width=True, hide_index=True)
            else:
                st.success("All columns are complete - no missing values!")
    
        file_format = st.radio("Choose download format", ["CSV", "Excel (.xlsx)"], horizontal=True)
        
        # Generate timestamp for filename
        from datetime import datetime
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
        if file_format == "CSV":
            csv_data = df.to_csv(index=False).encode("utf-8")
            st.download_button(
                label="Download CSV File",
                data=csv_data,
                file_name=f"cleaned_data_{timestamp}.csv",
                mime="text/csv"
            )
        else:
            from io import BytesIO
            output = BytesIO()
            with pd.ExcelWriter(output, engine='openpyxl') as writer:
                df.to_excel(writer, index=False, sheet_name='CleanedData')
            output.seek(0)
            st.download_button(
                label="Download Excel File",
                data=output.getvalue(),
                file_name=f"cleaned_data_{timestamp}.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            )
    else:
        st.info("Please upload and clean a dataset before downloading.")
