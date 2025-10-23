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
# SECTION 1: File Upload Section
if st.session_state.selected_section == "File Upload":
    st.header("1. Upload Data")
    uploaded_file = st.file_uploader("Choose a CSV file", type="csv")

    if uploaded_file is not None:
        # Load data if not already loaded
        if st.session_state.processed_df is None:
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
        st.subheader("Missing Values Summary")
        missing_summary = df.isnull().sum()
        missing_summary = missing_summary[missing_summary > 0].sort_values(ascending=False)
        
        if not missing_summary.empty:
            missing_percent = (missing_summary / df.shape[0]) * 100
            summary_df = pd.DataFrame({
                'Missing Count': missing_summary,
                'Missing %': missing_percent.map(lambda x: f"{x:.2f}%"),
                'Data Type': [df[col].dtype.name for col in missing_summary.index]
            })
            st.write(f"Total Missing Value Proportion: {(df.isnull().sum().sum() / (df.shape[0] * df.shape[1])) * 100:.2f}%")
            st.dataframe(summary_df)
        else:
            st.success("No missing values found in your dataset.")
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
# SECTION 4: Smart Object Column Conversion
elif st.session_state.selected_section == "Object Conversion":
    df = st.session_state.processed_df
    if df is None:
        st.error("Please upload a file first!")
        st.stop()
    st.header("4. Smart Object Column Conversion")
    st.markdown("Automatically detect and convert object columns to their appropriate data types.")
    if st.button("🔍 Analyze Object Columns"):
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
                    
                    st.info(f" **Recommendation:** Convert column '{col}' to **{suggested}** ({confidence_text}: {confidence:.1f}%)")
                
                with col2:
                    conversion_choice = st.selectbox(
                        "Convert to:",
                        options=['string', 'numeric', 'datetime'],
                        index=['string', 'numeric', 'datetime'].index(suggested),
                        key=f"convert_{col}",
                        help=f"Suggested: {suggested} (confidence: {info['confidence']:.1f}%)"
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
                    st.metric("Confidence", f"{info['confidence']:.1f}%")
                    if info['confidence'] > 85:
                        st.success("High confidence")
                    elif info['confidence'] > 60:
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
            st.success("🎉 No object columns found to convert.")
    
    # 4.3 Float → Int (keep this section as-is since it's working well)
    st.subheader("4.3 Convert float columns to integer (if safe)")
    enable_floatint = st.checkbox("Enable float-to-int optimization", key="enable_floatint")
    
    if enable_floatint:
        safe_int_cols = []
        for col in df.select_dtypes(include=['float64']):
            if df[col].dropna().apply(float.is_integer).all():
                safe_int_cols.append(col)
    
        if safe_int_cols:
            st.dataframe(df[safe_int_cols].head(), use_container_width=True)
            if st.button("Convert to Int", key="btn_floatint"):
                for col in safe_int_cols:
                    df[col] = df[col].astype('Int64')
                    st.success(f"{col}: float64 → Int64")
                st.session_state.processed_df = df
        else:
            st.info("No float columns are safely convertible to integers.")
#####################################################################################################################################
# SECTION 5: Optimize for Analysis
elif st.session_state.selected_section == "Optimize Analysis":
    df = st.session_state.processed_df
    if df is None:
        st.error("Please upload a file first!")
        st.stop()
    st.header("5. Optimize for Analysis")
    st.markdown("""
    Final optimization for memory and performance:
    - Convert low-cardinality string columns to `category`.
    - Optionally downcast numeric types (e.g., int64 → int32).
    """)
    
    optimize_cats = st.checkbox("Convert low-cardinality string columns to 'category'", value=True)
    downcast_nums = st.checkbox("Downcast numeric types to smaller types", value=False)
    
    if st.button("Run Optimization"):
        if optimize_cats:
            for col in df.select_dtypes(include='object'):
                if should_treat_as_categorical(df[col]):
                    df[col] = df[col].astype('category')
                    st.write(f"{col}: object → category")
    
        if downcast_nums:
            for col in df.select_dtypes(include=['int64', 'float64']):
                df[col] = pd.to_numeric(df[col], downcast='unsigned' if df[col].min() >= 0 else 'integer')
                st.write(f"{col}: downcasted for memory optimization")
    
        st.session_state.processed_df = df
        st.success("Optimization complete!")

#####################################################################################################################################
# SECTION 6: Detect and Handle Outliers
elif st.session_state.selected_section == "Handle Outliers":
    df = st.session_state.processed_df
    if df is None:
        st.error("Please upload a file first!")
        st.stop()
    st.header("6. Detect and Handle Outliers")
    with st.expander("ℹ️ What are outliers and how does this work?"):
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
    
        if outlier_report:
            st.success(f"Found {len(outlier_report)} columns with outliers.")
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
                        df.loc[stats["mask"], col] = np.nan
                        st.session_state.processed_df = df
                        st.success(f"Replaced outliers in '{col}' with NaN")
        
                with c2:
                    if st.button(f"Drop rows with outliers in '{col}'", key=f"drop_{col}"):
                        df = df[~stats["mask"]]
                        st.session_state.processed_df = df
                        st.success(f"Dropped {stats['count']} rows with outliers in '{col}'")
        
                with c3:
                    if st.button(f"Skip column '{col}'", key=f"skip_{col}"):
                        st.info(f"Skipped handling outliers in '{col}'")
        else:
            st.success("No outliers detected using the current IQR threshold.")

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
        detect_button = st.button("🔍 Run Text Detection", type="secondary")
    with col2:
        if st.session_state.text_analysis_done:
            st.write(f"Analysis complete - {len(st.session_state.text_columns)} text column(s) found")
        else:
            st.write("Click to analyze text columns in your current dataset")
    
    if detect_button or not st.session_state.text_analysis_done:
        # Get text columns from the current dataframe
        df = st.session_state.processed_df
        text_columns = df.select_dtypes(include=['object', 'string']).columns.tolist()
        
        if not text_columns:
            st.info("No text columns found in your current dataset.")
            st.session_state.text_columns = []
            st.session_state.text_overview = []
            st.session_state.text_analysis_done = True
        else:
            # Store results in session state
            st.session_state.text_columns = text_columns
            st.session_state.text_analysis_done = True
            
            # Create overview of text columns
            text_overview = []
            for col in text_columns:
                non_null_count = df[col].notna().sum()
                unique_count = df[col].nunique()
                
                # Check for null-like strings
                null_like_patterns = ["NA", "N/A", "na", "n/a", "--", "-", "", "null", "NULL", "Null"]
                null_like_count = df[col].isin(null_like_patterns).sum()
                
                # Check for whitespace issues
                whitespace_issues = 0
                if non_null_count > 0:
                    whitespace_issues = df[col].astype(str).apply(lambda x: x != x.strip()).sum()
                
                # Sample values (first 3 non-null unique values)
                sample_values = df[col].dropna().unique()[:3].tolist()
                sample_str = ", ".join([f'"{str(val)}"' for val in sample_values])
                if len(sample_values) == 3 and unique_count > 3:
                    sample_str += "..."
                
                text_overview.append({
                    'Column': col,
                    'Non-null Values': non_null_count,
                    'Unique Values': unique_count,
                    'Null-like Strings': null_like_count,
                    'Whitespace Issues': whitespace_issues,
                    'Sample Values': sample_str
                })
            
            st.session_state.text_overview = text_overview
            st.success(f"🔍 Text detection complete! Found {len(text_columns)} text column(s)")
    
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
        st.info("No text columns found in your current dataset. Try running text detection again if you've made changes to your data.")
    else:
        st.info("Click 'Run Text Detection' to analyze your dataset and find text columns to clean.")
        
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
# SECTION 9: Impute Missing Values
elif st.session_state.selected_section == "Impute Missing Values":
    df = st.session_state.processed_df
    if df is None:
        st.error("Please upload a file first!")
        st.stop()
    st.header("9. Impute Missing Values")
    
    # ========== IMPUTATION HELPER FUNCTIONS ==========
    
    def prepare_data_for_advanced_imputation(df, target_column):
        """
        Prepare dataframe for advanced imputation methods.
        
        Returns:
            prepared_df: DataFrame ready for sklearn methods (all numeric)
            metadata: Dictionary containing info needed to restore original state
        """
        metadata = {
            'original_dtypes': df.dtypes.to_dict(),
            'target_column': target_column,
            'categorical_mappings': {},
            'original_column_order': df.columns.tolist(),
            'target_missing_mask': df[target_column].isna()
        }
        
        # Create a copy to work with
        prepared_df = df.copy()
        
        # Encode categorical columns temporarily
        for col in prepared_df.columns:
            if prepared_df[col].dtype == 'object' or prepared_df[col].dtype.name == 'category':
                # Store the mapping
                unique_vals = prepared_df[col].dropna().unique()
                mapping = {val: idx for idx, val in enumerate(unique_vals)}
                reverse_mapping = {idx: val for val, idx in mapping.items()}
                
                metadata['categorical_mappings'][col] = {
                    'forward': mapping,
                    'reverse': reverse_mapping
                }
                
                # Apply encoding
                prepared_df[col] = prepared_df[col].map(mapping)
        
        # Convert any remaining non-numeric types to float
        for col in prepared_df.columns:
            if not pd.api.types.is_numeric_dtype(prepared_df[col]):
                try:
                    prepared_df[col] = pd.to_numeric(prepared_df[col], errors='coerce')
                except:
                    prepared_df[col] = np.nan
        
        # Ensure all numeric types are float64 for sklearn
        prepared_df = prepared_df.astype('float64')
        
        return prepared_df, metadata
    
    def restore_data_after_imputation(imputed_df, original_df, metadata):
        """
        Restore dataframe to original dtypes and decode categorical variables.
        
        Returns:
            restored_df: DataFrame with original dtypes and only target column imputed
        """
        restored_df = original_df.copy()
        target_col = metadata['target_column']
        
        # Only update the target column with imputed values
        target_missing_mask = metadata['target_missing_mask']
        
        # Get the imputed values
        imputed_values = imputed_df[target_col].copy()
        
        # If target column was categorical, decode it
        if target_col in metadata['categorical_mappings']:
            reverse_map = metadata['categorical_mappings'][target_col]['reverse']
            # Round to nearest integer for categorical encoding
            imputed_values = imputed_values.round().astype('Int64')
            # Map back to original values
            imputed_values = imputed_values.map(reverse_map)
        
        # Restore original dtype for target column
        original_dtype = metadata['original_dtypes'][target_col]
        
        # Fill in the missing values in the target column
        if pd.api.types.is_integer_dtype(original_dtype):
            # For integer types, try to maintain that
            try:
                # Use nullable Int64 to preserve NaN if any remain
                imputed_values = imputed_values.astype('float64').round()
                restored_df.loc[target_missing_mask, target_col] = imputed_values[target_missing_mask]
                # Try to convert back to original integer type
                if not restored_df[target_col].isna().any():
                    restored_df[target_col] = restored_df[target_col].astype(original_dtype)
                else:
                    # Use nullable integer type
                    restored_df[target_col] = restored_df[target_col].astype('Int64')
            except:
                restored_df.loc[target_missing_mask, target_col] = imputed_values[target_missing_mask]
        else:
            # For other types, just fill in the values
            restored_df.loc[target_missing_mask, target_col] = imputed_values[target_missing_mask]
            try:
                restored_df[target_col] = restored_df[target_col].astype(original_dtype)
            except:
                pass  # Keep as-is if conversion fails
        
        return restored_df
    
    def get_numeric_feature_columns(df, target_column):
        """
        Get all numeric columns except the target column for use as features.
        
        Returns:
            list: Column names suitable for use as features
        """
        # Get all numeric columns
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        
        # Remove target column
        feature_cols = [col for col in numeric_cols if col != target_column]
        
        return feature_cols
    
    def apply_knn_imputation(df, target_column, n_neighbors=5):
        """
        Apply KNN imputation to target column using all available numeric features.
        
        Returns:
            imputed_df: DataFrame with target column imputed
            message: Success/info message
        """
        try:
            # Prepare data
            prepared_df, metadata = prepare_data_for_advanced_imputation(df, target_column)
            
            # Check if we have enough features
            feature_cols = get_numeric_feature_columns(prepared_df, target_column)
            
            if len(feature_cols) < 1:
                return df, "Not enough numeric columns for KNN imputation. Need at least 1 other numeric column."
            
            # Select only columns with data for imputation
            cols_to_use = [target_column] + feature_cols
            imputation_df = prepared_df[cols_to_use].copy()
            
            # Apply KNN imputation
            from sklearn.impute import KNNImputer
            imputer = KNNImputer(n_neighbors=min(n_neighbors, len(imputation_df.dropna())))
            imputed_values = imputer.fit_transform(imputation_df)
            
            # Put imputed values back into prepared_df
            prepared_df[cols_to_use] = imputed_values
            
            # Restore to original format
            restored_df = restore_data_after_imputation(prepared_df, df, metadata)
            
            filled_count = metadata['target_missing_mask'].sum()
            message = f"KNN imputation successful! Filled {filled_count} missing values using {len(feature_cols)} feature column(s)."
            
            return restored_df, message
            
        except Exception as e:
            return df, f"KNN imputation failed: {str(e)}"
    
    def apply_linear_regression_imputation(df, target_column):
        """
        Apply Linear Regression imputation to target column.
        
        Returns:
            imputed_df: DataFrame with target column imputed
            message: Success/info message
        """
        try:
            # Prepare data
            prepared_df, metadata = prepare_data_for_advanced_imputation(df, target_column)
            
            # Get feature columns
            feature_cols = get_numeric_feature_columns(prepared_df, target_column)
            
            if len(feature_cols) < 1:
                return df, "Not enough numeric columns for Linear Regression. Need at least 1 other numeric column."
            
            # Split into complete and missing
            target_missing = metadata['target_missing_mask']
            complete_rows = prepared_df[~target_missing].copy()
            missing_rows = prepared_df[target_missing].copy()
            
            # Drop rows with NaN in features for training
            complete_rows_clean = complete_rows.dropna(subset=feature_cols)
            
            if len(complete_rows_clean) < 2:
                return df, "Not enough complete rows for Linear Regression training."
            
            # Prepare training data
            X_train = complete_rows_clean[feature_cols]
            y_train = complete_rows_clean[target_column]
            
            # Prepare prediction data (drop rows with NaN in features)
            missing_rows_clean = missing_rows.dropna(subset=feature_cols)
            
            if len(missing_rows_clean) == 0:
                return df, "All missing rows have NaN in feature columns. Cannot predict."
            
            X_missing = missing_rows_clean[feature_cols]
            
            # Train and predict
            from sklearn.linear_model import LinearRegression
            model = LinearRegression()
            model.fit(X_train, y_train)
            predictions = model.predict(X_missing)
            
            # Fill predictions back into prepared_df
            prepared_df.loc[missing_rows_clean.index, target_column] = predictions
            
            # Restore to original format
            restored_df = restore_data_after_imputation(prepared_df, df, metadata)
            
            filled_count = len(predictions)
            message = f"Linear Regression successful! Filled {filled_count} missing values using {len(feature_cols)} feature column(s)."
            
            return restored_df, message
            
        except Exception as e:
            return df, f"Linear Regression imputation failed: {str(e)}"
    
    def apply_iterative_imputation(df, target_column, max_iter=10):
        """
        Apply MICE (Iterative Imputer) to target column.
        
        Returns:
            imputed_df: DataFrame with target column imputed
            message: Success/info message
        """
        try:
            if IterativeImputer is None:
                return df, "IterativeImputer not available. Update sklearn to version 0.21+."
            
            # Prepare data
            prepared_df, metadata = prepare_data_for_advanced_imputation(df, target_column)
            
            # Get feature columns
            feature_cols = get_numeric_feature_columns(prepared_df, target_column)
            
            if len(feature_cols) < 1:
                return df, "Not enough numeric columns for MICE. Need at least 1 other numeric column."
            
            # Select columns for imputation
            cols_to_use = [target_column] + feature_cols
            imputation_df = prepared_df[cols_to_use].copy()
            
            # Apply Iterative Imputation
            imputer = IterativeImputer(max_iter=max_iter, random_state=0)
            imputed_values = imputer.fit_transform(imputation_df)
            
            # Put imputed values back
            prepared_df[cols_to_use] = imputed_values
            
            # Restore to original format
            restored_df = restore_data_after_imputation(prepared_df, df, metadata)
            
            filled_count = metadata['target_missing_mask'].sum()
            message = f"MICE imputation successful! Filled {filled_count} missing values using {len(feature_cols)} feature column(s)."
            
            return restored_df, message
            
        except Exception as e:
            return df, f"MICE imputation failed: {str(e)}"
    
    def apply_missforest_imputation(df, target_column):
        """
        Apply MissForest (Random Forest) imputation to target column.
        
        Returns:
            imputed_df: DataFrame with target column imputed
            message: Success/info message
        """
        try:
            if MissForest is None:
                return df, "MissForest not available. Install: pip install missingpy"
            
            # Prepare data
            prepared_df, metadata = prepare_data_for_advanced_imputation(df, target_column)
            
            # Get feature columns
            feature_cols = get_numeric_feature_columns(prepared_df, target_column)
            
            if len(feature_cols) < 1:
                return df, "Not enough numeric columns for MissForest. Need at least 1 other numeric column."
            
            # Select columns for imputation
            cols_to_use = [target_column] + feature_cols
            imputation_df = prepared_df[cols_to_use].copy()
            
            # Apply MissForest
            imputer = MissForest(random_state=0)
            imputed_values = imputer.fit_transform(imputation_df)
            
            # Put imputed values back
            prepared_df[cols_to_use] = imputed_values
            
            # Restore to original format
            restored_df = restore_data_after_imputation(prepared_df, df, metadata)
            
            filled_count = metadata['target_missing_mask'].sum()
            message = f"MissForest imputation successful! Filled {filled_count} missing values using {len(feature_cols)} feature column(s)."
            
            return restored_df, message
            
        except Exception as e:
            return df, f"MissForest imputation failed: {str(e)}"
    
    def apply_interpolation(df, target_column, method='linear'):
        """
        Apply interpolation to target column (works on single column only).
        
        Returns:
            imputed_df: DataFrame with target column imputed
            message: Success/info message
        """
        try:
            restored_df = df.copy()
            original_dtype = df[target_column].dtype
            missing_count = df[target_column].isna().sum()
            
            # Apply interpolation
            restored_df[target_column] = df[target_column].interpolate(method=method)
            
            # Restore dtype if needed
            if pd.api.types.is_integer_dtype(original_dtype):
                try:
                    restored_df[target_column] = restored_df[target_column].round()
                    if not restored_df[target_column].isna().any():
                        restored_df[target_column] = restored_df[target_column].astype(original_dtype)
                    else:
                        restored_df[target_column] = restored_df[target_column].astype('Int64')
                except:
                    pass
            
            filled_count = missing_count - restored_df[target_column].isna().sum()
            message = f"Interpolation successful! Filled {filled_count} missing values."
            
            return restored_df, message
            
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
        if total_missing == 0:
            st.success("No missing values in your dataset!")
        else:
            percent_missing = (total_missing / (df.shape[0] * df.shape[1])) * 100
            st.write(f"**Total Missing Cells:** `{total_missing}` ({percent_missing:.2f}%)")
            st.dataframe(missing_cols.to_frame("Missing Count"))
    
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
            elif is_numeric:
                impute_mode = st.radio("Select imputation complexity level:", ["Simple", "Advanced"], horizontal=True)
            else:
                st.error("Unsupported data type for imputation.")
                impute_mode = None
    
            # Apply to all toggle
            apply_all = st.checkbox("Apply to all columns of this type using this method")
    
            method_descriptions = {
                # Simple & categorical
                "Mean": "Replaces missing values with the average of the column. Best for continuous data without extreme outliers.",
                "Median": "Replaces missing values with the median (middle) value. Robust to outliers.",
                "Mode": "Replaces missing values with the most frequently occurring value.",
                "Fill with 'NA' (string literal)": "Inserts the string 'NA' into missing cells. Best for text columns where 'NA' is meaningful.",
                "Fill with custom value": "Lets you manually enter any value to replace missing cells.",
                "Forward Fill (LOCF)": "Carries the last known value forward. Good for time-series or ordered data.",
                "Backward Fill (NOCB)": "Pulls the next valid value backward. Also useful in ordered datasets.",
                # Advanced
                "KNN Imputer": "Finds the 'k' most similar rows and uses their values to fill in the blanks. Great when patterns exist across columns. Uses all available numeric columns as features.",
                "Linear Regression": "Uses other numeric columns to predict the missing value using a regression model. Automatically selects all numeric columns as predictors.",
                "Iterative Imputer (MICE)": "Models each column as a function of the others and iteratively predicts missing values. Uses all numeric columns for modeling.",
                "MissForest (Random Forest)": "Uses random forests to fill missing values non-linearly. Very powerful, slower to run. Uses all numeric columns as features.",
                "Interpolation": "Connects the dots in numeric data (linear/spline interpolation). Good for time-continuous data. Works on single column only.",
                "Expectation Maximization (EM)": "A statistical technique that guesses likely values. Not implemented.",
                "Bayesian Imputation": "Samples values from a posterior probability distribution. Not implemented here."
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
                
                # Store rows with missing values for before/after comparison
                if apply_all:
                    applicable_columns = [col for col in missing_columns if df[col].dtype == df[selected_column].dtype]
                    missing_mask = df[applicable_columns].isna().any(axis=1)
                else:
                    missing_mask = df[selected_column].isna()
                
                rows_with_missing = df[missing_mask].copy()
                missing_indices = df[missing_mask].index.tolist()
    
                def apply_imputation(col):
                    """Apply the selected imputation method to a column"""
                    try:
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
                            
                        elif selected_method == "KNN Imputer":
                            result_df, message = apply_knn_imputation(df, col, n_neighbors=n_neighbors)
                            if "successful" in message:
                                # Update the df with the imputed column
                                df[col] = result_df[col]
                            return message
                            
                        elif selected_method == "Linear Regression":
                            result_df, message = apply_linear_regression_imputation(df, col)
                            if "successful" in message:
                                df[col] = result_df[col]
                            return message
                            
                        elif selected_method == "Iterative Imputer (MICE)":
                            result_df, message = apply_iterative_imputation(df, col)
                            if "successful" in message:
                                df[col] = result_df[col]
                            return message
                            
                        elif selected_method == "MissForest (Random Forest)":
                            result_df, message = apply_missforest_imputation(df, col)
                            if "successful" in message:
                                df[col] = result_df[col]
                            return message
                            
                        elif selected_method == "Interpolation":
                            result_df, message = apply_interpolation(df, col)
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
                        if "successful" in result.lower() or result.startswith("Applied"):
                            st.session_state.impute_log.append((col, selected_method))
                    
                    # Show results
                    for result in results:
                        if "successful" in result.lower() or result.startswith("Applied") or result.startswith("Filled"):
                            st.success(result)
                        elif "not enough" in result.lower() or "no mode" in result.lower():
                            st.warning(result)
                        else:
                            st.error(result)
                    
                    st.info(f"Applied {selected_method} to {len(applicable_columns)} column(s).")
                else:
                    result = apply_imputation(selected_column)
                    if "successful" in result.lower() or result.startswith("Applied") or result.startswith("Filled"):
                        st.session_state.impute_log.append((selected_column, selected_method))
                        st.success(result)
                    elif "not enough" in result.lower() or "no mode" in result.lower():
                        st.warning(result)
                    else:
                        st.error(result)
    
                # Update session state
                st.session_state.processed_df = df
                
                # Show imputed rows with context
                st.subheader("Imputation Results")
                
                if len(missing_indices) > 0:
                    # Get the after state for the same rows
                    rows_after_imputation = df.loc[missing_indices].copy()
                    
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
                        
                        # Create a styled dataframe to highlight the imputed row
                        def highlight_imputed_row(row):
                            if row.name == idx:
                                return ['background-color: #90EE90'] * len(row)  # Light green
                            else:
                                return [''] * len(row)
                        
                        styled_df = context_df.style.apply(highlight_imputed_row, axis=1)
                        st.dataframe(styled_df, use_container_width=True)
                    
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
            st.dataframe(df.head())
    
        file_format = st.radio("Choose download format", ["CSV", "Excel (.xlsx)"], horizontal=True)
    
        if file_format == "CSV":
            csv_data = df.to_csv(index=False).encode("utf-8")
            st.download_button(
                label="Download CSV File",
                data=csv_data,
                file_name="cleaned_data.csv",
                mime="text/csv"
            )
        else:
            from io import BytesIO
            output = BytesIO()
            with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
                df.to_excel(writer, index=False, sheet_name='CleanedData')
                writer.save()
            st.download_button(
                label="Download Excel File",
                data=output.getvalue(),
                file_name="cleaned_data.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            )
    else:
        st.info("Please upload and clean a dataset before downloading.")
