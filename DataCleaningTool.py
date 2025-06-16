import streamlit as st
import pandas as pd
import numpy as np

# Set the page configuration at the very start
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
    Determine the predominant data type of each column based on a threshold.

    Parameters:
        dataframe (pd.DataFrame): The dataset to classify.
        type_threshold (float): The proportion threshold for classifying a column as numeric or string.

    Returns:
        dict: {column_name: 'numeric'|'string'|'mixed'|'empty'}

    Example usage:
        column_types = determine_column_data_types(df, 0.95)
    """
    column_data_types = {}
    
    for column in dataframe.columns:
        string_count = 0
        numeric_count = 0
        
        for entry in dataframe[column]:
            if pd.isna(entry):
                continue
            if isinstance(entry, str):
                # Check if the string can be parsed as a number
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

# Layout: two columns
left_col, right_col = st.columns([1, 4], gap="medium")

with left_col:
    st.subheader("Enhanced Information Table")
    # Button to refresh Enhanced Info Table - pressing it causes a rerun
    if st.button("Refresh Enhanced Info Table"):
        pass  # No action needed; button press triggers a rerun automatically

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
            st.success("🎉 No missing values found in your dataset.")
        
        # Handle Mixed Data Types
        st.header("3. Handle Mixed Data Types")
        st.markdown("Convert columns to their predominant data type based on a threshold.")
        
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

        # Preview mixed-type columns BEFORE conversion
        type_info = determine_column_data_types(df, type_threshold)
        mixed_cols = {k: v for k, v in type_info.items() if v == 'mixed'}
        
        if mixed_cols:
            st.subheader("🧪 Detected Mixed-Type Columns")
            preview = []
        
            incorrect_analysis = identify_incorrect_entries(df)
        
            for col in mixed_cols:
                details = incorrect_analysis.get(col, {})
                preview.append({
                    "Column": col,
                    "String Count": details.get('string_count', 0),
                    "Numeric Count": details.get('numeric_count', 0),
                    "String %": f"{details.get('string_ratio', 0):.2f}%",
                    "Numeric %": f"{details.get('numeric_ratio', 0):.2f}%",
                })
        
            st.dataframe(pd.DataFrame(preview))
        else:
            st.success("🎉 No mixed-type columns detected.")

        if st.button("Convert Mixed Data Types"):
            cleaned_df, conversion_report = convert_mixed_data_types(df, type_threshold)
            st.session_state.processed_df = cleaned_df
            df = st.session_state.processed_df
            
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
        st.markdown("Identify and fix columns with incorrect entries (mixed numeric/string).")

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
                    st.write(f"- String entries: {details['string_count']} ({details['string_ratio']:.2f}%)")
                    st.write(f"- Numeric entries: {details['numeric_count']} ({details['numeric_ratio']:.2f}%)")
                    
                    c1, c2, c3 = st.columns(3)
                    
                    with c1:
                        if st.button(f"Replace strings with NaN in {column}", key=f"str_nan_{column}"):
                            df.loc[details['string_indices'], column] = np.nan
                            st.session_state.processed_df = df
                            st.session_state.processed_columns.add(column)
                            st.success(f"✅ Replaced string entries with NaN in '{column}'")
                    
                    with c2:
                        if st.button(f"Replace numbers with NaN in {column}", key=f"num_nan_{column}"):
                            df.loc[details['numeric_indices'], column] = np.nan
                            st.session_state.processed_df = df
                            st.session_state.processed_columns.add(column)
                            st.success(f"✅ Replaced numeric entries with NaN in '{column}'")
                    
                    with c3:
                        if st.button(f"Skip {column}", key=f"skip_{column}"):
                            st.session_state.processed_columns.add(column)
                            st.info(f"⏭️ Skipped handling incorrect entries in '{column}'")

        # Processing History
        if st.session_state.processed_columns:
            st.markdown("---")
            st.markdown("**Processing History:**")
            processed_cols_list = list(st.session_state.processed_columns)
            if processed_cols_list:
                st.write(f"✓ Processed columns: {', '.join(processed_cols_list)}")
            
            if st.button("Reset Processing History"):
                st.session_state.processed_columns = set()
                st.success("Processing history has been reset.")

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
            
        # Section 6: Impute Missing Values
        st.header("6. Impute Missing Values")
        
        if 'impute_log' not in st.session_state:
            st.session_state.impute_log = []
        if 'df_backup' not in st.session_state:
            st.session_state.df_backup = None
        
        df = st.session_state.processed_df
        
        if df is not None:
            missing_columns = df.columns[df.isnull().any()]
            if len(missing_columns) == 0:
                st.success("🎉 No missing values found in your dataset.")
            else:
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
                apply_all = st.checkbox("Apply to all columns with missing values using this method")
        
                methods = []
                method_descriptions = {}
        
                # Shared across modes
                method_descriptions.update({
                    "Mode": "Replaces missing values with the most frequently occurring value in the column.",
                    "Fill with NA": "Replaces missing values with a 'NA' placeholder.",
                    "Fill with custom value": "Allows user to enter a specific value to fill missing cells.",
                    "Forward Fill (LOCF)": "Fills missing values with the last valid observation above (good for time-ordered data).",
                    "Backward Fill (NOCB)": "Fills missing values with the next valid observation below.",
                })
        
                # Simple methods
                if impute_mode == "Simple":
                    methods = ["Mean", "Median"] + list(method_descriptions.keys())
                    method_descriptions.update({
                        "Mean": "Fills missing values with the average of the column.",
                        "Median": "Fills missing values with the median value of the column.",
                    })
        
                elif impute_mode == "Advanced":
                    methods = [
                        "KNN Imputer", "Linear Regression", "Iterative Imputer (MICE)",
                        "MissForest (Random Forest)", "Interpolation", "Expectation Maximization (EM)", "Bayesian Imputation"
                    ]
                    method_descriptions.update({
                        "KNN Imputer": "Uses similar rows to predict missing values based on proximity in feature space.",
                        "Linear Regression": "Trains a regression model using complete rows to predict missing values.",
                        "Iterative Imputer (MICE)": "Fills missing values iteratively, modeling each column as a function of the others.",
                        "MissForest (Random Forest)": "Non-linear imputation using random forests (via `missingpy`).",
                        "Interpolation": "Estimates missing values from trends in nearby values (linear/spline).",
                        "Expectation Maximization (EM)": "Statistical method that maximizes likelihood to estimate missing data.",
                        "Bayesian Imputation": "Samples probable values from a posterior distribution (via `pymc`)."
                    })
        
                elif impute_mode == "Categorical":
                    methods = list(method_descriptions.keys())
        
                selected_method = st.selectbox("Select an imputation method", methods)
        
                with st.expander("ℹ️ About this method"):
                    st.markdown(method_descriptions.get(selected_method, "No description available."))
        
                if selected_method == "Fill with custom value":
                    custom_value = st.text_input("Enter value to fill missing cells with:")
        
                if st.button("Apply Imputation"):
                    st.session_state.df_backup = df.copy()
        
                    def apply_imputation(col):
        
                        if selected_method == "Mean":
                            df[col] = df[col].fillna(df[col].mean())
                        elif selected_method == "Median":
                            df[col] = df[col].fillna(df[col].median())
                        elif selected_method == "Mode":
                            df[col] = df[col].fillna(df[col].mode().iloc[0])
                        elif selected_method == "Fill with NA":
                            df[col] = df[col].fillna("NA")
                        elif selected_method == "Fill with custom value":
                            df[col] = df[col].fillna(custom_value)
                        elif selected_method == "Forward Fill (LOCF)":
                            df[col] = df[col].fillna(method="ffill")
                        elif selected_method == "Backward Fill (NOCB)":
                            df[col] = df[col].fillna(method="bfill")
                        elif selected_method == "KNN Imputer":
                            from sklearn.impute import KNNImputer
                            imputer = KNNImputer(n_neighbors=3)
                            df[df.columns] = imputer.fit_transform(df)
                        elif selected_method == "Linear Regression":
                            from sklearn.linear_model import LinearRegression
                            complete = df[df[col].notnull()]
                            missing = df[df[col].isnull()]
                            X_train = complete.drop(columns=[col]).select_dtypes(include=[np.number])
                            y_train = complete[col]
                            X_missing = missing.drop(columns=[col]).select_dtypes(include=[np.number])
                            if len(X_train) > 0 and len(X_missing) > 0:
                                model = LinearRegression().fit(X_train, y_train)
                                df.loc[df[col].isnull(), col] = model.predict(X_missing)
                        elif selected_method == "Iterative Imputer (MICE)":
                            from sklearn.experimental import enable_iterative_imputer
                            from sklearn.impute import IterativeImputer
                            imp = IterativeImputer(random_state=0)
                            df[df.columns] = imp.fit_transform(df)
                        elif selected_method == "Interpolation":
                            df[col] = df[col].interpolate(method='linear')
                        elif selected_method == "MissForest (Random Forest)":
                            from missingpy import MissForest
                            mf = MissForest()
                            df[df.columns] = mf.fit_transform(df)
                        elif selected_method == "Expectation Maximization (EM)":
                            st.warning("⚠️ EM is not implemented yet. Please install `fancyimpute` or skip.")
                        elif selected_method == "Bayesian Imputation":
                            st.warning("⚠️ Bayesian imputation is complex and requires PyMC. Not implemented in this version.")
                        else:
                            st.error("❌ Unsupported imputation method.")
        
                    if apply_all:
                        applicable_columns = [col for col in missing_columns if df[col].dtype == df[selected_column].dtype]
                        for col in applicable_columns:
                            apply_imputation(col)
                            st.session_state.impute_log.append((col, selected_method))
                        st.success(f"✅ Applied {selected_method} to all {len(applicable_columns)} applicable columns.")
                    else:
                        apply_imputation(selected_column)
                        st.session_state.impute_log.append((selected_column, selected_method))
                        st.success(f"✅ Applied {selected_method} to '{selected_column}'.")
        
                    st.session_state.processed_df = df
                    st.subheader("Updated Data Preview")
                    st.dataframe(df.head())
        
                if st.button("Undo Last Imputation"):
                    if st.session_state.df_backup is not None:
                        st.session_state.processed_df = st.session_state.df_backup.copy()
                        df = st.session_state.processed_df
                        if st.session_state.impute_log:
                            undone = st.session_state.impute_log.pop()
                            st.success(f"🔄 Undid imputation: {undone[1]} on '{undone[0]}'")
                        else:
                            st.info("ℹ️ No previous imputation to undo.")
                    else:
                        st.warning("⚠️ No backup available to undo.")
        
                # Optional: Show Imputation Log
                with st.expander("🧾 Imputation Log"):
                    if st.session_state.impute_log:
                        for idx, (col, method) in enumerate(reversed(st.session_state.impute_log[-10:]), 1):
                            st.write(f"{idx}. Column: **{col}**, Method: **{method}**")
                    else:
                        st.write("No imputations logged yet.")
        else:
            st.info("Please upload a CSV file to begin.")

            
        # Download Processed Data
        st.header("7. Download Processed Data")
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
