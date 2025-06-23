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

#####################################################################################################################################
### Functions ###
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

#####################################################################################################################################
### Code ###
#####################################################################################################################################

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
#####################################################################################################################################
    # SECTION 1: File Upload Section
    st.header("1. Upload Data")
    uploaded_file = st.file_uploader("Choose a CSV file", type="csv")

    if uploaded_file is not None:
        # Load data if not already loaded
        if st.session_state.processed_df is None:
            st.session_state.processed_df = pd.read_csv(uploaded_file)
        df = st.session_state.processed_df
        st.success("File successfully uploaded!")
        
#####################################################################################################################################
        # SECTION 2: Data Overview
        st.header("2. Data Overview")
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
            st.success("🎉 No missing values found in your dataset.")
        
#####################################################################################################################################
        # SECTION 3.1: Detect Mixed-Type Columns
        st.header("3.1. Detect Mixed-Type Columns")
        st.markdown("""
        This step identifies columns that contain both numeric and string values (e.g. `"12"`, `"hello"`, `3.14`).
        """)
        
        type_threshold = st.slider(
            "Threshold for Dominant Type",
            min_value=0.5, max_value=1.0, value=0.95, step=0.01,
            help="If 95% of a column is numeric, we classify it as numeric."
        )
        
        if st.button("🔍 Detect Mixed-Type Columns"):
            st.session_state.type_info = determine_column_data_types(df, type_threshold)
            st.session_state.mixed_type_cols = {
                k: v for k, v in st.session_state.type_info.items() if v['inferred_type'] == 'mixed'
            }
        
        if 'mixed_type_cols' in st.session_state:
            mixed_type_cols = st.session_state.mixed_type_cols
            if mixed_type_cols:
                st.subheader("🧪 Detected Mixed-Type Columns")
                for col, stats in mixed_type_cols.items():
                    st.markdown(f"**{col}** — `{stats['original_dtype']}`")
                    st.write(f"🔢 **String entries:** {stats['string_ratio']:.2f}% | 🔣 **Numeric entries:** {stats['numeric_ratio']:.2f}%")
            else:
                st.success("🎉 No mixed-type columns detected.")

        # SECTION 3.2: Fix Mixed-Type Columns
        st.header("3.2. Fix Mixed-Type Columns")
        if 'mixed_type_cols' in st.session_state and st.session_state.mixed_type_cols:
            st.markdown("""
        Choose how to fix each mixed-type column:
        - **Convert to Dominant Type**: Makes the entire column numeric or string.
        - **Replace Rogue Entries**: Replaces inconsistent values with `NaN` without changing the column type.
            """)
        
            if 'fixed_mixed_cols' not in st.session_state:
                st.session_state.fixed_mixed_cols = set()
        
            for col, stats in st.session_state.mixed_type_cols.items():
                if col in st.session_state.fixed_mixed_cols:
                    continue
        
                st.markdown(f"**{col}** — `{stats['original_dtype']}`")
                col1, col2, col3 = st.columns(3)
        
                with col1:
                    if st.button(f"Convert '{col}'", key=f"convert_{col}"):
                        if stats['numeric_ratio'] > stats['string_ratio']:
                            df[col] = pd.to_numeric(df[col], errors='coerce')
                        else:
                            df[col] = df[col].astype(str)
                        st.session_state.fixed_mixed_cols.add(col)
                        st.success(f"✅ Converted '{col}' to dominant type.")
        
                with col2:
                    if st.button(f"NaN Rogue '{col}'", key=f"nan_{col}"):
                        rogue_indices = []
                        for idx, val in enumerate(df[col]):
                            if pd.isna(val):
                                continue
                            try:
                                float(val)
                                if stats['string_ratio'] > stats['numeric_ratio']:
                                    rogue_indices.append(idx)
                            except:
                                if stats['numeric_ratio'] > stats['string_ratio']:
                                    rogue_indices.append(idx)
                        df.loc[rogue_indices, col] = np.nan
                        st.session_state.fixed_mixed_cols.add(col)
                        st.success(f"✅ Replaced rogue entries in '{col}' with NaN.")
        
                with col3:
                    if st.button(f"Skip '{col}'", key=f"skip_{col}"):
                        st.session_state.fixed_mixed_cols.add(col)
                        st.info(f"⏭️ Skipped fixing '{col}'")
        
            unfixed = [col for col in st.session_state.mixed_type_cols if col not in st.session_state.fixed_mixed_cols]
            if unfixed:
                if st.button("⚙️ Bulk Convert All"):
                    for col in unfixed:
                        stats = st.session_state.type_info[col]
                        if stats['numeric_ratio'] > stats['string_ratio']:
                            df[col] = pd.to_numeric(df[col], errors='coerce')
                        else:
                            df[col] = df[col].astype(str)
                        st.session_state.fixed_mixed_cols.add(col)
                    st.success("✅ All mixed-type columns converted to dominant types.")
        
            st.session_state.processed_df = df



#####################################################################################################################################
        # SECTION 4: Standardize Column Data Types
        st.header("4. Standardize Column Data Types")
        st.markdown("""
        This step ensures your columns use appropriate base types:
        - Convert `object` columns to numeric or string.
        - Convert `float64` to `int64` if values are all whole numbers.
        - Optionally detect datetime-like strings and convert them.
        """)
        
        preview_fixes = []
        for col in df.columns:
            orig_dtype = df[col].dtype
            if orig_dtype == 'object':
                try:
                    df[col] = pd.to_numeric(df[col])
                    preview_fixes.append(f"{col}: object → numeric")
                except:
                    df[col] = df[col].astype(str)
                    preview_fixes.append(f"{col}: object → string")
            elif orig_dtype == 'float64':
                if df[col].dropna().apply(float.is_integer).all():
                    df[col] = df[col].astype('Int64')
                    preview_fixes.append(f"{col}: float64 → Int64")
        
        if st.checkbox("🔍 Attempt datetime conversion for object columns"):
            for col in df.select_dtypes(include=['object']):
                try:
                    converted = pd.to_datetime(df[col], errors='coerce')
                    if converted.notnull().sum() > 0:
                        df[col] = converted
                        preview_fixes.append(f"{col}: object → datetime64")
                except Exception:
                    pass
        
        if preview_fixes:
            st.success("Standardization applied:")
            for fix in preview_fixes:
                st.write(f"✅ {fix}")
            st.session_state.processed_df = df
        else:
            st.info("No data type conversions were needed.")

#####################################################################################################################################
        # SECTION 5: Optimize for Analysis
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
                        st.write(f"✅ {col}: object → category")
        
            if downcast_nums:
                for col in df.select_dtypes(include=['int64', 'float64']):
                    df[col] = pd.to_numeric(df[col], downcast='unsigned' if df[col].min() >= 0 else 'integer')
                    st.write(f"✅ {col}: downcasted for memory optimization")
        
            st.session_state.processed_df = df
            st.success("✅ Optimization complete!")

#####################################################################################################################################
        # SECTION 6: Detect and Handle Outliers
        st.header("6. Detect and Handle Outliers")
        st.markdown("""
        Flag and optionally remove extreme numeric values that fall far outside the typical range.
        Uses the **IQR method** (Interquartile Range) by default:
        - Outliers are values < Q1 - 1.5×IQR or > Q3 + 1.5×IQR
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
                st.success(f"✅ Found {len(outlier_report)} columns with outliers.")
                for col, stats in outlier_report.items():
                    st.markdown(f"**Column: `{col}`**")
                    st.write(f"- Outliers: {stats['count']} ({stats['percent']:.2f}%)")
                    st.write(f"- Lower bound: {stats['lower']:.2f}")
                    st.write(f"- Upper bound: {stats['upper']:.2f}")
                    c1, c2, c3 = st.columns(3)
        
                    with c1:
                        if st.button(f"Replace outliers in '{col}' with NaN", key=f"nan_{col}"):
                            df.loc[stats["mask"], col] = np.nan
                            st.session_state.processed_df = df
                            st.success(f"✅ Replaced outliers in '{col}' with NaN")
        
                    with c2:
                        if st.button(f"Drop rows with outliers in '{col}'", key=f"drop_{col}"):
                            df = df[~stats["mask"]]
                            st.session_state.processed_df = df
                            st.success(f"🗑️ Dropped {stats['count']} rows with outliers in '{col}'")
        
                    with c3:
                        if st.button(f"Skip column '{col}'", key=f"skip_{col}"):
                            st.info(f"⏭️ Skipped handling outliers in '{col}'")
            else:
                st.success("🎉 No outliers detected using the current IQR threshold.")

#####################################################################################################################################
        # SECTION 7: Clean and Normalize Text Data
        st.header("7. Clean and Normalize Text Data")
        st.markdown("""
        Clean up string contents to reduce noise and standardize formatting:
        - Strip leading/trailing spaces
        - Convert to lowercase
        - Replace common 'null-like' values (e.g., "NA", "n/a", "--")
        - Remove invisible characters (e.g., \\xa0, \\u200b)
        """)
        
        text_opts = st.multiselect(
            "Choose text cleanup actions:",
            options=[
                "Strip leading/trailing whitespace",
                "Convert to lowercase",
                "Replace common null-like strings",
                "Remove invisible characters"
            ],
            default=[
                "Strip leading/trailing whitespace",
                "Convert to lowercase",
                "Replace common null-like strings"
            ]
        )
        
        if st.button("Apply Text Cleanup"):
            df = st.session_state.processed_df.copy()
            cleaned_cols = []
        
            for col in df.select_dtypes(include=['object', 'string']):
                original = df[col].copy()
        
                if "Strip leading/trailing whitespace" in text_opts:
                    df[col] = df[col].astype(str).str.strip()
        
                if "Convert to lowercase" in text_opts:
                    df[col] = df[col].astype(str).str.lower()
        
                if "Replace common null-like strings" in text_opts:
                    df[col] = df[col].replace(
                        to_replace=["NA", "N/A", "na", "n/a", "--", "-", "", "null"],
                        value=np.nan
                    )
        
                if "Remove invisible characters" in text_opts:
                    import re
                    df[col] = df[col].astype(str).apply(lambda x: re.sub(r'[\u200b\xa0]', '', x))
        
                if not df[col].equals(original):
                    cleaned_cols.append(col)
        
            st.session_state.processed_df = df
        
            if cleaned_cols:
                st.success(f"✅ Cleaned text data in {len(cleaned_cols)} column(s): {', '.join(cleaned_cols)}")
                st.dataframe(df[cleaned_cols].head())
            else:
                st.info("No changes were made during text cleanup.")
                
#####################################################################################################################################
        # SECTION 8: Clean Column Names
        st.header("8. Clean Column Names")
        st.markdown("""
        Fix inconsistent column naming:
        - Remove leading/trailing whitespace
        - Standardize format (e.g., `snake_case`)
        - Remove special characters
        """)
        
        rename_opts = st.multiselect(
            "Choose column name cleanup actions:",
            options=[
                "Strip whitespace",
                "Standardize to snake_case",
                "Remove special characters"
            ],
            default=["Strip whitespace", "Standardize to snake_case"]
        )
        
        # Preview cleanup before applying
        def clean_column_name(name):
            import re
            original = name
            if "Strip whitespace" in rename_opts:
                name = name.strip()
            if "Remove special characters" in rename_opts:
                name = re.sub(r'[^\w\s]', '', name)
            if "Standardize to snake_case" in rename_opts:
                name = re.sub(r'\s+', '_', name).lower()
            return name
        
        if st.checkbox("🔍 Show before/after preview"):
            col_preview = pd.DataFrame({
                "Original Name": df.columns,
                "Cleaned Name": [clean_column_name(col) for col in df.columns]
            })
            st.dataframe(col_preview)
        
        if st.button("Apply Column Name Cleanup"):
            new_names = [clean_column_name(col) for col in df.columns]
            df.columns = new_names
            st.session_state.processed_df = df
            st.success("✅ Column names cleaned and updated!")
            st.dataframe(df.head())


#####################################################################################################################################
        # SECTION 9: Impute Missing Values
        st.header("9. Impute Missing Values")
        
        if 'impute_log' not in st.session_state:
            st.session_state.impute_log = []
        if 'df_backup' not in st.session_state:
            st.session_state.df_backup = None
        
        df = st.session_state.processed_df
        
        if df is not None:
            missing_stats = df.isnull().sum()
            total_missing = missing_stats.sum()
            missing_cols = missing_stats[missing_stats > 0]
        
            st.subheader("🧮 Missing Value Summary")
            if total_missing == 0:
                st.success("🎉 No missing values in your dataset!")
            else:
                percent_missing = (total_missing / (df.shape[0] * df.shape[1])) * 100
                st.write(f"🔍 **Total Missing Cells:** `{total_missing}` ({percent_missing:.2f}%)")
                st.dataframe(missing_cols.to_frame("Missing Count"))
        
                if st.checkbox("🗑 Drop rows with more than 50% missing values"):
                    initial_rows = df.shape[0]
                    df = df[df.isnull().mean(axis=1) < 0.5]
                    st.session_state.processed_df = df
                    st.success(f"✅ Dropped {initial_rows - df.shape[0]} rows.")
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
                    "Mean": "👉 Replaces missing values with the average of the column. Best for continuous data without extreme outliers.",
                    "Median": "👉 Replaces missing values with the median (middle) value. Robust to outliers.",
                    "Mode": "👉 Replaces missing values with the most frequently occurring value.",
                    "Fill with 'NA' (string literal)": "👉 Inserts the string 'NA' into missing cells. Best for text columns where 'NA' is meaningful.",
                    "Fill with custom value": "👉 Lets you manually enter any value to replace missing cells.",
                    "Forward Fill (LOCF)": "👉 Carries the last known value forward. Good for time-series or ordered data.",
                    "Backward Fill (NOCB)": "👉 Pulls the next valid value backward. Also useful in ordered datasets.",
                    # Advanced
                    "KNN Imputer": "🔍 Finds the 'k' most similar rows and uses their values to fill in the blanks. Great when patterns exist across columns.",
                    "Linear Regression": "📈 Uses other numeric columns to predict the missing value using a regression model.",
                    "Iterative Imputer (MICE)": "🔁 Models each column as a function of the others and iteratively predicts missing values.",
                    "MissForest (Random Forest)": "🌲 Uses random forests to fill missing values non-linearly. Very powerful, slower to run.",
                    "Interpolation": "📉 Connects the dots in numeric data (linear/spline interpolation). Good for time-continuous data.",
                    "Expectation Maximization (EM)": "📊 A statistical technique that guesses likely values. Not implemented.",
                    "Bayesian Imputation": "🧠 Samples values from a posterior probability distribution. Not implemented here."
                }
        
                if impute_mode == "Simple":
                    methods = ["Mean", "Median", "Mode", "Fill with 'NA' (string literal)", "Fill with custom value", "Forward Fill (LOCF)", "Backward Fill (NOCB)"]
                elif impute_mode == "Advanced":
                    methods = ["KNN Imputer", "Linear Regression", "Iterative Imputer (MICE)", "MissForest (Random Forest)", "Interpolation", "Expectation Maximization (EM)", "Bayesian Imputation"]
                elif impute_mode == "Categorical":
                    methods = ["Mode", "Fill with 'NA' (string literal)", "Fill with custom value"]
        
                selected_method = st.selectbox("Choose your imputation method", methods)
        
                with st.expander("ℹ️ What this method does"):
                    st.markdown(method_descriptions.get(selected_method, "No description available."))
        
                if selected_method == "Fill with custom value":
                    custom_value = st.text_input("Enter value to fill missing cells with:")
        
                if st.button("Apply Imputation"):
                    st.session_state.df_backup = df.copy()
        
                    def apply_imputation(col):
                        try:
                            if selected_method == "Mean":
                                df[col] = df[col].fillna(df[col].mean())
                            elif selected_method == "Median":
                                df[col] = df[col].fillna(df[col].median())
                            elif selected_method == "Mode":
                                df[col] = df[col].fillna(df[col].mode().iloc[0])
                            elif selected_method == "Fill with 'NA' (string literal)":
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
                                st.warning("⚠️ EM not implemented. Requires `fancyimpute`.")
                            elif selected_method == "Bayesian Imputation":
                                st.warning("⚠️ Bayesian imputation requires `pymc` and is not implemented.")
                            else:
                                st.error("❌ Unknown imputation method.")
                        except Exception as e:
                            st.error(f"⚠️ Failed to apply imputation: {e}")
        
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
        
                with st.expander("🧾 Imputation Log"):
                    if st.session_state.impute_log:
                        for idx, (col, method) in enumerate(reversed(st.session_state.impute_log[-10:]), 1):
                            st.write(f"{idx}. Column: **{col}**, Method: **{method}**")
                    else:
                        st.write("No imputations logged yet.")
        else:
            st.info("Please upload a CSV file to begin.")


#####################################################################################################################################            
        # SECTION 10: Download Processed Data
        st.header("10. Download Processed Data")
        
        df = st.session_state.processed_df
        
        if df is not None:
            st.markdown(f"Your cleaned dataset has **{df.shape[0]} rows** and **{df.shape[1]} columns**.")
            
            if st.button("Show Final Data Preview"):
                st.subheader("📋 Final Data Preview")
                st.dataframe(df.head())
        
            file_format = st.radio("Choose download format", ["CSV", "Excel (.xlsx)"], horizontal=True)
        
            if file_format == "CSV":
                csv_data = df.to_csv(index=False).encode("utf-8")
                st.download_button(
                    label="⬇️ Download CSV File",
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
                    label="⬇️ Download Excel File",
                    data=output.getvalue(),
                    file_name="cleaned_data.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                )
        else:
            st.info("Please upload and clean a dataset before downloading.")

