# Data Cleaning Tool

A comprehensive Streamlit-based application for interactive cleaning and preprocessing of tabular data. This tool provides an intuitive interface for handling common data quality issues in CSV files and Excel spreadsheets.

## Overview

The Data Cleaning Tool streamlines the data preprocessing workflow by providing a guided, step-by-step interface for identifying and resolving data quality issues in tabular datasets. Whether you're preparing data for analysis, machine learning, or visualization, this tool helps you clean and standardize your data efficiently.

## Key Features

### Current Capabilities

**Data Type Management**
- Detect and resolve mixed-type columns (numeric strings vs actual strings)
- Smart object column conversion with confidence scoring
- Automatic type inference for numeric, datetime, and string columns
- Safe float-to-integer conversion for whole numbers

**Missing Value Handling**
- Multiple imputation methods: Mean, Median, Mode, Forward/Backward Fill
- Advanced techniques: KNN Imputation, Linear Regression, MICE (Iterative Imputer), Interpolation
- Visual preview of imputation results with row context
- Undo functionality for iterative refinement

**Data Cleaning**
- Text normalization (whitespace removal, lowercase conversion, null-like string replacement)
- Column name standardization (snake_case conversion, special character removal)
- Outlier detection using IQR method with configurable thresholds
- Duplicate handling and empty row removal

**Performance Optimization**
- Categorical type conversion for low-cardinality columns
- Numeric type downcasting to reduce memory footprint
- Memory usage reporting and optimization suggestions

**Interactive Workflow**
- Section-based navigation for organized cleaning process
- Real-time data preview and statistics
- Enhanced information tables with missing value summaries
- Session state management for undo/redo operations

### Planned Features

- Multi-step undo across all sections
- Batch processing for multiple files
- Custom cleaning rule definitions
- Export cleaning pipeline as reusable Python scripts
- Data profiling and quality reports

## Quick Start

The easiest way to use the Data Cleaning Tool is through the hosted Streamlit app:

**Access the app here: [https://datacleaningtool.streamlit.app/](https://datacleaningtool.streamlit.app/)**

No installation required - simply navigate to the URL and start cleaning your data immediately.

### Working with Database Data

If your data is stored in a database (MySQL, PostgreSQL, Oracle, SQLite, etc.):

1. **Export your data to CSV** from your database using your preferred method
2. **Clean the data** using this tool
3. **Import the cleaned CSV** back into your database

This export/import workflow is recommended because it:
- Keeps your original data untouched (audit trail)
- Works with any database system
- Avoids security risks of direct database connections
- Allows testing cleaning operations before applying to production data

### Security and Privacy

**Data privacy:**
- All data processing happens in your browser session via Streamlit
- Uploaded files are stored temporarily in Streamlit's session state
- Data is automatically cleared when you close your browser or end the session
- The application developer cannot access your uploaded data

**Security considerations:**
- This tool uses Streamlit's built-in security measures
- For sensitive data, consider running the tool locally (see Installation section)
- Never upload data containing passwords, API keys, or other credentials
- Review Streamlit's security documentation at [https://docs.streamlit.io/](https://docs.streamlit.io/) for detailed information

**Disclaimer:** While reasonable security measures are in place, this tool is provided as-is. For highly sensitive or regulated data (HIPAA, PCI-DSS, etc.), consult your organization's security policies before using cloud-hosted tools.

## Installation (For Local Development)

If you want to run the tool locally or contribute to development:

### Prerequisites

- Python 3.7 or higher
- pip package manager

### Setup

1. Clone the repository:
```bash
git clone https://github.com/Besoninja/DataCleaningTool.git
cd DataCleaningTool
```

2. Install required dependencies:
```bash
pip install -r requirements.txt
```

3. Run the application:
```bash
streamlit run DataCleaningTool.py
```

The application will open in your default web browser at `http://localhost:8501`

## Dependencies

The tool relies on the following Python libraries:

- **streamlit** - Web application framework
- **pandas** - Data manipulation and analysis
- **numpy** - Numerical computing
- **matplotlib** - Plotting and visualization
- **seaborn** - Statistical data visualization
- **scikit-learn** - Machine learning utilities (imputation, preprocessing)
- **openpyxl** - Excel file support
- **xlrd** - Legacy Excel file reading

All dependencies are listed in `requirements.txt`.

## Table of Contents

- [Quick Start](#quick-start)
  - [Working with Database Data](#working-with-database-data)
  - [Security and Privacy](#security-and-privacy)
- [Installation](#installation-for-local-development)
- [Usage Guide](#usage-guide)
- [Undo Functionality](#undo-functionality)
- [Feature Reference](#feature-reference)
- [Best Practices](#best-practices)
- [Troubleshooting](#troubleshooting)
- [Reporting Issues](#reporting-issues)
- [Contributing](#contributing)
- [Roadmap](#roadmap)

## Usage Guide

The tool is organized into 10 sequential sections accessible via the sidebar:

1. **File Upload** - Upload CSV files and view initial data summary
2. **Data Overview** - Comprehensive statistics and missing value analysis
3. **Mixed-Type Columns** - Detect and resolve columns with inconsistent data types
4. **Object Conversion** - Smart conversion of object columns to appropriate types
5. **Optimize Analysis** - Memory optimization through categorical conversion and downcasting
6. **Handle Outliers** - IQR-based outlier detection and removal
7. **Clean Text Data** - Normalize text columns (whitespace, casing, null-like strings)
8. **Clean Column Names** - Standardize column naming conventions
9. **Impute Missing Values** - Multiple imputation methods with preview and undo
10. **Download Processed Data** - Export cleaned data as CSV or Excel

Each section includes detailed explanations and previews before applying changes. For comprehensive documentation on each feature, refer to the in-app help text and expandable information sections.

## Undo Functionality

### Current Implementation

The undo feature is currently available **only in Section 9: Impute Missing Values**.

**How it works:**
- Before each imputation operation, the tool creates a backup of your dataframe
- Click "Undo Last Imputation" to revert to the previous state
- The imputation log shows your recent operations for reference

**Important Limitations:**
- **Single-step undo only**: You can only undo the most recent imputation operation
- **Section-specific**: Undo is not available for operations in other sections (mixed-type resolution, text cleaning, etc.)
- **Overwritten on next operation**: When you apply a new imputation, the previous backup is replaced
- **Lost on navigation**: Backups may be cleared if you navigate away from the section

**Workaround for Multi-Step Undo:**

Until multi-step undo is implemented, we recommend:
1. Export your data after critical cleaning steps (Section 10)
2. Keep timestamped backups of important intermediate states
3. Test imputation methods on a sample before applying to full dataset
4. Use the imputation log to track what methods you've tried

**Planned Improvements (Future Versions):**
- Multi-step undo across all sections
- Full operation history with selective rollback
- Named checkpoints for major cleaning stages
- Automatic backup before destructive operations

### Best Practices with Current Undo

- Apply imputation methods one column at a time to test results
- Review the imputation log regularly to track your progress
- Use the preview feature before applying to see expected results
- Export intermediate results before trying risky operations
- Document your cleaning decisions for reproducibility

## Feature Reference

### Imputation Methods

**Simple:** Mean, Median, Mode, Forward/Backward Fill  
**Advanced:** KNN Imputer, Linear Regression, MICE, Interpolation

Detailed explanations for each method are available in-app within Section 9.

### Outlier Detection

Uses IQR (Interquartile Range) method with configurable multiplier (default: 1.5)

### Memory Optimization

**Categorical conversion:** Reduces memory for low-cardinality string columns  
**Numeric downcasting:** Converts to smaller types when value range allows

## Technical Notes

The application uses Streamlit's session state to maintain the current dataframe, navigation state, imputation history, and backup dataframes for undo operations. Each cleaning operation modifies the dataframe in-place, and changes persist across section navigation.

## Best Practices

**Recommended cleaning order:**
1. Resolve mixed-type columns
2. Convert object columns to appropriate types
3. Clean column names
4. Handle outliers
5. Clean text data
6. Impute missing values
7. Optimize data types

**For large datasets:**
- Test cleaning pipeline on a sample first
- Use category conversion for repeated string values
- Apply imputation selectively (advanced methods can be slow)
- Export intermediate results frequently

## Troubleshooting

**Imputation fails with "Not enough data":** Ensure sufficient non-null values in feature columns  
**Column name standardization creates duplicates:** Manually rename conflicting columns first  
**Memory errors with large datasets:** Apply optimizations earlier in the workflow  
**Undo not working:** Backups only kept for last operation in Section 9

For additional help, see the [Reporting Issues](#reporting-issues) section.

## Reporting Issues

Found a bug or have a feature request? We track all issues on GitHub.

### How to Submit an Issue

1. **Navigate to the Issues page:**  
   Go to [https://github.com/Besoninja/DataCleaningTool/issues](https://github.com/Besoninja/DataCleaningTool/issues)

2. **Click "New Issue":**  
   You'll need a GitHub account (free to create)

3. **Select a template:**
   - **Bug Report** - For unexpected behavior or errors
   - **Feature Request** - For suggesting new functionality  
   - **Question** - For general questions or clarifications

4. **Fill out the template:**  
   The template will guide you through providing the necessary information

GitHub will automatically format your issue using the appropriate template. Please provide as much detail as possible to help resolve your issue quickly.

**Note:** When reporting bugs with datasets, describe the data characteristics (size, types, patterns) but never include actual data files or sensitive information.

## Roadmap

### Version 0.2.0 (Planned)
- Multi-step undo functionality across all sections
- Named checkpoints for saving cleaning states
- Batch file processing (multiple CSVs at once)
- Export cleaning operations as reusable Python script

### Version 0.3.0 (Planned)
- Custom cleaning rule builder
- Data profiling reports (distributions, correlations, quality scores)
- Automated cleaning suggestions based on data patterns
- Column relationship analysis

### Version 0.4.0 (Future)
- Scheduled cleaning workflows
- API for programmatic access
- Plugin system for custom transformations
- Docker containerization for enterprise deployment

## Contributing

Suggestions and feedback are very welcome! Here's how you can participate:

**Submit ideas and bug reports:**
- Open an issue on GitHub to suggest features or report bugs
- All suggestions are reviewed and considered for future versions

**Fork and customize:**
- Feel free to fork this repository and modify it for your own needs
- Create your own version with custom features
- No permission needed - that's what open source is for!

**Pull requests:**
- This repository is maintained solely by the project owner
- Pull requests are not accepted at this time
- If you've built something cool in your fork, share it in an issue - it might inspire future features!

The best way to influence the project's direction is through detailed feature requests and bug reports in the Issues section.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- Built with Streamlit for rapid web app development
- Uses scikit-learn for robust imputation algorithms
- Inspired by common data cleaning challenges in real-world analytics projects

## Contact

For questions, suggestions, or bug reports, please open an issue on GitHub.

## Version History

### Version 0.1.0 (Current)
- Initial release
- Core data cleaning functionality for CSV files
- 10 integrated cleaning sections
- Multiple imputation methods
- Interactive outlier detection
- Session state management with undo capability
