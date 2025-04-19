import pandas as pd
import re

def extract_numeric_dose(sig):
    if isinstance(sig, str):
        match = re.search(r'[\d.]+', sig)
        if match:
            return float(match.group())
    return None

def categorize_lab(row):
    if row['Collection Datetime'] < row['IN_OR_DTTM']:
        return 'Pre-Op'
    elif row['IN_OR_DTTM'] <= row['Collection Datetime'] <= row['OUT_OR_DTTM']:
        return 'Intra-Op'
    else:
        return 'Post-Op'
        
def height_to_inches(height):
    if pd.isna(height) or '\'' not in height:
        return None  # Return None if height is NaN or does not contain a proper format
    try:
        parts = height.split('\'')
        feet = int(parts[0])  # Convert feet part to integer
        inches = int(parts[1].replace('"', '')) if len(parts) > 1 and parts[1] != '' else 0
        return feet * 12 + inches  # Convert feet to inches and add inches
    except (ValueError, IndexError):
        return None  # Handle cases where conversion is not possible

def summary_stat(df_patient_info_final):
    columns_to_analyze = df_patient_info_final.columns
    # Initialize a list to store the results
    results = []

    # Loop through each specified column and calculate required metrics
    for column in columns_to_analyze:
        if column in df_patient_info_final.columns:  # Check if the column exists in the DataFrame
            null_count = df_patient_info_final[column].isnull().sum()  # Count of null values
            unique_count = df_patient_info_final[column].nunique()     # Unique count of values
            null_percentage = (null_count / len(df_patient_info_final)) * 100  # Null percentage
            
            # Append results to the list
            results.append({
                'Column Name': column,
                'Null Count': null_count,
                'Unique Count': unique_count,
                'Null Percentage': null_percentage
            })
        else:
            # If the column doesn't exist, you can log it or handle it as needed
            results.append({
                'Column Name': column,
                'Null Count': None,
                'Unique Count': None,
                'Null Percentage': None
            })

    # Create a DataFrame from the results
    summary_df = pd.DataFrame(results)

    # Display the summary
    print(summary_df)