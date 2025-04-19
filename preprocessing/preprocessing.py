import os
import warnings
import pandas as pd
import argparse
import tarfile
import logging
import pickle
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from dutils import *
from tqdm import tqdm
from sklearn.exceptions import DataConversionWarning
from sklearn.model_selection import train_test_split

# Suppress specific warnings
warnings.filterwarnings(action='ignore', category=DataConversionWarning)
# Or to ignore all warnings
warnings.filterwarnings(action='ignore')

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def patient_filtering(dataframes):
    def keyword_filter(df, column, keywords):
        """Return rows where all keywords are found in the specified column (case-insensitive)."""
        mask = df[column].str.contains(keywords[0], case=False, na=False)
        for kw in keywords[1:]:
            mask &= df[column].str.contains(kw, case=False, na=False)
        return df[mask]

    # Filter 1: From patient_coding - look for both 'fusion' and 'spine'
    fusion_spine_keywords = ['fusion', 'spine']
    filtered_coding = keyword_filter(dataframes["patient_coding"], 'NAME', fusion_spine_keywords)
    matched_coding = pd.merge(dataframes["patient_info"], filtered_coding, on='MRN')

    # Filter 2: From patient_visit - same keywords in 'dx_name'
    filtered_visit = keyword_filter(dataframes["patient_visit"], 'dx_name', fusion_spine_keywords)
    matched_visit = pd.merge(dataframes["patient_info"], filtered_visit, on='LOG_ID')

    # Filter 3: From patient_info - look for spinal-related procedures
    spinal_keywords = ['spine']
    filtered_procedures = dataframes["patient_info"][
        dataframes["patient_info"]['PRIMARY_PROCEDURE_NM'].str.contains('|'.join(spinal_keywords), case=False, na=False)
    ]

    # Combine all filtered results
    combined = pd.concat([matched_coding, matched_visit, filtered_procedures], ignore_index=True)

    # Final filter: Ensure final set has both 'fusion' and 'spin' in procedure name
    final_filter_keywords = ['fusion', 'spin']
    filtered_final = keyword_filter(combined, 'PRIMARY_PROCEDURE_NM', final_filter_keywords)

    # Remove duplicates
    final_dataset = filtered_final.drop_duplicates()
    
    final_dataset['HEIGHT_INCHES'] = final_dataset['HEIGHT'].apply(height_to_inches)

    # Summary stats
    logger.info(f"{final_dataset['MRN'].nunique()} unique patients who had spinal fusion surgery.")
    logger.info(f"{dataframes['patient_info']['MRN'].nunique()} total unique patients in the database.")

    return final_dataset

def preprocess_patient_labs(filtered_dataset, dataframes):
    # Drop irrelevant columns from filtered_dataset
    cols_to_drop = [
        'SOURCE_KEY', 'SOURCE_NAME', 'REF_BILL_CODE_SET_NAME', 'mrn', 'NAME', 'SURGERY_DATE', 
        'HEIGHT', 'REF_BILL_CODE', 'LOS', 'diagnosis_code', 'dx_name', 'AN_STOP_DATETIME',
        'AN_START_DATETIME', 'HOSP_DISCH_TIME', 'ICU_ADMIN_FLAG'
    ]
    filtered_dataset = filtered_dataset.drop(columns=[col for col in cols_to_drop if col in filtered_dataset.columns]).drop_duplicates()

    # Convert IN_OR_DTTM to datetime
    filtered_dataset['IN_OR_DTTM'] = pd.to_datetime(filtered_dataset['IN_OR_DTTM'], format='%m/%d/%y %H:%M', errors='coerce')

    # Merge lab data with patients present in final dataset
    labs = dataframes["patient_labs"]
    labs = labs.merge(filtered_dataset[['LOG_ID']], on='LOG_ID', how='inner')

    # Convert lab collection datetime
    labs['Collection Datetime'] = pd.to_datetime(labs['Collection Datetime'], format='%Y-%m-%d %H:%M:%S', errors='coerce')

    # Merge lab data with IN_OR_DTTM to filter preoperative labs
    labs_merged = labs.merge(filtered_dataset[['LOG_ID', 'MRN', 'IN_OR_DTTM']], on=['LOG_ID', 'MRN'], how='left')
    labs_preop = labs_merged[labs_merged['Collection Datetime'] < labs_merged['IN_OR_DTTM']].copy()

    # Find closest pre-op lab per patient
    labs_preop['time_diff'] = (labs_preop['IN_OR_DTTM'] - labs_preop['Collection Datetime']).abs()
    labs_closest = labs_preop.sort_values(by=['LOG_ID', 'MRN', 'time_diff']).groupby(['LOG_ID', 'MRN']).first().reset_index()

    # Drop unnecessary columns
    labs = labs.drop(columns=['Measurement Units'], errors='ignore').drop_duplicates()

    # Pivot lab values
    def pivot_labs(labs_df, value_col, suffix):
        pivot = labs_df.pivot_table(index=['LOG_ID', 'MRN'], columns='Lab Name', values=value_col, aggfunc='first')
        pivot.columns = [f"{col}_{suffix}" for col in pivot.columns]
        return pivot

    pivot_values = pivot_labs(labs_closest, 'Observation Value', 'value') # 959 features when passing labs instead of labs_closest
    pivot_flags = pivot_labs(labs_closest, 'Abnormal Flag', 'abnormal').applymap(lambda x: x.strip() if isinstance(x, str) else x)
    # pivot_times = pivot_labs(labs_closest, 'Collection Datetime', 'time')  # optional if you want timestamp features

    # Combine pivoted lab features
    pivot_combined = pd.concat([pivot_values.apply(pd.to_numeric, errors='coerce'), pivot_flags], axis=1).reset_index()

    # Merge back with final dataset
    filtered_dataset_with_labs = filtered_dataset.merge(pivot_combined, on=['LOG_ID', 'MRN'], how='left').drop_duplicates()

    return filtered_dataset_with_labs

def preprocess_patient_medications(filtered_dataset_with_labs, dataframes):
    meds = dataframes["patient_medications"].copy()
    
    # Keep only valid RECORD_TYPEs (non-null, PRE-OP)
    meds = meds[meds['RECORD_TYPE'].notna() & (meds['RECORD_TYPE'].str.upper() == 'PRE-OP')]

    # Convert to datetime
    meds['START_DATE'] = pd.to_datetime(meds['START_DATE'], errors='coerce')
    meds['END_DATE'] = pd.to_datetime(meds['END_DATE'], errors='coerce')

    # Merge with surgery time to identify true pre-op meds
    meds = meds.merge(
        filtered_dataset_with_labs[['LOG_ID', 'MRN', 'IN_OR_DTTM']],
        on=['LOG_ID', 'MRN'],
        how='left'
    )

    # Keep only meds administered before surgery
    meds = meds[meds['START_DATE'] < meds['IN_OR_DTTM']]

    # Compute duration in days
    meds['duration'] = (meds['END_DATE'] - meds['START_DATE']).dt.days

    # Filter on medication action types (given)
    given_actions = {
        'Given', 'Given by Other', 'Given in incremental doses', 
        'Bolus', 'Bolus From Bag', 'IV Resume', 'Patch Applied', 'Restarted'
    }
    meds = meds[meds['MAR_ACTION_NM'].isin(given_actions)]

    # Extract dose from ADMIN_SIG
    meds['dose'] = meds['ADMIN_SIG'].apply(extract_numeric_dose)

    # Clean medication name
    meds['MEDICATION_NAME_DOSAGE'] = meds['MEDICATION_NM'].fillna('Unknown') + ', ' + meds['DOSE_UNIT_NM'].fillna('Unknown')

    # Group by patient and medication
    med_summary = meds.groupby(['LOG_ID', 'MRN', 'MEDICATION_NAME_DOSAGE']).agg(
        total_duration=('duration', 'sum'),
        total_dose=('dose', 'sum')
    ).reset_index()

    # Pivot duration and dose
    pivot_duration = med_summary.pivot(index=['LOG_ID', 'MRN'], columns='MEDICATION_NAME_DOSAGE', values='total_duration')
    pivot_dose = med_summary.pivot(index=['LOG_ID', 'MRN'], columns='MEDICATION_NAME_DOSAGE', values='total_dose')

    pivot_duration.columns = [f"{col}_duration" for col in pivot_duration.columns]
    pivot_dose.columns = [f"{col}_dose" for col in pivot_dose.columns]

    # Combine features
    meds_pivoted = pd.concat([pivot_duration, pivot_dose], axis=1).reset_index().fillna(0)

    # Merge with main dataset
    merged = filtered_dataset_with_labs.merge(meds_pivoted, on=['LOG_ID', 
    'MRN'], how='left')

    return merged

def preprocess_patient_events(final_dataset_with_medications, dataframes):
    # Merge procedure events with surgery times
    merged_data = pd.merge(
        dataframes["procedure_events"], 
        final_dataset_with_medications[['LOG_ID', 'MRN', 'IN_OR_DTTM']], 
        on=['LOG_ID', 'MRN'], 
        how='left'
    )

    # Keep only preoperative events
    preop_events = merged_data[merged_data['EVENT_TIME'] < merged_data['IN_OR_DTTM']]

    # Merge event data with main dataset
    merged_with_events = final_dataset_with_medications.merge(
        preop_events[['LOG_ID', 'MRN', 'EVENT_DISPLAY_NAME', 'EVENT_TIME']],
        on=['LOG_ID', 'MRN'], 
        how='left'
    )

    # Convert EVENT_TIME to seconds
    merged_with_events['EVENT_TIME'] = pd.to_timedelta(
        merged_with_events['EVENT_TIME'], errors='coerce'
    ).dt.total_seconds()

    # Aggregate total event time per event type
    grouped_events = merged_with_events.groupby(
        ['LOG_ID', 'MRN', 'EVENT_DISPLAY_NAME']
    ).agg({'EVENT_TIME': 'sum'}).reset_index()

    # Pivot to wide format: one feature per event type
    pivoted_events = grouped_events.pivot_table(
        index=['LOG_ID', 'MRN'], 
        columns='EVENT_DISPLAY_NAME', 
        values='EVENT_TIME', 
        fill_value=0
    ).reset_index()

    # Rename columns for clarity
    pivoted_events.columns.name = None
    pivoted_events.columns = ['LOG_ID', 'MRN'] + [f"{col}_time" for col in pivoted_events.columns if col not in ['LOG_ID', 'MRN']]

    # Merge back to main dataset
    final_dataset_with_events = final_dataset_with_medications.merge(
        pivoted_events, 
        on=['LOG_ID', 'MRN'], 
        how='left'
    )

    return final_dataset_with_events

def preprocess_patient_lda(final_dataset_with_event_encoded, dataframes):
    # Ensure OR timestamps are in datetime format
    final_dataset_with_event_encoded['IN_OR_DTTM'] = pd.to_datetime(final_dataset_with_event_encoded['IN_OR_DTTM'])
    final_dataset_with_event_encoded['OUT_OR_DTTM'] = pd.to_datetime(final_dataset_with_event_encoded['OUT_OR_DTTM'])

    # Merge LDA data with surgical timestamps
    lda = dataframes["patient_lda"].merge(
        final_dataset_with_event_encoded[['LOG_ID', 'MRN', 'IN_OR_DTTM']],
        on=['LOG_ID', 'MRN']
    )

    # Convert placement and removal timestamps
    lda['placement_instant'] = pd.to_datetime(lda['placement_instant'], format='%m/%d/%y %H:%M', errors='coerce')
    lda['removal_instant'] = pd.to_datetime(lda['removal_instant'], format='%m/%d/%y %H:%M', errors='coerce')

    # Filter for LDAs removed before surgery
    lda_preop = lda[lda['removal_instant'] < lda['IN_OR_DTTM']].copy()
    lda_preop['placement_dur'] = (
        lda_preop['removal_instant'] - lda_preop['placement_instant']
    ).dt.total_seconds()

    # Drop non-informative durations
    lda_preop = lda_preop[lda_preop['placement_dur'] != 0]

    # Keep relevant columns and group
    lda_grouped = lda_preop[['LOG_ID', 'MRN', 'flo_meas_name', 'placement_dur', 'site']]
    lda_grouped = lda_grouped.groupby(['LOG_ID', 'MRN', 'flo_meas_name']).agg(
        placement_dur=('placement_dur', 'sum'),
        site=('site', 'first')
    ).reset_index()

    # One-hot encode flo_meas_name and site, weighted by duration
    one_hot_flo_meas = pd.get_dummies(lda_grouped['flo_meas_name'], prefix='flo_meas')
    one_hot_site = pd.get_dummies(lda_grouped['site'], prefix='site')

    one_hot_flo_meas_dur = one_hot_flo_meas.mul(lda_grouped['placement_dur'], axis=0)
    one_hot_site_dur = one_hot_site.mul(lda_grouped['placement_dur'], axis=0)

    # Combine with identifiers
    lda_encoded = pd.concat([lda_grouped[['LOG_ID', 'MRN']], one_hot_flo_meas_dur, one_hot_site_dur], axis=1)
    lda_encoded = lda_encoded.groupby(['LOG_ID', 'MRN']).sum().reset_index()

    # Merge with main dataset
    final_dataset_with_lda = final_dataset_with_event_encoded.merge(
        lda_encoded,
        on=['LOG_ID', 'MRN'],
        how='left'
    )

    return final_dataset_with_lda

def preprocess_patient_history(final_dataset_with_lda_encoded, dataframes, nan_threshold=0.80):

    # Step 1: Filter history for patients in final dataset
    filtered_history = dataframes["patient_history"][
        dataframes["patient_history"]['mrn'].isin(final_dataset_with_lda_encoded['MRN'])
    ]

    # Step 2: One-hot encode diagnosis names
    one_hot_encoded = pd.get_dummies(filtered_history['dx_name'], prefix='', prefix_sep='')
    one_hot_encoded.columns = [f"{col}_history" for col in one_hot_encoded.columns]

    # Step 3: Concatenate MRN with encoded history
    history_encoded = pd.concat([filtered_history[['mrn']], one_hot_encoded], axis=1)

    # Step 4: Aggregate to ensure one row per MRN
    history_encoded = history_encoded.groupby('mrn', as_index=False).max()
    history_encoded.rename(columns={'mrn': 'MRN'}, inplace=True)

    # Step 5: Merge with main dataset
    final_df = final_dataset_with_lda_encoded.merge(history_encoded, on='MRN', how='left')

    # Step 6: Drop duplicate suffix columns if any (_x, _y)
    suffix_cols = [col for col in final_df.columns if col.endswith('_x') or col.endswith('_y')]
    final_df.drop(columns=suffix_cols, inplace=True)

    # Step 7: Drop columns with > threshold % missing values
    max_nan_count = int(nan_threshold * len(final_df))
    initial_column_count = final_df.shape[1]
    final_df_trimmed = final_df.loc[:, final_df.isnull().sum() <= max_nan_count]
    columns_deleted = initial_column_count - final_df_trimmed.shape[1]

    logger.info(f"Number of columns deleted: {columns_deleted}")

    return final_df_trimmed

def add_post_op_complication_label(final_df, dataframes, cap=3):
    """
    Adds a complication label to the final dataset based on post-operative complications.
    Caps the label value at a specified maximum (default = 3).
    """

    # Step 1: Filter post-op complications for patients in final dataset
    filtered_post_op = dataframes["post_op_complications"].merge(
        final_df[['LOG_ID', 'MRN']], 
        on=['LOG_ID', 'MRN'], 
        how='inner'
    )

    # Step 2: Drop irrelevant columns
    filtered_post_op = filtered_post_op.drop(
        columns=['CONTEXT_NAME', 'Element_Name', 'SMRTDTA_ELEM_VALUE'], 
        errors='ignore'
    )

    # Step 3: Count complications per patient
    grouped_patients = filtered_post_op.groupby(['LOG_ID', 'MRN']).size().reset_index(name='count')

    # Step 4: Cap the count to avoid large outliers
    grouped_patients['count'] = grouped_patients['count'].apply(lambda x: min(x, cap))

    # Step 5: Merge back into the main dataset
    merged_with_labels = final_df.merge(
        grouped_patients[['LOG_ID', 'MRN', 'count']],
        on=['LOG_ID', 'MRN'],
        how='left'
    ).fillna(0)

    # Step 6: Rename and extract label column
    merged_with_labels['complication_label'] = merged_with_labels['count']
    complication_labels = merged_with_labels[['LOG_ID', 'MRN', 'complication_label']]

    # Step 7: Merge back to retain only patients with labels
    complete_df = final_df.merge(complication_labels, on=['LOG_ID', 'MRN'], how='inner').drop_duplicates()

    return complete_df

def finalize_feature_table(complete_df, mapping_file_path='./label_mappings.txt', test_size=0.2, random_state=42, save_path='./temp_data'):
    # Ensure LOG_ID and MRN are strings
    complete_df['LOG_ID'] = complete_df['LOG_ID'].astype(str)
    complete_df['MRN'] = complete_df['MRN'].astype(str)

    # Identify numeric columns and _value columns
    numerical_cols = complete_df.select_dtypes(include=['float64', 'int64']).columns
    value_cols = complete_df.columns[complete_df.columns.str.endswith('_value')]
    numerical_cols = pd.Index(set(numerical_cols).union(value_cols))

    # Convert to numeric and fill NaNs with 0
    for col in numerical_cols:
        complete_df[col] = pd.to_numeric(complete_df[col], errors='coerce')
    complete_df[numerical_cols] = complete_df[numerical_cols].fillna(0)

    # Convert HOSP_ADMSN_TIME to datetime
    complete_df['HOSP_ADMSN_TIME'] = pd.to_datetime(complete_df['HOSP_ADMSN_TIME'], errors='coerce')

    # Handle categorical columns
    categorical_cols = complete_df.select_dtypes(include=['object']).columns.drop(['LOG_ID', 'MRN'])
    complete_df[categorical_cols] = complete_df[categorical_cols].fillna('NA')

    # Custom mapping for abnormal flags
    abnormal_mapping = {'L': 0, 'N': 1, 'H': 2, 'NA': 3}

    # Write mappings to file and encode
    with open(mapping_file_path, 'w') as f:
        for col in categorical_cols:
            if col.endswith("_abnormal"):
                complete_df[col] = complete_df[col].astype(str).str.upper().map(abnormal_mapping)
                f.write(f"Column: {col} (custom abnormal encoding)\n")
                for key, value in abnormal_mapping.items():
                    f.write(f"  {key} = {value}\n")
            else:
                label_encoder = LabelEncoder()
                complete_df[col] = label_encoder.fit_transform(complete_df[col].astype(str))
                mapping = dict(zip(label_encoder.classes_, label_encoder.transform(label_encoder.classes_)))
                f.write(f"Column: {col}\n")
                for key, value in mapping.items():
                    f.write(f"  {key} = {value}\n")
            f.write("\n")

    # Convert timedelta columns to seconds
    for col in complete_df.select_dtypes(include='timedelta64[ns]').columns:
        complete_df[col] = complete_df[col].dt.total_seconds()

    # Extract parts from datetime columns
    datetime_columns = complete_df.select_dtypes(include='datetime64[ns]').columns
    for col in datetime_columns:
        complete_df[col + '_year'] = complete_df[col].dt.year
        complete_df[col + '_month'] = complete_df[col].dt.month
        complete_df[col + '_day'] = complete_df[col].dt.day
    complete_df = complete_df.drop(columns=datetime_columns)

    # Map surgical regions and approaches
    surgical_regions_mapping = {
        0: "Cervical", 1: "Lumbar", 2: "Cervical", 3: "Occiput and Cervical", 4: "Unspecified",
        5: "Unspecified", 6: "Cervical", 7: "Cervical", 8: "Lumbar and Lumbosacral", 9: "Lumbar and Lumbosacral",
        10: "Lumbar", 11: "Lumbar", 12: "Lumbar", 13: "Lumbar", 14: "Thoracic", 15: "Thoracic and/or Lumbar",
        16: "Thoracic", 17: "Thoracolumbar", 18: "Thoracolumbar", 19: "Unspecified", 20: "Unspecified"
    }
    approach_mapping = {
        0: 'Anterior Approach', 1: 'Posterior Approach', 2: 'Anterior Approach', 3: 'Posterior Approach',
        4: 'Anterior Approach', 5: 'Anterior Approach', 6: 'Anterior Approach', 7: 'Posterior Approach',
        8: 'Anterior Approach', 9: 'Posterior Approach', 10: 'Anterior Approach', 11: 'Anterior Approach',
        12: 'Posterior Approach', 13: 'Posterior Approach', 14: 'Posterior Approach', 15: 'Posterior Approach',
        16: 'Anterior Approach', 17: 'Posterior Approach', 18: 'Posterior Approach', 19: 'Posterior Approach',
        20: 'Anterior Approach'
    }
    complete_df['Surgical_Region'] = complete_df['PRIMARY_PROCEDURE_NM'].map(surgical_regions_mapping)
    complete_df['Treatment_Approach'] = complete_df['PRIMARY_PROCEDURE_NM'].map(approach_mapping)

    # Convert specified columns to Int64
    for col in ['DISCH_DISP_C', 'ASA_RATING_C']:
        if col in complete_df.columns:
            complete_df[col] = complete_df[col].astype('Int64')

    # Convert BIRTH_DATE to float
    if 'BIRTH_DATE' in complete_df.columns:
        complete_df['BIRTH_DATE'] = complete_df['BIRTH_DATE'].astype(float)

    # Convert extracted date parts to Int64
    date_part_cols = [col for col in complete_df.columns if col.endswith(('_year', '_month', '_day'))]
    complete_df[date_part_cols] = complete_df[date_part_cols].astype('Int64')

    # --- Model Prep ---
    regions_encoder = LabelEncoder()
    complete_df['Surgical_Region'] = regions_encoder.fit_transform(complete_df['Surgical_Region'])

    approach_encoder = LabelEncoder()
    complete_df['Treatment_Approach'] = approach_encoder.fit_transform(complete_df['Treatment_Approach'])

    # Save encoders' mappings to label_mappings.txt
    with open(mapping_file_path, 'a') as f:
        f.write("Column: Surgical_Region\n")
        for key, value in zip(regions_encoder.classes_, regions_encoder.transform(regions_encoder.classes_)):
            f.write(f"  {key} = {value}\n")
        f.write("\n")

        f.write("Column: Treatment_Approach\n")
        for key, value in zip(approach_encoder.classes_, approach_encoder.transform(approach_encoder.classes_)):
            f.write(f"  {key} = {value}\n")
        f.write("\n")

    # List columns with missing values
    nan_columns = complete_df.columns[complete_df.isna().any()]
    if not nan_columns.empty:
        logger.info("Columns with NaNs:")
        for col in nan_columns:
            logger.info(f"  {col} - {complete_df[col].isna().sum()} missing")

    # List columns that are still categorical (object type)
    categorical_remaining = complete_df.select_dtypes(include=['object']).columns
    if not categorical_remaining.empty:
        logger.info("Columns still categorical (object dtype):")
        for col in categorical_remaining:
            logger.info(f"  {col}")

    # Handle missing values in extracted date parts using most frequent (mode)
    date_part_cols_with_na = [
        'IN_OR_DTTM_year', 'IN_OR_DTTM_month', 'IN_OR_DTTM_day',
        'OUT_OR_DTTM_year', 'OUT_OR_DTTM_month', 'OUT_OR_DTTM_day'
    ]

    for col in date_part_cols_with_na:
        if col in complete_df.columns:
            mode_val = complete_df[col].mode(dropna=True)
            if not mode_val.empty:
                complete_df[col].fillna(mode_val[0], inplace=True)
            complete_df[col] = complete_df[col].astype('Int64')  # Ensure consistent dtype

    # Separate features and labels
    X = complete_df.drop(columns=['LOG_ID', 'MRN', 'complication_label', 'Treatment_Approach', 'PRIMARY_PROCEDURE_NM'])
    treatment_and_outcome = complete_df[['Treatment_Approach', 'complication_label']]

    # Normalize numeric features
    numeric_features = X.select_dtypes(include='float64').columns.tolist()
    scaler = MinMaxScaler()
    X_scaled = X.copy()
    X_scaled[numeric_features] = scaler.fit_transform(X[numeric_features])

    # Train/test split
    X_train, X_test, treatment_and_outcome_train, treatment_and_outcome_test = train_test_split(
        X_scaled, treatment_and_outcome, test_size=test_size, random_state=random_state
    )
    X_train = X_train.astype(np.float32).values
    X_test = X_test.astype(np.float32).values
    y_train = treatment_and_outcome_train['complication_label'].astype(np.float32).values
    y_test = treatment_and_outcome_test['complication_label'].astype(np.float32).values
    w_train = treatment_and_outcome_train['Treatment_Approach'].astype(np.float32).values
    w_test = treatment_and_outcome_test['Treatment_Approach'].astype(np.float32).values

    logger.info(f"X_train shape: {X_train.shape}")
    logger.info(f"y_train shape: {y_train.shape}")
    logger.info(f"w_train shape: {w_train.shape}")
    logger.info(f"X_test shape: {X_test.shape}")
    logger.info(f"y_test shape: {y_test.shape}")
    logger.info(f"w_test shape: {w_test.shape}")
    

    # --- Save to Pickle ---
    os.makedirs(save_path, exist_ok=True)

    with open(os.path.join(save_path, 'X_train.pkl'), 'wb') as f:
        pickle.dump(X_train, f)
    with open(os.path.join(save_path, 'y_train.pkl'), 'wb') as f:
        pickle.dump(y_train, f)
    with open(os.path.join(save_path, 'w_train.pkl'), 'wb') as f:
        pickle.dump(w_train, f)
    with open(os.path.join(save_path, 'X_test.pkl'), 'wb') as f:
        pickle.dump(X_test, f)
    with open(os.path.join(save_path, 'y_test.pkl'), 'wb') as f:
        pickle.dump(y_test, f)
    with open(os.path.join(save_path, 'w_test.pkl'), 'wb') as f:
        pickle.dump(w_test, f)

def main():
    """
    Main function to handle input and output directories and run preprocessing.
    """
    parser = argparse.ArgumentParser(description="Preprocess data and save to output directory.")
    parser.add_argument("input_dir", type=str, help="Path to the raw input gz data.")
    parser.add_argument("output_dir", type=str, help="Path to the output directory to save preprocessed data.")

    args = parser.parse_args()

    # Retrieve files to preprocess
    dataframes = {}

    # Initialize a progress bar
    with tqdm(total=9, desc="Preprocessing Steps", unit="step") as pbar:
        # Open the tar.gz archive
        tqdm.write("Loading data")
        with tarfile.open(args.input_dir, 'r:gz') as tar:
            # Identify all .csv files in the archive
            csv_files = [member.name for member in tar.getmembers() if member.name.endswith('.csv')]
            # logger.info("CSV files in the archive:", csv_files)
            
            # Loop through the CSV files and load them
            for csv_file in csv_files:
                with tar.extractfile(csv_file) as file:
                    # Determine which DataFrame to load into based on file name
                    if "patient_labs" in csv_file:
                        dataframes["patient_labs"] = pd.read_csv(file)
                    elif "patient_info" in csv_file:
                        dataframes["patient_info"] = pd.read_csv(file)
                    elif "procedure events" in csv_file:
                        dataframes["procedure_events"] = pd.read_csv(file)
                    elif "patient_medications" in csv_file:
                        dataframes["patient_medications"] = pd.read_csv(file)
                    elif "post_op_complications" in csv_file:
                        dataframes["post_op_complications"] = pd.read_csv(file)
                    elif "patient_history" in csv_file:
                        dataframes["patient_history"] = pd.read_csv(file)
                    elif "patient_lda" in csv_file:
                        dataframes["patient_lda"] = pd.read_csv(file)
                    elif "patient_coding" in csv_file:
                        dataframes["patient_coding"] = pd.read_csv(file)
                    elif "patient_visit" in csv_file:
                        dataframes["patient_visit"] = pd.read_csv(file)
                    else:
                        logger.info(f"Uncategorized file: {csv_file}")
        pbar.update(1)
        # Step 1: Patients Filtering
        tqdm.write("Step 1: Patient Filtering")
        filtered_dataset = patient_filtering(dataframes)
        pbar.update(1)
        
        # Step 2: Preprocess Patients Labs
        tqdm.write("Step 2: Preprocessing Patients labs")
        filtered_dataset_with_labs =  preprocess_patient_labs(filtered_dataset, dataframes)
        pbar.update(1)

        # Step 3: Preprocess Patient Medications 
        tqdm.write("Step 3: Preprocessing Patient Medications")
        final_dataset_with_medications =  preprocess_patient_medications(filtered_dataset_with_labs, dataframes)
        pbar.update(1)

        # Step 4: Preprocess Patient Procedure Events
        tqdm.write("Step 4: Preprocessing Procedure Events")
        final_dataset_with_event_encoded =  preprocess_patient_events(final_dataset_with_medications, dataframes)
        pbar.update(1)

        # Step 5: Preprocess Patient LDA
        tqdm.write("Step 5: Preprocessing Patient LDA")
        final_dataset_with_lda_encoded = preprocess_patient_lda(final_dataset_with_event_encoded, dataframes)
        pbar.update(1)

        # Step 6: Preprocess Patient History
        tqdm.write("Step 6: Preprocessing Patient History")
        final_df = preprocess_patient_history(final_dataset_with_lda_encoded, dataframes)
        pbar.update(1)

        # Step 7: Add Post Op Complication
        tqdm.write("Step 7: Add Post Op Complication")
        complete_df = add_post_op_complication_label(final_df, dataframes)
        pbar.update(1)

        # Step 8: Cleaning and Handling Missing Values
        tqdm.write("Step 8: Feature Cleaning and Handling Missing Values")
        finalize_feature_table(complete_df, save_path=args.output_dir)
        pbar.update(1)

    tqdm.write("Preprocessing complete.")


if __name__ == "__main__":
    main()