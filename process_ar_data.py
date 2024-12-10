import pandas as pd
import os
import logging
import re
import argparse


# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def find_table_start(sheet_data):
    """
    Find the start of the table by looking for the first row that starts with any letter
    followed by empty cells or numbers
    """
    for idx, row in sheet_data.iterrows():
        first_cell = str(row.iloc[0]).strip()
        # Check if the first cell is a single letter
        if len(first_cell) == 1 and first_cell.isalpha():
            return idx
    return None

def process_all_files_in_folder(resistance_folder, threshold=0.1):
    output_folder = os.path.join(resistance_folder, "processed_data")
    os.makedirs(output_folder, exist_ok=True)

    # Get all Excel files in the folder
    files = [file for file in os.listdir(resistance_folder) if file.endswith('.xlsx')]
    if not files:
        logging.info("No Excel files found in the folder.")
        return

    for file in files:
        input_file_path = os.path.join(resistance_folder, file)
        logging.info(f"Processing file: {input_file_path}")

        # Validate the file
        try:
            data = pd.read_excel(input_file_path, sheet_name=None, engine='openpyxl')
        except Exception as e:
            logging.error(f"Error reading {input_file_path}: {e}")
            continue

        for sheet_name, sheet_data in data.items():
            logging.info(f"Processing sheet: {sheet_name}")
            
            # Find the start of the table
            start_row = find_table_start(sheet_data)
            
            if start_row is None:
                logging.warning(f"No valid table start found in sheet {sheet_name}. Skipping...")
                continue
                
            logging.info(f"Start row detected at: {start_row}")

            # Extract the table dynamically based on the starting row
            data_table = sheet_data.iloc[start_row:start_row + 9, :]  # Adjusted for 9 rows
            
            # Get original column headers
            column_headers = list(sheet_data.iloc[start_row - 1, :])
            data_table.columns = column_headers

            # Extract the measurement columns (all except first and last columns)
            time_points = column_headers[1:-1]
            plate_data = data_table.iloc[:, 1:-1]
            plate_data.columns = time_points

            # Extract row labels and medium column
            row_labels = data_table.iloc[:, 0].values
            mediums = data_table.iloc[:, -1].values

            # Handle row label mismatches
            if len(row_labels) != plate_data.shape[0]:
                logging.warning(
                    f"Row label mismatch: {len(row_labels)} row labels, {plate_data.shape[0]} data rows."
                )
                if len(row_labels) < plate_data.shape[0]:
                    row_labels = list(row_labels) + [
                        f"Row_{i}" for i in range(len(row_labels), plate_data.shape[0])
                    ]
                else:
                    row_labels = row_labels[:plate_data.shape[0]]

            # Process each strain (row in the table)
            results = []
            for idx, (_, row) in enumerate(plate_data.iterrows()):
                row_data = pd.to_numeric(row, errors='coerce')
                
                result_dict = {
                    'Row': row_labels[idx],
                    'Medium': mediums[idx]
                }
                
                # Add all individual measurements
                for col, value in row_data.items():
                    result_dict[col] = value
                    
                # Add mean and standard deviation
                result_dict['Mean'] = row_data.mean()
                result_dict['Stdev'] = row_data.std()
                
                # Add resistance determination (Yes/No)
                for col, value in row_data.items():
                    resistance_col = f"{col}_Resistance"
                    if pd.isna(value):
                        result_dict[resistance_col] = 'NA'
                    else:
                        result_dict[resistance_col] = 'Yes' if value >= threshold else 'No'
                
                results.append(result_dict)

            # Create DataFrame and organize columns
            results_df = pd.DataFrame(results)
            
            # Organize columns in the desired order
            measurement_cols = time_points
            resistance_cols = [f"{col}_Resistance" for col in time_points]
            final_cols = ['Row', 'Medium'] + list(measurement_cols) + resistance_cols + ['Mean', 'Stdev']
            results_df = results_df[final_cols]

            # Save results
            clean_sheet_name = re.sub(r'[\\/*?:"<>|]', "", sheet_name)
            processed_file_name = f"{os.path.splitext(file)[0]}_processed_{clean_sheet_name}.xlsx"
            processed_file_path = os.path.join(output_folder, processed_file_name)
            results_df.to_excel(processed_file_path, index=False)
            logging.info(f"Results saved to {processed_file_path}")

if __name__ == "__main__":

    # Set up argument parser
    parser = argparse.ArgumentParser(description="Process all Excel files in a folder for antibiotic resistance data.")
    parser.add_argument("resistance_folder", help="Path to the folder containing Excel files.")
    parser.add_argument("--threshold", type=float, default=0.1, help="Threshold for resistance determination (default: 0.1).")
    
    args = parser.parse_args()
    
    # Call the processing function with arguments
    process_all_files_in_folder(args.resistance_folder, threshold=args.threshold)