import pandas as pd
import os

def print_column_min_max(file_path, column_indices):
    """
    Print minimum and maximum values of specified columns from a CSV or TSV file.
    Also print percentage of values based on angle ranges.
    
    Parameters:
    file_path (str): Path to the CSV or TSV file
    column_indices (list): List of column indices to analyze (0-based)
    """
    try:
        # Check if file exists
        if not os.path.exists(file_path):
            print(f"Error: File '{file_path}' not found!")
            return
        
        # Determine file type and read accordingly
        if file_path.lower().endswith('.tsv'):
            df = pd.read_csv(file_path, delimiter="\t", header=None)
            file_type = "TSV"
        else:
            df = pd.read_csv(file_path)
            file_type = "CSV"
        
        # Get column names
        columns = df.columns.tolist()
        
        print(f"File: {file_path} ({file_type})")
        print(f"Number of rows: {len(df)}")
        print(f"Number of columns: {len(columns)}")
        print("-" * 60)
        
        # Analyze each specified column
        for col_idx in column_indices:
            if col_idx < 0 or col_idx >= len(columns):
                print(f"Error: Column index {col_idx} is out of range. File has {len(columns)} columns.")
                continue
            
            col_name = columns[col_idx]
            col_data = df.iloc[:, col_idx]
            
            min_val = col_data.min()
            max_val = col_data.max()
            
            # Print min and max in one line with 2 decimal places
            print(f"Column {col_idx} ('{col_name}'): Min = {min_val:.2f}, Max = {max_val:.2f}")
            
            # Calculate percentage based on angle range
            total_values = len(col_data)
                      
            # Check if data is in range 0 to 180
            if min_val >= 0 and max_val <= 180:
                above_90_count = len(col_data[col_data > 90])
                above_90_percentage = (above_90_count / total_values) * 100
                print(f"  Percentage of values > 90: {above_90_percentage:.2f}%")
            # Check if data is in range -180 to 180
            elif min_val >= -180 and max_val <= 180:
                positive_count = len(col_data[col_data > 0])
                positive_percentage = (positive_count / total_values) * 100
                print(f"  Percentage of values > 0: {positive_percentage:.2f}%")
            
            print()  # Empty line for better readability
            
    except Exception as e:
        print(f"Error processing file '{file_path}': {e}")

# Example usage
if __name__ == "__main__":
    # Example 1: Analyze CSV file
    file_path_csv = "./SCOP/challenging/data.csv"
    print_column_min_max(file_path_csv, [7, 8])
    
    print("\n" + "=" * 60 + "\n")
    
    # Example 2: Analyze TSV file
    file_path_tsv = "./SCOP/challenging/data.tsv"
    print_column_min_max(file_path_tsv, [7, 8])
    
    print("\n" + "=" * 60 + "\n")
    
    # Example 3: Analyze all files in folders
    folders = [
        './SCOP/challenging',
        './SCOP/easy', 
        './SCOP/hard',
        './SCOP/moderate'
    ]
    
    for folder in folders:
        # Try both CSV and TSV files
        for extension in ['.csv', '.tsv']:
            file_path = os.path.join(folder, f'data{extension}')
            if os.path.exists(file_path):
                print(f"\nAnalyzing: {file_path}")
                print_column_min_max(file_path, [7, 8])
                print("-" * 40)
                break