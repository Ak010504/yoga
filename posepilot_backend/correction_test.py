# correction_test.py

from correction_predict import predict_correction_from_dataframe # Import the DataFrame function
import pandas as pd

def main():
    # --- Configure ---
    csv_path = "D:/PosePilot/data/warrior/P7.csv"
    pose = "warrior"
    # --- End Configuration ---

    print(f"Testing correction for {pose} from {csv_path}")

    try:
        # Load the CSV file
        data = pd.read_csv(csv_path)
        print(f"Loaded data with shape: {data.shape}")

        # --- Handle the 'asana' column ---
        if 'asana' in data.columns:
            print("Found 'asana' column. Dropping it before passing to prediction function.")
            data = data.drop(columns=['asana'])
            print(f"Shape after dropping 'asana': {data.shape}")

        # Call the function that accepts a DataFrame
        # This function will handle the internal processing (structure_data, etc.)
        result = predict_correction_from_dataframe(data, pose)

        if result and result.get("status") == "success":
            print("Test completed successfully.")
        else:
            print("Test failed.")
            if result:
                print(f"Error details: {result.get('error', 'N/A')}")

    except FileNotFoundError:
        print(f"Error: CSV file not found: {csv_path}")
    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    main()
