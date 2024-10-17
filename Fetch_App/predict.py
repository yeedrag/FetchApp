import argparse
import torch
import pandas as pd
import re
import os
import torch.nn as nn
import calendar
from models.LSTM_Model import LSTM_Model
from models.MLP_Model import MLP_Model
from data.prepare_data import PrepareDataset
device = "cuda" if torch.cuda.is_available() else "cpu"
def generate_prediction(args):
    _, scaled_data, scales = PrepareDataset(args.data_path, 1, 1)
    scaled_data = scaled_data[-60:] # use same as sequence size
    scaled_data = scaled_data.unsqueeze(0)
    match = re.search(r'(\w+)_hidden_(\d+)_epochs_(\d+)', args.checkpoint)
    model = None

    if match.group(1) == 'LSTM':
        model = LSTM_Model(1, int(match.group(2))).to(device)
    elif match.group(1) == 'MLP':
        model = MLP_Model(60, int(match.group(2))).to(device)

    model.load_state_dict(torch.load(args.checkpoint))
    model.eval()

    # First run through the whole file to get H and C
    days = 366 if calendar.isleap(args.year) else 365
    H = C = None
    out = []
    X_max, X_min = scales
    with torch.no_grad():
        if match.group(1) == 'LSTM':
            # First run through the whole file to get H and C
            pred, H_new, C_new = model(scaled_data, H, C)
            inp = pred[0, -1]
            inp = inp.unsqueeze(0)
            inp = inp.unsqueeze(0)
            H = H_new
            C = C_new
            # Run predictions
            for i in range(days):
                pred, H_new, C_new = model(inp, H, C)
                out.extend((pred[0,-1].item() * (X_max - X_min)) + X_min) # Scale back
                inp = pred[0, -1]
                inp = inp.unsqueeze(0)
                inp = inp.unsqueeze(0)
                H = H_new
                C = C_new
        else:
            for i in range(days):
                index = torch.tensor(i + 365, dtype=torch.float32)
                index = index.unsqueeze(0).unsqueeze(0)
                pred = model(index)
                out.extend((pred[0,-1].item() * (X_max - X_min)) + X_min) # Scale back        

    date_range = pd.date_range(start=f"{args.year}-01-01", end=f"{args.year}-12-31")

    predictions_df = pd.DataFrame({
        '# Date': date_range.strftime('%Y-%m-%d'),  # Format date as YYYY-MM-DD
        'Receipt_Count': out
    })

    # Save the DataFrame to a CSV file
    folder_path = os.path.join('predictions', match.group(1))
    os.makedirs(folder_path, exist_ok=True)
    #TODO: this is ugly :(
    file_name = f"{match.group(1)}_hidden_{match.group(2)}_epochs_{match.group(3)}_{args.year}_predictions.csv"
    output_file = os.path.join(folder_path, file_name)
    predictions_df.to_csv(output_file, index=False)
    print(f"Predictions file saved as {output_file}")
    return predictions_df, file_name

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('data_path', type=str, help='Path to the previous year data file')
    parser.add_argument(
        '--checkpoint',
        required=True,
        help='Which checkpoint is used for prediction'        
    )
    parser.add_argument(
        '--year',
        type=int, 
        default=2022, 
        help='The date to be registered in the predictions'        
    )
    args = parser.parse_args()
    generate_prediction(args)
if __name__ == "__main__":
    main()