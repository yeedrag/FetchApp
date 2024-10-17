import argparse
import torch
import os
import torch.nn as nn
from models.LSTM_Model import LSTM_Model
from models.MLP_Model import MLP_Model
from data.prepare_data import PrepareDataset
device = "cuda" if torch.cuda.is_available() else "cpu"
def train_model(args):
    loader, _, _ = PrepareDataset(args.data_path)
    model = None
    if args.model == "LSTM":
        model = LSTM_Model(1, args.hidden_size).to(device)
    elif args.model == "MLP":
        model = MLP_Model(1, args.hidden_size).to(device)
    else:
        #TODO: Implement more models?
        pass

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    # Training loop
    for epoch in range(args.epochs):
        for x, y, idx in loader:
            # Forward pass
            if args.model == "MLP":
                idx = idx.unsqueeze(1) # (batch_size, 1)
                output = model(idx) # use the index to predict
                y = y[:,-1,:] # last element
            else:
                output, _, _ = model(x)
            loss = criterion(output, y)

            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        print(f'Epoch {epoch + 1}/{args.epochs}, Loss: {loss.item():.4f}')

    # Save the trained model
    file_name = os.path.join('checkpoints', f'{args.model}_hidden_{args.hidden_size}_epochs_{args.epochs}.pth')
    torch.save(model.state_dict(), file_name)
    print("Training complete, model saved!")
    return file_name
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('data_path', type=str, help='Path to the data file')
    parser.add_argument(
        '--model', 
        choices=['LSTM', 'MLP'], 
        required=True, 
        default='MLP,
        help='Which model to use: LSTM or MLP'
    )
    parser.add_argument(
        '--hidden_size', 
        type=int, 
        default=30,  
        help='Hidden size for the model (if applicable, default: 30)'
    )
    parser.add_argument(
        '--epochs', 
        type=int, 
        default=100,  
        help='How many epochs to train the model with (default: 100)'
    )

    args = parser.parse_args()
    train_model(args)

if __name__ == "__main__":
    main()