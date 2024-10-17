# Fetch Monthly Receipt Predictor
An Application where given the number of the observed scanned receipts each day for a year, it will give predictions of the receipts next year.
## Getting Started
- Clone the repository:
  ```
  git clone https://github.com/yeedrag/FetchApp.git
  cd FetchApp
  ```
- Build the Docker image:
  ```
  docker build . -t fetchapp
  ```
- Run the Docker container:
  ```
  docker run -p 6969:6969 fetchapp
  ```
## Generating Predictions
The Code supports two ways of generating predictions: Using Streamlit Web App or Interactively via the command line.

The Streamlit Web App enables visualization and more user-friendly infercace, while the command line provides more flexibility.

### Generating via the Streamlit Web App
If you want interact with the Web Applplication, run the docker container as such:
```
docker run -p 6969:6969 fetch-rewards-app
```
Then, open your browzer and navigate to ```http://localhost:6969```

Inside the interface, you will first be required upload the data from last year.

Then, you can choose to either retrain a model with your chosen parameters or predict from an existing checkpoint file.


### Generating via the command-line interface 
If you want interact with the command-line interface, run the docker container as such:
```
docker run -it fetchapp /bin/bash
```
In order to train the model, you can run the following command:
```
python train.py data/data_daily.csv
```
This will train a basic LSTM model with 30 hidden size and 100 epoch.

There are three flags you can set in training, ```--model``` ```--hidden_size``` ```--epochs```:
```
python train.py --model {LSTM,MLP} [--hidden_size HIDDEN_SIZE] [--epochs EPOCHS] data_path
```
The checkpoints are stored inside ```./checkpoint``` in the format of ```{model}_hidden_{hidden_size}_epochs_{epochs}.pth```

To generate a prediction, run the following command:
```
python predict.py data/data_daily.csv --checkpoint checkpoints/ --year 2022
```
The ```--year``` flag is used to format the output excel file.

The output predictions are stored in ```predictions/{model}/{model}_hidden_{hidden_size}_epochs_{epochs}_predictions.csv```.




