# QSTGNN
This is a PyTorch implementation of the paper: [QSTGCN: Quaternion Spatio-temporal Graph Neural Networks].

## Requirements
The model is implemented using Python3 with dependencies specified in requirements.txt
## Data Preparation
### Multivariate time series datasets

### Download datasets

Download the PEMS-BAY dataset from [Baidu Yun](https://pan.baidu.com/s/14Yy9isAIZYdU__OYEQGa_g) provided by [Li et al.](https://github.com/liyaguang/DCRNN.git) . Move them into the data folder. Download the AQI dataset from (http://research.microsoft.com/apps/pubs/?id=246398) and PV-US datasets from (https://www.nrel.gov/grid/solar-power-data.html)

# example
We present the PEMSD7(M) data set in the data folder

# Create data directories
For example, for PEMS-BAY dataset. You can also modify the code according to its specific time series.
python generate_training_data.py --output_dir=data/PEMS-BAY --traffic_df_filename=data/PEMS-BAY.h5

## Model Training
### run code
Run the following file directly: main_QSTGNN.py

## Code reference
https://github.com/nnzhan/MTGNN
https://github.com/daiquocnguyen/QGNN
https://github.com/Orkis-Research/Pytorch-Quaternion-Neural-Networks
https://github.com/JorisWeeda/Quaternion-Convolutional-Neural-Networks

