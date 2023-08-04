import pandas as pd
import geopandas as gpd
import numpy as np
import xarray as xr
import os
import rioxarray
from tqdm.notebook import tqdm
from pathlib import Path
import rasterio
import contextily as cx
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import LSTM, Dense
from keras.initializers import HeNormal
from keras.layers import BatchNormalization


epsg = 3035

# Check if GPU is available
if tf.config.list_physical_devices('GPU'):
    print("GPU is available")
else:
    print("Training on CPU")

# Load Response Variable
response = xr.open_dataset('small/insxr_small.nc')  # Y

# Load Dynamic Variable
dynvars = xr.open_dataset('small/dynvars_small.nc')  # X_t

# Load Static Variable
sttvars = xr.open_dataset('small/sttvars_small.nc')  # X_s

# load Geospatial Reference
su = gpd.read_file('small/su_filtered_small.gpkg').to_crs(epsg=epsg)

# Convert the Xarray datasets to Pandas DataFrames
response_data = response.to_dataframe().reset_index()
dynamic_data = dynvars.to_dataframe().reset_index()
static_data = sttvars.to_dataframe().reset_index()

response_data.dropna(inplace=True)
dynamic_data.dropna(inplace=True)
static_data.dropna(inplace=True)

# Standardize the data
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
response_data[['d_h', 'd_v']] = scaler.fit_transform(response_data[['d_h', 'd_v']])
dynamic_data[['lai', 'precp', 'temp', 'twss']] = scaler.fit_transform(dynamic_data[['lai', 'precp', 'temp', 'twss']])
static_data[['Slope', 'Elevation', 'tanurve', 'Aspect', 'profCurv', 'ruggedness']] = scaler.fit_transform(static_data[['Slope', 'Elevation', 'tanurve', 'Aspect', 'profCurv', 'ruggedness']])

# Merge the data using 'cat' as the primary key
final_data = pd.merge(response_data, dynamic_data, on=['time', 'cat'])
final_data = pd.merge(final_data, static_data, on=['cat'])

# Sorting by time and cat
final_data = final_data.sort_values(by=['time', 'cat'])

print("Finish prepocessing data")

# Sequence length
sequence_length = 30

# Splitting the data into train, validation, and test
train_data = final_data[:int(len(final_data) * 0.7)]
val_data = final_data[int(len(final_data) * 0.7):int(len(final_data) * 0.85)]
test_data = final_data[int(len(final_data) * 0.85):]

print("Finish splitting data")

# Define a function to create sequences
def create_sequences(data, sequence_length):
    data_values = data.drop(columns=['time', 'cat']).values
    sequences = []
    for i in range(0, len(data_values) - sequence_length, sequence_length):
        seq = data_values[i:i+sequence_length]
        sequences.append(seq)
    return np.array(sequences)

# Create sequences for training, validation, and testing
X_train = create_sequences(train_data, sequence_length)[:, :-1, :]
y_train = create_sequences(train_data, sequence_length)[:, -1, :2]
X_val = create_sequences(val_data, sequence_length)[:, :-1, :]
y_val = create_sequences(val_data, sequence_length)[:, -1, :2]
X_test = create_sequences(test_data, sequence_length)[:, :-1, :]
y_test = create_sequences(test_data, sequence_length)[:, -1, :2]

print("Finish creating sequences")

# Build the LSTM model
model = Sequential()
model.add(LSTM(50, activation='relu', input_shape=(sequence_length-1, X_train.shape[-1])))
model.add(Dense(20, activation='relu'))
model.add(Dense(2)) # Predicting d_h and d_v

# Compile the model
model.compile(optimizer='adam', loss='mse')

print("Finish defining the model")

# Train the model
model.fit(X_train, y_train, epochs=100, batch_size=64, validation_data=(X_val, y_val))

print("Finish training")

# Making predictions for the next sequence
last_sequence = final_data.groupby('cat').tail(sequence_length-1).drop(columns=['time', 'cat']).values.reshape(len(final_data['cat'].unique()), sequence_length-1, -1)
next_prediction = model.predict(last_sequence)

print("Next sequence prediction: ")
print(next_prediction)
print("Finish prediction")

print(model.summary)
