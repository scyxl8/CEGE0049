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

epsg = 3035

# Load Response Variable
response = xr.open_dataset('small/insxr_small.nc')  # Y

# Load Dynamic Variable
dynvars = xr.open_dataset('small/dynvars_small.nc')  # X_t

# Load Static Variable
sttvars = xr.open_dataset('small/sttvars_small.nc')  # X_s

# load Geospatial Reference
su = gpd.read_file('small/su_filtered_small.gpkg').to_crs(epsg=epsg)

# Convert xarray datasets to pandas DataFrames
response_df = response.to_dataframe().reset_index()
dynvars_df = dynvars.to_dataframe().reset_index()
sttvars_df = sttvars.to_dataframe().reset_index()

# Function to create data batches
def create_data_batches(X_t, X_s, Y, sequence_length, batch_size):
    data = []
    target = []
    for i in range(len(Y) - sequence_length + 1):
        dynamic_input = X_t[i:i + sequence_length]
        static_input = X_s[i:i + sequence_length]
        # Check for nodata variables in dynamic and static inputs
        if np.isnan(dynamic_input).any() or np.isnan(static_input).any() or np.isnan(Y[i + sequence_length - 1]).any():
            continue  # Skip this data point if it contains nodata variables
        data.append([dynamic_input, static_input])
        target.append(Y[i + sequence_length - 1])
    return data, target

# Hyperparameters
sequence_length = 365
batch_size = 32
learning_rate = 0.0001
epochs = 50

# Prepare data
X_t_values, X_s_values = dynvars_df[['lai', 'precp', 'temp', 'twss']].values, sttvars_df[['Slope', 'Elevation', 'tanurve', 'Aspect', 'profCurv', 'ruggedness']].values
Y_values = response_df[['d_h', 'd_v']].values

# Data normalization
scaler = MinMaxScaler()
X_t_values_normalized = scaler.fit_transform(X_t_values)
X_s_values_normalized = scaler.fit_transform(X_s_values)

# Proceed with data preparation and model training as before using the normalized data
train_data, train_target = create_data_batches(X_t_values_normalized, X_s_values_normalized, Y_values, sequence_length, batch_size)

# Pad or truncate sequences to ensure consistent length
def pad_or_truncate_sequence(sequence, sequence_length):
    if len(sequence) < sequence_length:
        return np.vstack([sequence, np.zeros((sequence_length - len(sequence), sequence.shape[1]))])
    else:
        return sequence[:sequence_length]

train_data_padded = [pad_or_truncate_sequence(dynamic_input, sequence_length) for dynamic_input, static_input in train_data]
train_data_padded_static = [pad_or_truncate_sequence(static_input, sequence_length) for dynamic_input, static_input in train_data]

# Define the LSTM model using the Functional API
num_dynamic_vars = 4
num_static_vars = 6

# Input layers
dynamic_input = tf.keras.layers.Input(shape=(sequence_length, num_dynamic_vars), dtype=tf.float64, name='dynamic_input')
static_input = tf.keras.layers.Input(shape=(sequence_length, num_static_vars), dtype=tf.float64, name='static_input')

# LSTM layers for each input
dynamic_lstm = tf.keras.layers.LSTM(64)(dynamic_input)
static_lstm = tf.keras.layers.LSTM(64)(static_input)

# Concatenate LSTM outputs
combined_lstm = tf.keras.layers.concatenate([dynamic_lstm, static_lstm])

# Additional layers
dense_layer = tf.keras.layers.Dense(32, activation='relu')(combined_lstm)
output_layer = tf.keras.layers.Dense(2, activation='relu', name='output')(dense_layer)  # Output layer with 2 units for d_h and d_v predictions
# output_layer = tf.keras.layers.Dense(2, name='output')(dense_layer)  # Output layer with 2 units for d_h and d_v predictions

# Create the model
model = tf.keras.Model(inputs=[dynamic_input, static_input], outputs=output_layer)

# Compile the model with mean_absolute_error loss
optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
model.compile(optimizer=optimizer, loss='mean_absolute_error')

# Create the generator function
def data_generator():
    for i in range(len(train_data)):
        dynamic_input = tf.convert_to_tensor(train_data_padded[i], dtype=tf.float64)
        static_input = tf.convert_to_tensor(train_data_padded_static[i], dtype=tf.float64)
        target = tf.convert_to_tensor(train_target[i], dtype=tf.float64)
        yield (dynamic_input, static_input), target

# Create the dataset from the generator
train_dataset = tf.data.Dataset.from_generator(data_generator, output_signature=(
    ((tf.TensorSpec(shape=(sequence_length, num_dynamic_vars), dtype=tf.float64),
      tf.TensorSpec(shape=(sequence_length, num_static_vars), dtype=tf.float64)),
    tf.TensorSpec(shape=(2,), dtype=tf.float64))
)).batch(batch_size)

# Train the model
model.fit(train_dataset, epochs=epochs)
