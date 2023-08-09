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
# from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import LSTM, Dense
from keras.initializers import HeNormal
from keras.layers import BatchNormalization
from sklearn.preprocessing import StandardScaler
from keras.layers import Dropout
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from matplotlib import colors

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

print("Finish loading data")

# Convert the Xarray datasets to Pandas DataFrames
response_data = response.to_dataframe().reset_index()
dynamic_data = dynvars.to_dataframe().reset_index()
static_data = sttvars.to_dataframe().reset_index()

# Clear data with NaN value
response_data.dropna(inplace=True)
dynamic_data.dropna(inplace=True)
static_data.dropna(inplace=True)

# Standardize the data
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

# # Splitting the data into train, validation, and test
# train_data = final_data[:int(len(final_data) * 0.7)]
# val_data = final_data[int(len(final_data) * 0.7):int(len(final_data) * 0.85)]
# test_data = final_data[int(len(final_data) * 0.85):]

# Splitting the data into train, validation, and test
train_data, temp_data = train_test_split(final_data, test_size=0.2, shuffle=False)
test_data, val_data = train_test_split(final_data, test_size=0.5, shuffle=False)

print("Finish splitting data")

# Define a function to create sequences
def create_sequences(data, sequence_length):
    data_values = data.drop(columns=['time', 'cat']).values
    sequences = []
    for i in range(0, len(data_values) - sequence_length, sequence_length):
        seq = data_values[i:i+sequence_length]
        sequences.append(seq)
    return np.array(sequences)

# Sequence length
sequence_length = 30

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
# First LSTM layer
model.add(LSTM(50, activation='relu', return_sequences=True, input_shape=(sequence_length-1, X_train.shape[-1])))
model.add(Dropout(0.2))
# Second LSTM layer
model.add(LSTM(50, activation='relu'))
model.add(Dropout(0.2))
# Two dense layers
model.add(Dense(20, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(2))

# model.add(LSTM(50, activation='relu', input_shape=(sequence_length-1, X_train.shape[-1])))
# model.add(Dense(20, activation='relu'))
# model.add(Dense(2)) # Predicting d_h and d_v

# Compile the model
# model.compile(optimizer='adam', loss='mse')
model.compile(optimizer='adam', loss='mse', metrics=['mae'])

print("Finish defining the model")

# Train the model
model.fit(X_train, y_train, epochs=100, batch_size=64, validation_data=(X_val, y_val))

print("Finish training")

# Making predictions for the next sequence
# last_sequence = final_data.groupby('cat').tail(sequence_length-1).drop(columns=['time', 'cat']).values.reshape(len(final_data['cat'].unique()), sequence_length-1, -1)
# y_pred = model.predict(last_sequence)
y_pred = model.predict(X_test)

print("Finish prediction")

# Reshape y_test and y_pred to 1D arrays
y_test_dh = y_test[:, 0].flatten()
y_pred_dh = y_pred[:, 0].flatten()

y_test_dv = y_test[:, 1].flatten()
y_pred_dv = y_pred[:, 1].flatten()

# Compute correlation coefficients
corr_dh = np.corrcoef(y_test_dh, y_pred_dh)[0, 1]
corr_dv = np.corrcoef(y_test_dv, y_pred_dv)[0, 1]

print("Correlation coefficient for d_h: ", corr_dh)
print("Correlation coefficient for d_v: ", corr_dv)

# Get 'd_v' predictions
d_v_predictions = y_pred[:, 1]

# Compute the average rate of vertical deformation
avg_rate_of_deformation = np.mean(d_v_predictions)

print("Average rate of vertical deformation: ", avg_rate_of_deformation)

# Evaluate the model
mse, mae = model.evaluate(X_test, y_test, verbose=0)

# calculate the RMSE
rmse = np.sqrt(mse)

print('Root Mean Squared Error on test set: ', rmse)
print('Mean Absolute Error on test set: ', mae)

# Fit linear regression for both components
lr_d_h = LinearRegression()
lr_d_v = LinearRegression()

lr_d_h.fit(y_test[:, 0].reshape(-1, 1), y_pred[:, 0])
lr_d_v.fit(y_test[:, 1].reshape(-1, 1), y_pred[:, 1])

# Regression plots
fig, ax = plt.subplots(1, 2, figsize=(14, 7))

sns.regplot(x=y_test[:, 0], y=y_pred[:, 0], ax=ax[0], color='b', line_kws={'color':'r'})
ax[0].set_title('Regression Plot for d_h')
ax[0].set_xlabel('Observed')
ax[0].set_ylabel('Predicted')

sns.regplot(x=y_test[:, 1], y=y_pred[:, 1], ax=ax[1], color='b', line_kws={'color':'r'})
ax[1].set_title('Regression Plot for d_v')
ax[1].set_xlabel('Observed')
ax[1].set_ylabel('Predicted')

plt.tight_layout()
plt.show()

# Merge the observed and predicted values with the GeoDataFrame
# For the purpose of this demonstration, let's assume the last N rows of `su` correspond to the `y_test` values.
N = y_test.shape[0]
su_subset = su.tail(N).copy()
su_subset['obs_d_h'] = y_test[:, 0]
su_subset['pred_d_h'] = y_pred[:, 0]
su_subset['obs_d_v'] = y_test[:, 1]
su_subset['pred_d_v'] = y_pred[:, 1]

# Plotting
fig, axes = plt.subplots(2, 2, figsize=(20, 20))

# Observed d_h map
su_subset.to_crs(epsg=3857).plot(column='obs_d_h', ax=axes[0, 0], alpha=0.5, edgecolor="k", legend=True)
axes[0, 0].set_title("Observed d_h")
cx.add_basemap(axes[0, 0], source=cx.providers.CartoDB.Positron)

# Predicted d_h map
su_subset.to_crs(epsg=3857).plot(column='pred_d_h', ax=axes[0, 1], alpha=0.5, edgecolor="k", legend=True)
axes[0, 1].set_title("Predicted d_h")
cx.add_basemap(axes[0, 1], source=cx.providers.CartoDB.Positron)

# Observed d_v map
su_subset.to_crs(epsg=3857).plot(column='obs_d_v', ax=axes[1, 0], alpha=0.5, edgecolor="k", legend=True)
axes[1, 0].set_title("Observed d_v")
cx.add_basemap(axes[1, 0], source=cx.providers.CartoDB.Positron)

# Predicted d_v map
su_subset.to_crs(epsg=3857).plot(column='pred_d_v', ax=axes[1, 1], alpha=0.5, edgecolor="k", legend=True)
axes[1, 1].set_title("Predicted d_v")
cx.add_basemap(axes[1, 1], source=cx.providers.CartoDB.Positron)

plt.tight_layout()
plt.show()
