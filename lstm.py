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
import tensorflow as tf
from tensorflow.keras.layers import LSTM, Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Nadam
from tensorflow.keras.callbacks import ReduceLROnPlateau
from tensorflow.keras.models import Sequential
import pywt

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
response_data.fillna(response_data.median(numeric_only=True), inplace=True)
dynamic_data.fillna(dynamic_data.median(numeric_only=True), inplace=True)
static_data.fillna(static_data.median(numeric_only=True), inplace=True)

# Remove useless data
date_to_remove = ['2016-12-30', '2017-12-31', '2019-12-27']
response_data = response_data[~response_data['time'].isin(date_to_remove)]

# Merge the data using 'cat' as the primary key
final_data = pd.merge(response_data, dynamic_data, on=['time', 'cat'])
final_data = pd.merge(final_data, static_data, on=['cat'])

# Sorting by time and cat
final_data = final_data.sort_values(by=['time', 'cat'])

# Splitting the data into train, validation, and test
train_data, temp_data = train_test_split(final_data, test_size=1/3, shuffle=False)
test_data, val_data = train_test_split(temp_data, test_size=1/2, shuffle=False)

# Standardize the train data
scaler_response = StandardScaler().fit(train_data[['d_h', 'd_v']])
train_data[['d_h', 'd_v']] = scaler_response.transform(train_data[['d_h', 'd_v']])

scaler_dynamic = StandardScaler().fit(train_data[['lai', 'precp', 'temp', 'twss']])
train_data[['lai', 'precp', 'temp', 'twss']] = scaler_dynamic.transform(train_data[['lai', 'precp', 'temp', 'twss']])

scaler_static = StandardScaler().fit(train_data[['Slope', 'Elevation', 'tanurve', 'Aspect', 'profCurv', 'ruggedness']])
train_data[['Slope', 'Elevation', 'tanurve', 'Aspect', 'profCurv', 'ruggedness']] = scaler_static.transform(train_data[['Slope', 'Elevation', 'tanurve', 'Aspect', 'profCurv', 'ruggedness']])

# Apply the transformations to the validation and test data using scalers trained on the train data
val_data[['d_h', 'd_v']] = scaler_response.transform(val_data[['d_h', 'd_v']])
test_data[['d_h', 'd_v']] = scaler_response.transform(test_data[['d_h', 'd_v']])

val_data[['lai', 'precp', 'temp', 'twss']] = scaler_dynamic.transform(val_data[['lai', 'precp', 'temp', 'twss']])
test_data[['lai', 'precp', 'temp', 'twss']] = scaler_dynamic.transform(test_data[['lai', 'precp', 'temp', 'twss']])

val_data[['Slope', 'Elevation', 'tanurve', 'Aspect', 'profCurv', 'ruggedness']] = scaler_static.transform(val_data[['Slope', 'Elevation', 'tanurve', 'Aspect', 'profCurv', 'ruggedness']])
test_data[['Slope', 'Elevation', 'tanurve', 'Aspect', 'profCurv', 'ruggedness']] = scaler_static.transform(test_data[['Slope', 'Elevation', 'tanurve', 'Aspect', 'profCurv', 'ruggedness']])

print("Finish preprocessing data")

sequence_length = 30

def create_sequences(data, sequence_length):
    x, y = [], []

    for cat in data['cat'].unique():
        cat_data = data[data['cat'] == cat]
        for i in range(len(cat_data) - sequence_length):
            x.append(cat_data.iloc[i:i+sequence_length].drop(columns=['d_h', 'd_v', 'time', 'cat']).values)
            y.append(cat_data.iloc[i+sequence_length][['d_h', 'd_v']].values)

    return np.array(x, dtype=np.float32), np.array(y, dtype=np.float32)

X_train, y_train = create_sequences(train_data, sequence_length)
X_val, y_val = create_sequences(val_data, sequence_length)
X_test, y_test = create_sequences(test_data, sequence_length)

print("Finish creating sequences")

number_of_features = X_train.shape[2]

model = Sequential()

# First LSTM layer
model.add(LSTM(64, activation='tanh', recurrent_activation='sigmoid', recurrent_dropout=0, unroll=False, use_bias=True, 
               return_sequences=True,   # This argument ensures the output is a sequence
               input_shape=(X_train.shape[1], number_of_features)))

model.add(Dropout(0.2))
model.add(BatchNormalization())

# Second LSTM layer
model.add(LSTM(64, activation='tanh', recurrent_activation='sigmoid', recurrent_dropout=0, unroll=False, use_bias=True))
model.add(Dropout(0.2))
model.add(BatchNormalization())

# Dense layers
model.add(Dense(32, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(2))

optimizer = Nadam(learning_rate=0.002)
model.compile(optimizer=optimizer, loss='mse', metrics=['mae'])

# Consider using the ReduceLROnPlateau callback for dynamic learning rate adjustments
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, min_lr=0.0001)

with tf.device('/cpu:0'):

    # Train the model
    history = model.fit(X_train, y_train, epochs=50, batch_size=64, validation_data=(X_val, y_val), callbacks=[reduce_lr])

print("Finish training")

# Save the model and history
model.save('my_model_2.h5')

# Convert the history.history dict to a DataFrame
history_df = pd.DataFrame(history.history)
history_df.to_csv('training_history.csv', index=False)


y_pred = model.predict(X_test)

print("Finish predicting")

# Fit linear regression for both components
lr_d_h = LinearRegression()
lr_d_v = LinearRegression()

lr_d_h.fit(y_test_ori[:, 0].reshape(-1, 1), y_pred_ori[:, 0])
lr_d_v.fit(y_test_ori[:, 1].reshape(-1, 1), y_pred_ori[:, 1])

# Regression plots
fig, ax = plt.subplots(1, 2, figsize=(14, 7))

sns.regplot(x=y_test_ori[:, 0], y=y_pred_ori[:, 0], ax=ax[0], color='b', line_kws={'color':'r'})
ax[0].set_title('Regression Plot for d_h')
ax[0].set_xlabel('Observed')
ax[0].set_ylabel('Predicted')

sns.regplot(x=y_test_ori[:, 1], y=y_pred_ori[:, 1], ax=ax[1], color='b', line_kws={'color':'r'})
ax[1].set_title('Regression Plot for d_v')
ax[1].set_xlabel('Observed')
ax[1].set_ylabel('Predicted')

plt.tight_layout()
plt.show()

# Reshape y_test and y_pred to 1D arrays
y_test_dh = y_test_ori[:, 0].flatten()
y_pred_dh = y_pred_ori[:, 0].flatten()

y_test_dv = y_test_ori[:, 1].flatten()
y_pred_dv = y_pred_ori[:, 1].flatten()

# Compute correlation coefficients
corr_dh = np.corrcoef(y_test_dh, y_pred_dh)[0, 1]
corr_dv = np.corrcoef(y_test_dv, y_pred_dv)[0, 1]

print("Correlation coefficient for d_h: ", corr_dh)
print("Correlation coefficient for d_v: ", corr_dv)

# Evaluate the model
mse, mae = model.evaluate(X_test, y_test, verbose=0)

# calculate the RMSE
rmse = np.sqrt(mse)

print('Root Mean Squared Error on test set: ', rmse)
print('Mean Absolute Error on test set: ', mae)

# Reshape the arrays
y_test_reshaped = y_test_ori.reshape(6352, 30, 2)
y_pred_reshaped = y_pred_ori.reshape(6352, 30, 2)

# Sum along the sequence length axis
y_test_summed = y_test_reshaped.sum(axis=1)
y_pred_summed = y_pred_reshaped.sum(axis=1)

# Merge the observed and predicted values with the GeoDataFrame
# For the purpose of this demonstration, let's assume the last N rows of `su` correspond to the `y_test` values.
N = y_test_summed.shape[0]
su_subset = su.tail(N).copy()
su_subset['obs_d_h'] = y_test_summed[:, 0]
su_subset['pred_d_h'] = y_pred_summed[:, 0]
su_subset['obs_d_v'] = y_test_summed[:, 1]
su_subset['pred_d_v'] = y_pred_summed[:, 1]

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

# Extract d_h and d_v for a single 'cat' for simplification
sample_data = final_data[final_data['cat'] == final_data['cat'].iloc[5]].copy()
sample_data.sort_values(by=['time'], inplace=True)

def plot_wavelet_transform(data, title):
    # Continuous Wavelet Transform
    widths = np.arange(1, 61)  # considering sequence lengths from 1 to 60
    cwtmatr, freqs = pywt.cwt(data, widths, 'morl')

    # Plot
    plt.figure(figsize=(12, 4))
    plt.imshow(np.abs(cwtmatr), aspect='auto', extent=[0, len(data), 1, 60], cmap='jet')
    plt.colorbar(label='Magnitude')
    plt.ylabel('Sequence Length')
    plt.title(f'Wavelet Transform of {title}')
    plt.show()

# Apply wavelet transform to d_h and d_v
plot_wavelet_transform(sample_data['d_h'], 'd_h')
plot_wavelet_transform(sample_data['d_v'], 'd_v')

# Reshape the arrays
y_test_reshaped = y_test_ori.reshape(6352, 30, 2)
y_pred_reshaped = y_pred_ori.reshape(6352, 30, 2)

# Sum along the sequence length axis
y_test_summed = y_test_reshaped.sum(axis=1)
y_pred_summed = y_pred_reshaped.sum(axis=1)

# Merge the observed and predicted values with the GeoDataFrame
# For the purpose of this demonstration, let's assume the last N rows of `su` correspond to the `y_test` values.
N = y_test_summed.shape[0]
su_subset = su.tail(N).copy()
su_subset['obs_d_h'] = y_test_summed[:, 0]
su_subset['pred_d_h'] = y_pred_summed[:, 0]
su_subset['obs_d_v'] = y_test_summed[:, 1]
su_subset['pred_d_v'] = y_pred_summed[:, 1]

# Plotting
fig, axes = plt.subplots(2, 2, figsize=(20, 20))
vmin_value_dh = -100
vmax_value_dh = 100

vmin_value_dv= -400
vmax_value_dv= 0

# Observed d_h map
su_subset.to_crs(epsg=3857).plot(column='obs_d_h', ax=axes[0, 0], alpha=0.5, edgecolor="k", legend=True, vmin=vmin_value_dh, vmax=vmax_value_dh)
axes[0, 0].set_title("Observed sum(d_h)")
cx.add_basemap(axes[0, 0], source=cx.providers.CartoDB.Positron)

# Predicted d_h map
su_subset.to_crs(epsg=3857).plot(column='pred_d_h', ax=axes[0, 1], alpha=0.5, edgecolor="k", legend=True, vmin=vmin_value, vmax=vmax_value)
axes[0, 1].set_title("Predicted sum(d_h)")
cx.add_basemap(axes[0, 1], source=cx.providers.CartoDB.Positron)

# Observed d_v map
su_subset.to_crs(epsg=3857).plot(column='obs_d_v', ax=axes[1, 0], alpha=0.5, edgecolor="k", legend=True, vmin=vmin_value_dv, vmax=vmax_value_dv)
axes[1, 0].set_title("Observed sum(d_v)")
cx.add_basemap(axes[1, 0], source=cx.providers.CartoDB.Positron)

# Predicted d_v map
su_subset.to_crs(epsg=3857).plot(column='pred_d_v', ax=axes[1, 1], alpha=0.5, edgecolor="k", legend=True, vmin=vmin_value_dv, vmax=vmax_value_dv)
axes[1, 1].set_title("Predicted sum(d_v)")
cx.add_basemap(axes[1, 1], source=cx.providers.CartoDB.Positron)

plt.tight_layout()
plt.show()
