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
response_data.fillna(response_data.median(), inplace=True)
dynamic_data.fillna(dynamic_data.median(), inplace=True)
static_data.fillna(static_data.median(), inplace=True)

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

from keras.layers import LSTM, Dense, Dropout, BatchNormalization
from keras.optimizers import Nadam
from keras.callbacks import ReduceLROnPlateau

number_of_features = X_train.shape[2]

model = Sequential()
# Adjust LSTM layer to use cuDNN optimized implementation
model.add(LSTM(64, activation='tanh', recurrent_activation='sigmoid', recurrent_dropout=0, unroll=False, use_bias=True, input_shape=(X_train.shape[1], number_of_features)))
model.add(Dropout(0.2))
model.add(BatchNormalization())
model.add(Dense(32, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(2))

optimizer = Nadam(learning_rate=0.002)
model.compile(optimizer=optimizer, loss='mse', metrics=['mae'])

# Consider using the ReduceLROnPlateau callback for dynamic learning rate adjustments
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, min_lr=0.0001)

# Train the model
model.fit(X_train, y_train, epochs=50, batch_size=64, validation_data=(X_val, y_val), callbacks=[reduce_lr])

# Assuming your model is named 'model'
model.save('my_model_1.h5')

y_pred = model.predict(X_test)

print("Finish predicting")

# Convert normalized predictions to original scale
y_pred_ori = scaler_response.inverse_transform(y_pred)

# If you also need to convert the normalized test labels back to their original scales:
y_test_ori = scaler_response.inverse_transform(y_test)

print("Finish converting")
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
