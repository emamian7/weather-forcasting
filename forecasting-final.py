
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Dense, LSTM, Dropout
from keras import callbacks

#Importing the Full Dataset
full_dataset = pd.read_csv("forecasting.csv")
print(np.shape(full_dataset))
#Cleaning the Data
full_dataset.isna().sum()

full_dataset = full_dataset.dropna()
#Input Data - Just the output for 24 - Step Univariate Time Series Forecasting
X = full_dataset.iloc[:,4:5].values

#Data Pre-Processing 
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_scaled = sc.fit_transform(X)

#Preparing the Training Data for 24 hour Steps and Shape it into 3-D tensor for LSTM
X_train = []
y_train = []


for i in range(24, 63300):  
    X_train.append(X_scaled[i-24:i, 0]) # Independent variable with first 24 samples
    y_train.append(X_scaled[i, 0]) # 25th Sample used as the Target Variable
X_train, y_train = np.array(X_train), np.array(y_train) # converting the normal lists to Numpy arrays for keras to understand

from keras.regularizers import l2

#Reshaping to fit the Input 3-D tensor of the LSTM
X_train = np.reshape(X_train, (X_train.shape[0],X_train.shape[1], 1))

#Building the Model
regressor = Sequential()
#Input Layer
regressor.add(LSTM(units = 50, return_sequences = True, input_shape = (X_train.shape[1],1)))
regressor.add(Dropout(rate = 0.65))
#Hidden Layer
regressor.add(LSTM(units = 50, kernel_regularizer= l2(1)))
regressor.add(Dropout(rate = 0.65))
#Output Layer
regressor.add(Dense(units = 1))

regressor.compile(optimizer='adam', loss = 'mse', metrics = ['mae'])

monitor = callbacks.EarlyStopping(monitor='val_loss', 
                                        verbose = 0, 
                                        mode='min', 
                                        min_delta= 0.05, 
                                        patience = 3)
    # Fitting the ANN to the Training set
regressor.fit(X_train, y_train, 
              validation_split = 0.2, 
              callbacks=[monitor], 
              batch_size = 24, 
              epochs = 100, 
              verbose = 1,
              shuffle = True)

# Preparing the Testing Data
X_test = []
y_test = []

X_scaled_test = X_scaled[63200:95900,:]
for i in range(24, 32643):  
    X_test.append(X_scaled_test[i-24:i, 0]) # Independent variable with first 24 samples
    y_test.append(X_scaled_test[i, 0]) # 25th Sample used as the Target Variable
X_test, y_test = np.array(X_test), np.array(y_test) # converting the normal lists to Numpy arrays for keras to understand

X_test = np.reshape(X_test, (X_test.shape[0],X_test.shape[1], 1))

preds = regressor.predict(X_test)

#Test Set Metrics for Performance Evaluvation
#R2 Score metric
from sklearn.metrics import r2_score
print(r2_score(y_test,preds))

#MSE Score
#from sklearn.metrics import mean_squared_error
#print(mean_squared_error(y_test[],preds))

#MAE Score
from sklearn.metrics import mean_absolute_error
print(mean_absolute_error(y_test,preds))


# Visualization

y_test = sc.inverse_transform(y_test)
preds = sc.inverse_transform(preds)

#Plot of a Small Subset of the Test Set
plt.plot(y_test[0:720], color = 'blue', label = 'Real voltage')
plt.plot(preds[0:720], color = 'red', label = 'Predicted voltage')
plt.title('output - Univariate Single Step Forecasting')
plt.xlabel('Hours')
plt.ylabel('output')
plt.legend()
plt.show()
