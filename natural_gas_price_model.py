# -*- coding: utf-8 -*-
"""
Created on Sat Jan 18 13:16:59 2020

@author: Tisi
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import json
import requests

# Retrieve natural gas price data
url = "http://api.eia.gov/series/?api_key=ADD_YOUR_KEY_HERE&series_id=NG.RNGWHHD.D"
resp = requests.get(url)
r = json.loads(resp.content)

# Convert data to a dataframe
# covert the series dict into a dataframe 
init_data= pd.DataFrame(list(r["series"]))

# extract data column from it.
df = init_data.data

#convert the data column into a dataframe
nat_gas= pd.DataFrame(np.concatenate(df, axis=0),
                         columns= ['date', 'price'])


# Preprocessing
# check data
nat_gas.head()
nat_gas.info()

# format date 
nat_gas['date'] = pd.to_datetime(pd.Series(nat_gas['date']))

# remove time 
nat_gas['date'] = nat_gas['date'].dt.date

# set date as index 
nat_gas.set_index('date', inplace=True)

# plot data
nat_gas.plot() #the is a clear structural break in natural gas price between year 2003 and 2009. Economic boom in early - mid 2000's and recession in the late 2000's may be the cause of this break

# to avoid the structural break (2003-2011) I will only select the data from 2012 to 2020.
df = nat_gas[nat_gas.index >= pd.to_datetime("2012-01-01").date()]

# plot data
df.plot()

# check data
df.info() # one missing data and price is an object

# change price data to float
df = df.astype(float)

#deal with missing data
#check the distribution
sns.distplot(df['price'].dropna()) #the distribution look normal-ish so replace the missing value with the mean price

#replace missing value
df['price'] = df['price'].fillna(value=df['price'].mean())

# check data
df.info() # yeey no more missing values

#sort index
df.sort_index(inplace=True)

# split the data into traina and test data
# forecast about 6 months into the future - our data is daily - weekdays - roughly around 126 days, assuming an average of 21 days a month. 
test_size = 126
test_ind = len(df)- test_size
train = df.iloc[:test_ind]
test = df.iloc[test_ind:]

# scaling the train and test data
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
scaler.fit(train)
train_scaled = scaler.transform(train)
test_scaled =  scaler.transform(test)

# create  data format needed in the model for train and test 
from tensorflow.keras.preprocessing.sequence import TimeseriesGenerator

length = 10 #get the first two weeks - weekdays and predict the next day
train_gen = TimeseriesGenerator(train_scaled, train_scaled, length=length, stride=1, batch_size=1)
test_gen = TimeseriesGenerator(test_scaled, test_scaled, length=length, stride=1, batch_size=1)
#chek the input and output data generated
X,y = train_gen[0]


# Create the model
# import library
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import LSTM

# define the model
model = Sequential()
model.add(LSTM(10, activation='relu', input_shape=(length, 1)))
model.add(Dense(1))
model.compile(loss='mse', optimizer='adam')
model.summary()

# Early stopping and validation generator
from tensorflow.keras.callbacks import EarlyStopping

# early stopping
early_stop = EarlyStopping(monitor='val_loss', patience=5)

'''# Enable this area to retrain the model
# fit the model
model.fit_generator(train_gen, epochs=100,
                    validation_data=test_gen,
                   callbacks=[early_stop])

# plot loss
losses = pd.DataFrame(model.history.history)
losses.plot()
plt.savefig('loss.png')''' #the behavior of the loss curves is typical when you have too few validation set

# save the mode;
from tensorflow.keras.models import load_model
#model.save('natural_gas.h5')  #enable this line to re-save a new model

#load the model
model = load_model('natural_gas.h5')
model.summary()

# get prediction
prediction = model.predict_generator(test_gen) 


# Plot results
# revert the variables back to original scale 
prediction = scaler.inverse_transform(prediction)
train = scaler.inverse_transform(train_scaled)
test = scaler.inverse_transform(test_scaled)

# get dates from df index and split into train date and test date and get prediction dates as well from test date
date = pd.DataFrame(list(df.index), columns= ['date'] )
train_date = date.iloc[:test_ind]
test_date = (date.iloc[test_ind:]).reset_index(drop=True)
prediction_date = test_date.iloc[10:]

# plot
plt.plot(train_date, train, color = "blue",  label = "Train Set")
plt.plot(test_date, test, color = "green", label = "Test Set")
plt.plot(prediction_date, prediction, color = "red", label = "Prediction")
plt.xlabel('Date')
plt.ylabel('Price')
plt.title('Henry Hub Natural Gas Spot Price')
plt.legend()
plt.show()
plt.savefig('natural_gas.png')


''' # Enable this section if you want to test in unseen data before forecast
# Test on new data
# recall this data - let's try to use the data before 2002 as test data for our model because this data is not used by our model
nat_gas.plot() 

new_data = nat_gas[nat_gas.index <= pd.to_datetime("2002-12-31").date()]

new_data.plot()
new_data.sort_index(inplace=True)
new_data = new_data.astype(float)
test = new_data
test_scaled =  scaler.transform(test)

date = pd.DataFrame(list(new_data.index), columns= ['date'] )
test_date = date
prediction_date = test_date.iloc[10:]

length = 10
test_gen = TimeseriesGenerator(test_scaled, test_scaled, length=length, batch_size=1)

# get prediction
prediction = model.predict_generator(test_gen) 

# Plot results
# revert the predictions back to original scale 
prediction = scaler.inverse_transform(prediction)
plt.plot(test_date, test, color = "green", label = "New Test Set")
plt.plot(prediction_date, prediction, color = "red", label = "Prediction")
plt.xlabel('Date')
plt.ylabel('Price')
plt.title('Henry Hub Natural Gas Spot Price')
plt.legend()
plt.show()
'''

# Forecast 4 weeks out
#change price data to a required format
forecast_data = df.iloc[:].values
forecast_data = forecast_data.reshape((-1))

#get last date
last_date = date['date'].values[-1]

# formula to use for forecast
def predict(prediction_length, model):
    prediction_list = forecast_data[-length:]
    
    for _ in range(prediction_length):
        x = prediction_list[-length:]
        x = x.reshape(1, length, 1)
        out = model.predict(x)[0][0]
        prediction_list = np.append(prediction_list, out)
    prediction_list = prediction_list[length-1:]
    
    return prediction_list

def predict_dates(num_prediction):
    prediction_dates = pd.date_range(last_date, periods = prediction_length+1).tolist()
    
    return prediction_dates

prediction_length = 20
forecast = predict(prediction_length, model)
forecast_dates = predict_dates(prediction_length)

# plot forecast
# remove time 
forecast_dates =pd.DataFrame(forecast_dates, columns= ['date'])
forecast_dates['date']  = forecast_dates['date'].dt.date

#actual data
df_forecast = df[df.index >= pd.to_datetime("2019-01-01").date()]
date_forecast = pd.DataFrame(list(df_forecast.index), columns= ['date'] )
data_forecast = df_forecast.iloc[:].values

# plot
plt.plot(date_forecast, data_forecast , color = "green", label = "Actual Data")
plt.plot(forecast_dates, forecast, color = "red", label = "Forecast")
plt.xlabel('Date')
plt.ylabel('Price -$/MBtu ')
plt.title('Henry Hub Natural Gas Spot Price')
plt.xticks(rotation=45)
plt.legend()
plt.show()
plt.savefig('gas_price_forecast.png')

#forecast dataframe
natural_gas_forecast = pd.DataFrame(forecast, columns=['Price'])
natural_gas_forecast ['Date'] = forecast_dates['date']
natural_gas_forecast.set_index('Date', inplace=True) # natural gas price for the next 4 weeks