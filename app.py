import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
import pandas_datareader as data
from pandas_datareader import data as data
yf.pdr_override()
from keras.models import load_model
import streamlit as st





start='2012-01-01'
end='2022-12-31'
st.title('Stock trend prediction')

user_input=st.text_input('Enter stock ticker','AAPL')
df= data.DataReader(user_input,start,end)

#describing data

st.subheader('Data from 2012 to 2022')
st.write(df.describe())

#visualization
st.subheader('Closing Price vs TIme chart')
fig=plt.figure(figsize=(12,6))
plt.plot(df.Close,'b')
st.pyplot(fig)



st.subheader('Closing Price vs TIme chart with 100days moving average')
ma100=df.Close.rolling(100).mean()
fig=plt.figure(figsize=(12,6))
plt.plot(ma100,'r')
plt.plot(df.Close,'b')
st.pyplot(fig)


st.subheader('Closing Price vs TIme chart with 100days and 200days moving average')
ma100=df.Close.rolling(100).mean()
ma200=df.Close.rolling(200).mean()
fig=plt.figure(figsize=(12,6))
plt.plot(ma100,'r')
plt.plot(ma200,'g')
plt.plot(df.Close,'b')
st.pyplot(fig)

#spliting data in traning and testing part 

data_training=pd.DataFrame(df['Close'][0:int(len(df)*0.70)])
data_testing=pd.DataFrame(df['Close'][int(len(df)*0.70): int(len(df))])


#scaling data
from sklearn.preprocessing import MinMaxScaler
scaler=MinMaxScaler(feature_range=(0,1))

data_training_array=scaler.fit_transform(data_training)


#loading the model(pre-trained)

model=load_model('Kera_model.keras')


#testing 
past_days=data_training.tail(100)
final_df=pd.concat([past_days,data_testing],axis =0)
input_data=scaler.fit_transform(final_df)

#testing data
x_test=[]
y_test=[]


for i in range(100,input_data.shape[0]):
    x_test.append(input_data[i-100 :i])
    y_test.append(input_data[i,0])

x_test,y_test=np.array(x_test),np.array(y_test)
y_predicted=model.predict(x_test)
scaler= scaler.scale_

scale_factor=1/scaler[0]
y_predicted=y_predicted*scale_factor
y_test=y_test*scale_factor

#final graph
st.subheader('Predictions vs original')
fig2=plt.figure(figsize=(12,6))
plt.plot(y_test,'b',label='original price')
plt.plot(y_predicted,'r',label='predicted price')
plt.xlabel('Time')
plt.ylabel('Price')
plt.legend()
st.pyplot(fig2)





