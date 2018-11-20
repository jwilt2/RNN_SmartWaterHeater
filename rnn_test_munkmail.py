# -*- coding: utf-8 -*-
"""
Created on Fri Aug  3 16:36:53 2018

@author: iem
"""

import tensorflow as tf
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import random
import tensorflow.contrib.learn as tflearn
import tensorflow.contrib.layers as tflayers
from tensorflow.contrib.learn.python.learn import learn_runner
import tensorflow.contrib.metrics as metrics
import tensorflow.contrib.rnn as rnn

#random.seed(111)
#rng=pd.date_range(start='2000',periods=209,freq='M')
#ts=pd.Series(np.random.uniform(-10,10,size=len(rng)),rng).cumsum()
#ts.plot(c='b',title='Time Series Data')
#plt.show()

COL_VHW='Water Flow (DHW)'
targetDir =r'C:\Users\3kw\.spyder-py3\house_dir'
f = []
for (dirpath, dirnames, filenames) in os.walk(targetDir):
    f.extend(filenames)
    break

file_num=14   #3 is good, 14 is amazing
hwdraw_df=pd.DataFrame()
df_inputs_multi = pd.read_csv(os.path.join(targetDir,f[file_num]),sep=',',index_col=0)       
df_inputs_multi.index=pd.to_datetime(df_inputs_multi.index)      
ts=pd.Series(df_inputs_multi[COL_VHW]/4.0)
#for i in range(0,len(ts)):
#    if i %720 ==0:
#        ts[i]=20.0
#    else:
#        ts[i]=0.0


ts=ts.resample('60T').sum()
ts=ts.fillna(0)
ts.plot(c='b',title='Time Series Data')
plt.show()


TS=np.array(ts)
num_periods=24
f_horizon=24
x_data=TS[:(len(TS)-(len(TS) % num_periods))-f_horizon]
x_batches=x_data.reshape(-1,num_periods,1)
y_data=TS[f_horizon:(len(TS)-(len(TS) % num_periods))]
y_batches = y_data.reshape(-1,num_periods,1)

def test_data(series,forecast,num_periods):
    test_x_setup=TS[-(forecast):]
    testX = test_x_setup[:num_periods].reshape(-1,num_periods,1)
    testY = TS[-(forecast):].reshape(-1,num_periods,1)
    return testX,testY

X_test,Y_test = test_data(TS,f_horizon,num_periods)

tf.reset_default_graph()

inputs=1
hidden=400
output=1

X=tf.placeholder(tf.float32,[None,num_periods,inputs])
y=tf.placeholder(tf.float32,[None,num_periods,output])

basic_cell=tf.contrib.rnn.BasicRNNCell(num_units=hidden,activation=tf.nn.relu)
rnn_output,states=tf.nn.dynamic_rnn(basic_cell, X,dtype=tf.float32)

learning_rate=0.0005

stacked_rnn_output=tf.reshape(rnn_output, [-1,hidden])
stacked_outputs=tf.layers.dense(stacked_rnn_output,output)
outputs=tf.reshape(stacked_outputs,[-1,num_periods,output])

loss=tf.reduce_sum(tf.square(outputs-y))
optimizer=tf.train.AdamOptimizer(learning_rate=learning_rate)
training_op=optimizer.minimize(loss)

init=tf.global_variables_initializer()

epochs=10000
with tf.Session() as sess:
    init.run()
    for ep in range(epochs):
        sess.run(training_op,feed_dict={X: x_batches, y: y_batches})
        if ep % 100 ==0:
            mse=loss.eval(feed_dict={X: x_batches, y: y_batches})
            print(ep, "\tMSE:",mse)
    y_pred=sess.run(outputs,feed_dict={X:X_test})
    print(y_pred)

y_valid=pd.Series(np.ravel(Y_test))
y_forecast=pd.Series(np.ravel(y_pred))
y_forecast[y_forecast<0]=0


plt.title('Forecast vs Actual')
plt.plot(y_valid,'b',label='Actual')
plt.plot(y_forecast,'r',label='Forecast')
plt.legend(loc='best')
plt.show()

print('Actual Gallons '+str(y_valid.sum()))
print('Predicted Gallons '+str(y_forecast.sum()))
y_valid.index=ts[-f_horizon:].index
y_valid_5=y_valid.resample('5T')
