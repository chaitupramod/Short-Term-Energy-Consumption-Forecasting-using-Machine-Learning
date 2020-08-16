# -*- coding: utf-8 -*-
"""
Created on Thu Nov 14 22:48:48 2019

@author: chait
"""


import pandas as pd
import numpy as np
import sys
from sklearn.model_selection import TimeSeriesSplit
import sklearn.ensemble as skl_ens
import sklearn.svm as skl_svm
import sklearn.metrics as skl_metrics 
from sklearn import preprocessing
import math
from matplotlib import pyplot
import keras.layers as ksl
import warnings
import keras.models as ksm
warnings.filterwarnings('ignore')
#from keras.utils import plot_model




def text_to_csv():
    
    #convert text to CSV
    df_convert = pd.read_csv(r'C:/Users/aishw/OneDrive/Documents/kafka files/TeamTermProject/dataset/household_power_consumption/household_power_consumption.txt', delimiter=';',encoding='utf-8')
    df_convert.to_csv(r"C:/Users/aishw/OneDrive/Documents/kafka files/TeamTermProject/dataset/household_power_consumption/household_power_consumption.csv",index=False)



def preprocessing_data():
    
    df = pd.read_csv(r"C:/Users/aishw/OneDrive/Documents/kafka files/TeamTermProject/dataset/household_power_consumption/household_power_consumption.csv")    
    df_new = df[df.isin(["?"])].dropna(how="all")
    
    index_list = df_new.index.tolist()

    for i in index_list:
        print("PREPROCESSING ROW:\t"+str(i))
        for j in df_new.columns:
            if(df.loc[i,j]=="?" or (str(df.loc[i,j]=="nan") and j not in ["Date","Time"])):
                df.loc[i,j]=df.loc[i-1,j]
                

    convert_to_numeric_list = ['Global_active_power','Global_reactive_power','Voltage','Global_intensity','Sub_metering_1','Sub_metering_2','Sub_metering_3']
    
    for i in convert_to_numeric_list:
        df[i] = pd.to_numeric(df[i])

    
    dates = df["Date"].unique()
    df = df.groupby(by='Date').agg({'Global_active_power':'sum',
                              'Global_reactive_power':'sum',
                              'Voltage':'sum',
                              'Global_intensity':'sum',
                              'Sub_metering_1':'sum',
                              'Sub_metering_2':'sum',
                              'Sub_metering_3':'sum',
                              })
    
     
    df["Date"] = dates
    
    df.to_csv(r"C:/Users/aishw/OneDrive/Documents/kafka files/TeamTermProject/dataset/household_power_consumption/pre_processed_date.csv",index=False)
    
def data_visualization():
    
    import matplotlib.pyplot as plt
    df = pd.read_csv(r"C:/Users/aishw/OneDrive/Documents/kafka files/TeamTermProject/dataset/household_power_consumption/pre_processed_date.csv")
    df = df.iloc[1:1442,]
    for colName in list(df.columns.values):
        if(colName == "Date"):
            break
        print(colName)
        dates = df["Date"]  
        Global_active_power = df[colName]
        nums = [i for i in range(0,len(dates))]
        plt.figure()  
        plt.scatter(nums, Global_active_power)
        plt.savefig(colName+".png")
        plt.xlabel("days(in numbers)")
        plt.ylabel(colName+" value")
        plt.savefig(colName+".png")
        #plt.show()
       

    corr = df.corr()
    fig, ax = plt.subplots(figsize=(15, 15))
    ax.matshow(corr)
    plt.xticks(range(len(corr.columns)), corr.columns);
    plt.yticks(range(len(corr.columns)), corr.columns);
    plt.savefig("correlation_plot.png")
    #plt.show()

    
def split_train_test():
    
    df = pd.read_csv(r"C:/Users/aishw/OneDrive/Documents/kafka files/TeamTermProject/dataset/household_power_consumption/pre_processed_date.csv")
    tscv = TimeSeriesSplit(n_splits=5)
    X = np.array(df)
    cv_list = tscv.split(X)
    split_list = [i for i in cv_list]
    N=2
    cv_split_list = [split_list[n:n+N] for n in range(0, len(split_list), N)]
    return(cv_split_list)


def get_machine_learning_models():
    
    models_dict = dict()
    models_dict['RFR'] = skl_ens.RandomForestRegressor(bootstrap=True, criterion='mse', random_state=1)
    models_dict['SVR'] = skl_svm.SVR(gamma='scale')
    models_dict['LSTM']= "LSTM"
    return models_dict


def cross_validations_box_plot():
    df_plot = pd.read_csv(r"C:/Users/aishw/OneDrive/Documents/kafka files/TeamTermProject/dataset/household_power_consumption/cross_validation_data.csv")
    #df_plot.boxplot(column='Voltage', by='Model')
    #df_plot.boxplot(column='Global_intensity', by='Model')
    #df_plot.boxplot(column='Sub_metering_1', by='Model')
    #df_plot.boxplot(column='Sub_metering_2', by='Model')
    #df_plot.boxplot(column='Sub_metering_3', by='Model')



def fitting_models(model_input,model_name,featureCount,cv_num):

    df = pd.read_csv(r"C:/Users/aishw/OneDrive/Documents/kafka files/TeamTermProject/dataset/household_power_consumption/pre_processed_date.csv",usecols=["Voltage", "Global_intensity", "Sub_metering_1","Sub_metering_2","Sub_metering_3","Global_reactive_power"])
    df = df[['Voltage', 'Global_intensity', 'Sub_metering_1', 'Sub_metering_2', 'Sub_metering_3','Global_reactive_power']]
    
    
    
    df2 = df.values
    mm_scaler = preprocessing.MinMaxScaler()
    df2 = mm_scaler.fit_transform(df2)
    
    cv_count=cv_num
    
    train = []
    test = []
    
    if(cv_count==0):
        train = df2[0:182]
        test = df2[182:224]
        
    elif(cv_count==1):
        train = df2[0:357]
        test = df2[357:448]
        
    elif(cv_count==2):
        train = df2[0:539]
        test = df2[539:672]
        
    elif(cv_count==3):
        train = df2[0:714]
        test = df2[714:896]

    elif(cv_count==4):
        train = df2[0:917]
        test = df2[917:1141]
    
    elif(cv_count==5):
        train = df2[1:1142]
        test = df2[1142:-6]
        
        
    
    input_size = 7
    output_size=7
    
    train_length = len(train)/input_size
    train_split = np.split(train,train_length)
    train = np.array(train_split)
    
    test_length = len(test)/input_size
    test_split = np.split(test,test_length)
    test = np.array(test_split)
    
    recent_list = [samples for samples in train]
    predicted_list = list()
    count=0
    
    for i in range(len(test)):
        
        if(count==0):
            reshape_x = train.shape[0]*train.shape[1]
            reshape_y = train.shape[2]
            data = train.reshape((reshape_x, reshape_y))
        else:
            array_recent_list = np.array(recent_list)
            array_reshape_x = array_recent_list.shape[0]*array_recent_list.shape[1]
            array_reshape_y = array_recent_list.shape[2]
            data = array_recent_list.reshape((array_reshape_x, array_reshape_y))
            

        X_list = list()
        y_list = list()
        
        start_index = 0
        
        for l in range(len(data)):
            end_index = start_index + input_size
            out_end = end_index + output_size
            data_size = len(data)
            
            if out_end <= data_size:
                input_data_x = data[start_index:end_index, featureCount]   
                reshape_data_x = (len(input_data_x), 1)
                input_data_x = input_data_x.reshape(reshape_data_x)
                X_list.append(input_data_x)
                y_list.append(data[end_index:out_end, 0])
                
            start_index += 1
        
        x_train = np.array(X_list)
        y_train = np.array(y_list)

        forecast = list()
        
        if model_name=="LSTM":
            number_of_epochs = 5
            number_of_elements_in_batch = 16
            number_of_timesteps = x_train.shape[1] 
            number_of_features = x_train.shape[2]  
            number_of_outputs = y_train.shape[1]
            model_lstm = ksm.Sequential()
            input_shape_size = (number_of_timesteps, number_of_features)
            layer_1 = ksl.LSTM(200, activation='relu', input_shape=input_shape_size)
            layer_2 = ksl.Dense(100, activation='relu')
            layer_3 = ksl.Dense(number_of_outputs)
            model_lstm.add(layer_1)
            model_lstm.add(layer_2)
            model_lstm.add(layer_3)
            model_lstm.compile(loss='mse', optimizer='adam')
            model_lstm.fit(x_train, y_train, epochs=number_of_epochs, batch_size=number_of_elements_in_batch, verbose=1)
            model = model_lstm
            #plot_model(model, to_file='model.png')


        else:
            model = model_input
            
        x_train_reshaped = []
        temp = []
            
        for k in range(0,len(x_train)):
            x_train_reshaped.append(x_train[k].reshape(1,len(x_train[k])))
            temp.append(list(x_train_reshaped[k][0]))
    
        x_train = temp
            
        if(model_name!="LSTM"):
            model.fit(x_train,list(y_train[:,0]))
                
        data_input = x_train[-1:][0]
        
        for j in range(7):
                
            if(model_name!="LSTM"):
                X = np.array(data_input[-input_size:]).reshape(1, input_size)
                y_predict = model.predict(X)[0]

            if(model_name=="LSTM"):
                data = np.array(recent_list)
                reshape_lstm_x = data.shape[0]*data.shape[1]
                reshape_lstm_y = data.shape[2]
                data = data.reshape((reshape_lstm_x,reshape_lstm_y))
                backward_size = -7
                input_set = data[backward_size:, 0]
                reshape_for_lstm = (1, len(input_set), 1)
                input_set = input_set.reshape(reshape_for_lstm)
                y_predict = model.predict(input_set, verbose=1)
                y_predict = y_predict[0][0]
            
            forecast.append(y_predict)
            data_input.append(y_predict)
        
        predicted_list.append(forecast)
        recent_list.append(test[i, :])
        count=count+1

    actual_values = test[:,:,0]
    predicted_values = predicted_list
    predicted_values = np.array(predicted_values)
    
    #actual[:, 0] - ALl Mondays; actual[:, 1]-All Tuesdays; actual[:, 2] - All Wednesdays; actual[:, 3] - All Thursdays etc

    if(model_name!="LSTM"):
        loop_sum = 0 
        scores_list_loop = list()
        actual_arr = actual_values.shape[1]
        for i in range(actual_arr):
            actual_numpy_arr = actual_values[:, i]
            predicted_numpy_arr = predicted_values[:, i]
            mean_sqr_err = skl_metrics.mean_squared_error(actual_numpy_arr,predicted_numpy_arr)
            root_mean_squared_error = math.sqrt(mean_sqr_err)
            scores_list_loop.append(root_mean_squared_error)
            
        actual_x_shape = actual_values.shape[0]
        actual_y_shape = actual_values.shape[1]
        for row_value in range(actual_x_shape):
            for column_value in range(actual_y_shape):
                loop_sum = loop_sum + (actual_values[row_value, column_value] - predicted_values[row_value, column_value])**2
        overall_score = math.sqrt(loop_sum / (actual_values.shape[0] * actual_values.shape[1]))
        
        print("*** "+model_name+" SCORING - BY DAY ****")
        print(scores_list_loop)
        print("*** "+model_name+" - AVERAGE RMSE ****")
        print(overall_score)
    
    
    if(model_name=="LSTM"):
        iterator = 0
        lstm_records = list()
        actual_shape_1 = actual_values.shape[1]
        for i in range(actual_shape_1):
            actual_set = actual_values[:, i]
            predicted_set = predicted_values[:, i]
            mean_sqr_err = skl_metrics.mean_squared_error(actual_set, predicted_set)
            root_mean_sqrr_err = math.sqrt(mean_sqr_err)
            lstm_records.append(root_mean_sqrr_err)
        
        shape_0 = actual_values.shape[0]
        shape_1 = actual_values.shape[1]
        for row in range(shape_0):
            for col in range(shape_1):
                difference = actual_values[row, col] - predicted_values[row, col]
                iterator = iterator+(difference)**2
        shape_prod = (actual_values.shape[0] * actual_values.shape[1])
        evaluated_value = math.sqrt(iterator/shape_prod)
        scores_list_loop = lstm_records
        overall_score = evaluated_value
        
        print("*** LSTM SCORING - BY DAY ****")
        print(scores_list_loop)
        print("*** LSTM - AVERAGE RMSE ****")
        print(overall_score)

    
    return(scores_list_loop,overall_score)
    


#text_to_csv()
#preprocessing_data()
#data_visualization()
#cv_split_list = split_train_test()
models_dict = get_machine_learning_models()
print("*** Models used****")
print(models_dict)
#cv_num = [0,1,2,3,4] #for cross validation
cv_num = [5]   #only test set
cv_dict = {}
overall_RMSE = []
for cv in cv_num:
    
    for featureCount in range(0,2):
        if(cv==5):
            print("Training and Prediction for --- FEATURE"+" "+str(featureCount))
        else:
            print("Cross Validation --- FEATURE"+" "+str(featureCount)+"CV"+str(cv))
            
        scores_dict = {}
        score_dict = {}
        for i in models_dict.keys():
            result_1, result_2 = fitting_models(models_dict[i],i,featureCount,cv)
            scores_dict[i] = result_1
            score_dict[i] = result_2
            #global_scores.append(score_dict[i])
            
            cv_dict["FEATURE_"+str(featureCount)+"_"+i+"_"+str(cv_num.index(cv))] = result_2
            
                
        for i in scores_dict.keys():
            days = ['sun', 'mon', 'tue', 'wed', 'thr', 'fri', 'sat']
            pyplot.plot(days, scores_dict[i], marker='o', label=i)
            pyplot.legend()
        
        print(cv_dict)
        
        
        if(cv==5):
            pyplot.savefig(str(featureCount)+".png")
            
        #pyplot.show()  


cross_validations_box_plot()

print("OVERALL - RMSE by Feature Number")
print(cv_dict)
print("EXECUTION COMPLETED")

