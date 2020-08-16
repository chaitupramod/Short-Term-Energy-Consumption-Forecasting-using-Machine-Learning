import threading
import logging
import time
import json
import pandas as pd
from kafka import KafkaConsumer, KafkaProducer
import os

class Producer():
        print("PRODUCER RUNNING ...")
        producer = KafkaProducer(bootstrap_servers='localhost:9094',
                                 value_serializer=lambda v: json.dumps(v).encode('utf-8'))
        
        df_read_data_readings = pd.read_csv("collection_of_readings.csv")
        producer.send('my-topic4', df_read_data_readings.to_json(orient='index'))
        print("SENT TO CONSUMER")
        
            
class Consumer():
        print("COSUMER RUNNING ...")
        consumer = KafkaConsumer(bootstrap_servers='localhost:9094',
                                 auto_offset_reset='earliest',
                                 value_deserializer=lambda m: json.loads(m.decode('utf-8')))
        
        consumer.subscribe(['my-topic4'])
        
        for message in consumer:
            dict_source = json.loads(message.value)            
            df_source = pd.DataFrame.from_dict(dict_source,orient='index')
            df_destination = pd.read_csv("C:/Users/aishw/OneDrive/Documents/kafka files/TeamTermProject/dataset/household_power_consumption/pre_processed_date.csv")
            df_concat = pd.concat([df_destination,df_source], axis=0)
            df_concat.to_csv("C:/Users/aishw/OneDrive/Documents/kafka files/TeamTermProject/dataset/household_power_consumption/pre_processed_date.csv",index=False)
            os.system("python ./TeamTermProject/code/predict_household_power_consumption.py")
            break
            
            
def main():
        
        Producer()
        Consumer()
   
main()







'''
import threading
import logging
import time
import json
from kafka import KafkaConsumer, KafkaProducer
class Producer():
        print("PROD")
        producer = KafkaProducer(bootstrap_servers='localhost:9094',
                                 value_serializer=lambda v: json.dumps(v).encode('utf-8'))
        for i in range(0,3):
            print("IN PROD")
            producer.send('my-topic1', {"dataObjectID": "TESTING"})
            time.sleep(1)
            
class Consumer():
        print("COSUMER")
        consumer = KafkaConsumer(bootstrap_servers='localhost:9094',
                                 auto_offset_reset='smallest',
                                 value_deserializer=lambda m: json.loads(m.decode('utf-8')))
        
        consumer.subscribe(['my-topic1'])
        
        for message in consumer:
            print (message.value)
            
def main():
        Producer()
        Consumer()
   
main()
'''