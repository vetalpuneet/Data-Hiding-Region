import cv2
import matplotlib.pyplot as plt
import numpy as np
import os
from random import shuffle
import pandas as pd
import Test as dt
from sklearn.metrics import confusion_matrix,accuracy_score

import tflearn
from tflearn.layers.conv import conv_2d, max_pool_2d
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.estimator import regression
import tensorflow as tf

lr=0.001
img_size=100

model_name="Data_Hinding_Region-()-()-model".format(lr,"6conv-basic")

def label_image(img):
    
    word_label=img[-1]
    print(word_label)
    if word_label=='ROI': return [1,0]
    elif word_label=='NROI': return [0,1]


def create_train_data(pixel1,q3,kern,thresold):
    
    dt.make_dataset(pixel1,q3,kern,thresold,"training_file.csv")
    train_dataset=pd.read_csv("training_file.csv").values
    training_data=[]
    for img in train_dataset:
        label=label_image(img)
        print(label)
        a=[]
        for i in range(0,len(img)-1,kern):
            a.append(img[i:i+kern])
        a=np.array(a)
        print(a)
        training_data.append([a,np.array(label)])
    if training_data:
        print("\ntraining data is created.")
    shuffle(training_data)
    np.save("training_data.npy",training_data)
    return training_data


def process_test_data(kern,thresold,strd):
    img=cv2.resize(cv2.imread('C:/Users/User/Untitled Folder 2/23.jpg',0),(img_size,img_size))
    plt.imshow(img,cmap="gray")
    plt.xticks([]), plt.yticks([])  # to hide tick values on X and Y axis
    plt.show()
    pixel=np.array(img)
    print("\nprocessing testing data....")
    blocks,index=dt.make_blocks(pixel,strd,kern,img_size)
    q3,pixel1,e2=dt.diff_block_div_one(blocks)
    dt.make_dataset(pixel1,q3,kern,thresold,"testing_file.csv")
    test_dataset=pd.read_csv("testing_file.csv").values
    testing_data=[]
    y_test=[]
    for img in test_dataset:
        a=[]
        for i in range(0,len(img)-1,kern):
            a.append(img[i:i+kern])
        a=np.array(a)
        print(a)
        
        testing_data.append(a)
        y_test.append(img[-1])
    if testing_data:
        print("\ntesting data is processed..")
    np.save("test_data.npy",(testing_data,y_test))
    return testing_data,y_test

def train_model(train_data,kern):
    
    train=train_data[:-500]
    test=train_data[-500:]
    x=np.array([i[0] for i in train]).reshape(-1,kern,kern,1)
    y=[i[1] for i in train]
    
    test_x=np.array([i[0] for i in test]).reshape(-1,kern,kern,1)
    test_y=[i[1] for i in test]
    
    tf.reset_default_graph()

    convnet = input_data(shape=[None, kern,kern, 1], name='input')
    #print(convnet)
    convnet = conv_2d(convnet, 16, 2, activation='relu')
    #print(convnet)
    convnet = max_pool_2d(convnet, 2)
    #print(convnet)
    
    convnet = conv_2d(convnet, 32, 2, activation='relu')
    convnet = max_pool_2d(convnet, 2)
    
    convnet = conv_2d(convnet, 16, 2, activation='relu')
    convnet = max_pool_2d(convnet, 2)
    
    convnet = conv_2d(convnet, 32, 2, activation='relu')
    convnet = max_pool_2d(convnet, 2)
    
    convnet = conv_2d(convnet, 16, 2, activation='relu')
    convnet = max_pool_2d(convnet, 2)
    
    convnet = conv_2d(convnet, 32, 2, activation='relu')
    convnet = max_pool_2d(convnet, 2)
    #print(convnet)
    convnet = fully_connected(convnet, 512, activation='relu')
    convnet = dropout(convnet, 0.8)
    
    convnet = fully_connected(convnet, 2, activation='softmax')
    convnet = regression(convnet, optimizer='adam', learning_rate=lr, loss='categorical_crossentropy', name='targets')
    
    model = tflearn.DNN(convnet,tensorboard_dir='log')
    
    model.fit({'input': x}, {'targets': y}, n_epoch=5, validation_set=({'input': test_x}, {'targets': test_y}), 
    snapshot_step=500, show_metric=True, run_id=model_name)

    model.save(model_name)
    
    if os.path.exists('{}.meta'.format(model_name)):
        model.load(model_name)
        print("model loaded!")
    return model

def test_model(model,test_data,kern):
    y_pred=[]
    for data in test_data:
        img_data=data
        data=img_data.reshape(kern,kern,1)
        model_out=model.predict([data])[0]
        if np.argmax(model_out)==1:
            str_label='NROI'
        else:
            str_label='ROI'
        y_pred.append(str_label)
    return y_pred
        
def confusion_metrics(y_test,y_pred):
    return confusion_matrix(y_test,y_pred)

def accuracy(y_test,y_pred):
    return accuracy_score(y_test, y_pred, normalize=True, sample_weight=None)


thresold=10
pixel1,q3,kern,strd=dt.main(img_size,thresold)

train_data=create_train_data(pixel1,q3,kern,thresold)

test_data,y_test=process_test_data(kern,thresold,strd)
print("\ntrainging the model.")
model=train_model(train_data,kern)
y_pred=test_model(model,test_data,kern)

matrix=confusion_metrics(y_test,y_pred)

print("confusion matrix: ",matrix)

print("accuracy of the model: ",accuracy(y_test,y_pred))










    

    

        
    