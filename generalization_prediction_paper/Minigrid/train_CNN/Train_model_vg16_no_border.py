import tensorflow as tf
import tensorflow_addons.metrics as met
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import copy
import random

train_data = []
train_labels = []
validation_data = []
validation_labels =[]
test_data = []

test_labels = []
data = []
labels = []

np.random.seed(123456)

ra=[]
for i in range(0,700):
   ra.append(i)
for i in range(700,741):
   ra.append(i)
for i in range(800,841):
   ra.append(i)
for i in range(900,941):
   ra.append(i)
for i in range(1000,1041):
   ra.append(i)
for i in range(1100,1141):
   ra.append(i)
for i in range(1200,1550):
   ra.append(i)
cpt =0
for i in ra:
   weights_npy_EXT_1=np.load('../../../../experiments/Models_init/Weights_after/Weights_EXT_1_'+str(i)+'.npy')
   weights_npy_EXT_2=np.load('../../../../experiments/Models_init/Weights_after/Weights_EXT_2_'+str(i)+'.npy')
   weights_npy_ACT=np.load('../../../../experiments/Models_init/Weights_after/Weights_ACT_'+str(i)+'.npy')
   image_EXT_1=np.zeros((64,147, 3), dtype=np.uint8)
   min_weight = np.min(weights_npy_EXT_1)
   if min_weight<0:
      min_weight=-min_weight
   weights_npy_EXT_1=weights_npy_EXT_1+min_weight
   max_weight = np.max(weights_npy_EXT_1)
   weights_npy_EXT_1=weights_npy_EXT_1/max_weight				
   for kx in range(0,64):
      for ky in range(0,147):
         image_EXT_1[kx,ky]=[int(weights_npy_EXT_1[kx,ky]*255),int(weights_npy_EXT_1[kx,ky]*255),int(weights_npy_EXT_1[kx,ky]*255)]
   
   image_EXT_2=np.zeros((64,64, 3), dtype=np.uint8)
   min_weight = np.min(weights_npy_EXT_2)
   if min_weight<0:
      min_weight=-min_weight
   weights_npy_EXT_2=weights_npy_EXT_2+min_weight
   max_weight = np.max(weights_npy_EXT_2)
   weights_npy_EXT_2=weights_npy_EXT_2/max_weight				
   for kx in range(0,64):
      for ky in range(0,64):
         image_EXT_2[ky,kx]=[int(weights_npy_EXT_2[kx,ky]*255),int(weights_npy_EXT_2[kx,ky]*255),int(weights_npy_EXT_2[kx,ky]*255)]
               
   image_ACT=np.zeros((64,7, 3), dtype=np.uint8)
   min_weight = np.min(weights_npy_ACT)
   if min_weight<0:
      min_weight=-min_weight
   weights_npy_ACT=weights_npy_ACT+min_weight
   max_weight = np.max(weights_npy_ACT)
   weights_npy_ACT=weights_npy_ACT/max_weight				
   for kx in range(0,7):
      for ky in range(0,64):
         image_ACT[ky,kx]=[int(weights_npy_ACT[kx,ky]*255),int(weights_npy_ACT[kx,ky]*255),int(weights_npy_ACT[kx,ky]*255)]
   image=np.concatenate((image_EXT_1,image_EXT_2,image_ACT),axis=1)
   #print(image.shape)
   data.append(image)
   gen=np.load('../../../../experiments/Models_init/Generalization_after/Average_reward'+str(i)+'.npy')
   labels.append(gen[0])
   
randomize = np.arange(len(data))
np.random.shuffle(randomize)
 
data=np.array(data)
data= data[randomize]

labels=np.array(labels)
labels= labels[randomize]



test_data=data[0:100]
test_labels=labels[0:100]

validation_data=data[100:150]
validation_labels=labels[100:150]

input = tf.keras.Input(shape=(64,147+64+7,3))

x = tf.keras.layers.Conv2D(64, (3,3), activation='relu')(input)
x = tf.keras.layers.Conv2D(64, (3,3), activation='relu')(x)
x = tf.keras.layers.MaxPooling2D((2,2))(x)
x = tf.keras.layers.Conv2D(128, (3,3), activation='relu')(x)
x = tf.keras.layers.Conv2D(128, (3,3), activation='relu')(x)
x = tf.keras.layers.MaxPooling2D((2,2))(x)
x = tf.keras.layers.Conv2D(256, (3,3), activation='relu')(x)
x = tf.keras.layers.Conv2D(256, (3,3), activation='relu')(x)
#x = tf.keras.layers.MaxPooling2D((2,2))(x)
x = tf.keras.layers.Conv2D(512, (3,3), activation='relu')(x)
x = tf.keras.layers.Conv2D(512, (3,3), activation='relu')(x)
#x = tf.keras.layers.MaxPooling2D((2,2))(x)
x = tf.keras.layers.Conv2D(512, (3,3), activation='relu')(x)
x = tf.keras.layers.Conv2D(512, (3,3), activation='relu')(x)
#x = tf.keras.layers.MaxPooling2D((2,2))(x)
x = tf.keras.layers.Flatten()(x)
el = tf.keras.layers.Dense(256, activation='relu')(x)
el = tf.keras.layers.Dropout(0.2)(el)
el = tf.keras.layers.Dense(128, activation='relu')(el)
el = tf.keras.layers.Dense(1, activation='linear')(el)
model = tf.keras.Model(inputs=input, outputs=el)

optimizer = tf.keras.optimizers.Adam(learning_rate=0.0001)
model.compile(optimizer=optimizer, loss='mean_squared_error', metrics=['mean_squared_error'])

for split in range(0,50):
   print("Running Data Chunk - ",str(split))
   list_elements=150+random.randint(0,1104)
   train_data_round=np.array([data[list_elements]])
   train_labels_round=np.array([labels[list_elements]])
   for r in range(0,400):
      list_elements=150+random.randint(0,1104)
      train_data_round=np.append(train_data_round,[data[list_elements]],axis=0)
      train_labels_round=np.append(train_labels_round,[labels[list_elements]],axis=0)
   model.fit(train_data_round, train_labels_round, epochs=20,validation_data=(validation_data,validation_labels))
   test_loss, test_acc = model.evaluate(test_data, test_labels)
   print('Test accuracy:', test_acc)
   pred = model.predict(test_data)   
   chart_prediction = np.zeros(100)
   for i in range(0,100):
      chart_prediction[i]=pred[i]
   chart_labels = np.zeros(100)
   for i in range(0,100):
      chart_labels[i]=test_labels[i]
   x_indices = np.argsort(chart_labels)
   sorted_labels = chart_labels[x_indices]
   sorted_preds = chart_prediction[x_indices]
   x_ind = np.zeros(100)
   for i in range(0,100):
      x_ind[i]=i
   plt.scatter(x_ind,sorted_labels,zorder=2,s=5,label='Actual')
   plt.scatter(x_ind,sorted_preds,zorder=1,s=5,label='Predicted')
   plt.title('Generalization prediction/labels')
   plt.legend()
   plt.savefig('../../../../experiments/Models_init/CNN_model/Predict_simple_reward_'+str(split)+'.pdf')
   plt.clf()
   model.save_weights('../../../../experiments/Models_init/CNN_model/predict_vg16_no_border.model')


