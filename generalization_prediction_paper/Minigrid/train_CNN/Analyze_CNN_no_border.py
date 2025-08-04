import tensorflow as tf

import numpy as np
from PIL import Image as im
import matplotlib.pyplot as plt
import copy
import random
from PIL import Image
import gc

data = []

np.random.seed(123456)


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

generalization_model = tf.keras.Model(inputs=input, outputs=el)
optimizer = tf.keras.optimizers.Adam(learning_rate=0.0001)

generalization_model.compile(optimizer=optimizer, loss='mean_squared_error', metrics=['mean_squared_error'])
generalization_model.load_weights('../../../experiments/Models_init/CNN_model/predict_vg16_no_border.model')
masked_model_overall_3 = np.zeros((64,147+64+7,3))
masked_model_overall_6 = np.zeros((64,147+64+7,3))
masked_model_overall_12 = np.zeros((64,147+64+7,3))

variance_change = []
variance_change_value = []

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
for m in ra:
	print("Masking model - ",str(m))
	selection = np.random.randint(1255)
	weights_npy_EXT_1=np.load('../../../experiments/Models_init/Weights_after/Weights_EXT_1_'+str(m)+'.npy')
	weights_npy_EXT_2=np.load('../../../experiments/Models_init/Weights_after/Weights_EXT_2_'+str(m)+'.npy')
	weights_npy_ACT=np.load('../../../experiments/Models_init/Weights_after/Weights_ACT_'+str(m)+'.npy')
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
	origin=np.concatenate((image_EXT_1,image_EXT_2,image_ACT),axis=1)
	data_origin=[]
	data_origin.append(origin)
	data_origin = np.array(data_origin)
	gen = generalization_model.predict(data_origin)  
	for siz in [3,6,12]:
		masked_model = np.zeros((64,147+64+7,3))
		data=[]
		origin=[]
		for x in range(0,64-siz-1,siz):
			for y in range(0,147+64+7-siz-1,siz):
				weights_npy_EXT_1=np.load('../../../experiments/Models_init/Weights_after/Weights_EXT_1_'+str(ra[selection])+'.npy')
				weights_npy_EXT_2=np.load('../../../experiments/Models_init/Weights_after/Weights_EXT_2_'+str(ra[selection])+'.npy')
				weights_npy_ACT=np.load('../../../experiments/Models_init/Weights_after/Weights_ACT_'+str(ra[selection])+'.npy')
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
				for xp in range(0,siz):
					for yp in range(0,siz):
						image[x+xp,y+xp,:]=0
				data.append(image)
		data = np.array(data)
		results = generalization_model.predict(data)
		pos_res=0
		for x in range(0,64-siz-1,siz):
			for y in range(0,147+64+7-siz-1,siz):
				weights_npy_EXT_1=np.load('../../../experiments/Models_init/Weights_after/Weights_EXT_1_'+str(ra[selection])+'.npy')
				weights_npy_EXT_2=np.load('../../../experiments/Models_init/Weights_after/Weights_EXT_2_'+str(ra[selection])+'.npy')
				weights_npy_ACT=np.load('../../../experiments/Models_init/Weights_after/Weights_ACT_'+str(ra[selection])+'.npy')
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
				image_chk=np.concatenate((image_EXT_1,image_EXT_2,image_ACT),axis=1)	
				#print(np.max(image_chk[x:x+xp,y:y+yp]),np.min(image_chk[x:x+xp,y:y+yp]))
				variance_change.append(np.max(image_chk[x:x+xp,y:y+yp])-np.min(image_chk[x:x+xp,y:y+yp]))
				variance_change_value.append(gen-results[pos_res])
				for xp in range(0,siz):
					for yp in range(0,siz):
						masked_model[x+xp,y+yp,0]+=np.sqrt((gen-results[pos_res])*(gen-results[pos_res]))
						masked_model[x+xp,y+yp,1]+=np.sqrt((gen-results[pos_res])*(gen-results[pos_res]))
						masked_model[x+xp,y+yp,2]+=np.sqrt((gen-results[pos_res])*(gen-results[pos_res]))	
						if siz==3:
							masked_model_overall_3[x+xp,y+yp,0]+=np.sqrt((gen-results[pos_res])*(gen-results[pos_res]))
							masked_model_overall_3[x+xp,y+yp,1]+=np.sqrt((gen-results[pos_res])*(gen-results[pos_res]))
							masked_model_overall_3[x+xp,y+yp,2]+=np.sqrt((gen-results[pos_res])*(gen-results[pos_res]))
						if siz==6:
							masked_model_overall_6[x+xp,y+yp,0]+=np.sqrt((gen-results[pos_res])*(gen-results[pos_res]))
							masked_model_overall_6[x+xp,y+yp,1]+=np.sqrt((gen-results[pos_res])*(gen-results[pos_res]))
							masked_model_overall_6[x+xp,y+yp,2]+=np.sqrt((gen-results[pos_res])*(gen-results[pos_res]))
						if siz==12:
							masked_model_overall_12[x+xp,y+yp,0]+=np.sqrt((gen-results[pos_res])*(gen-results[pos_res]))
							masked_model_overall_12[x+xp,y+yp,1]+=np.sqrt((gen-results[pos_res])*(gen-results[pos_res]))
							masked_model_overall_12[x+xp,y+yp,2]+=np.sqrt((gen-results[pos_res])*(gen-results[pos_res]))
				pos_res+=1
		max = np.max(masked_model)
		masked_model = (masked_model / max)*255
		masked_model = masked_model.astype(np.uint8)
		np.save('../../../experiments/Models_init/Masked_models/Numpy_mask_'+str(ra[selection])+'_'+str(siz)+'.npy',masked_model)
		img = Image.fromarray(masked_model)
		img.save('../../../experiments/Models_init/Masked_models/Masked_model_'+str(ra[selection])+'_'+str(siz)+'.jpeg')	
		gc.collect()

