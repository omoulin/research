import numpy as np
import copy
from datetime import datetime
from scipy.special import softmax
import matplotlib.pyplot as plt
import os, sys, time, datetime, json, random
import gym
from PIL import Image
import torch as th

from stable_baselines3 import PPO

import time
from gym_minigrid.wrappers import *

from catboost import CatBoostRegressor, Pool, CatBoostClassifier

			

def generate_metrics():

	nb_data = 1255
	w_data = np.zeros((nb_data,11))
	w_labels = np.zeros(nb_data)
	w_class = np.zeros(nb_data)

	ra= []
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
		dat = np.load('../../../experiments/Models_init/Model_training/data_vector_optim_'+str(i)+'.npy')
		for j in range(0,11):
			w_data[cpt,j]=dat[j]
		val = np.load('../../../experiments/Models_init/Generalization_after/Average_reward'+str(i)+'.npy')
		w_labels[cpt]=val[0][0]
		if val[0][0]>=0 and val[0][0]<0.1:
			w_class[cpt] = 0
		if val[0][0]>=0.1 and val[0][0]<0.2:
			w_class[cpt] = 1
		if val[0][0]>=0.2 and val[0][0]<0.3:
			w_class[cpt] = 2
		if val[0][0]>=0.3 and val[0][0]<0.4:
			w_class[cpt] = 3
		if val[0][0]>=0.4 and val[0][0]<0.5:
			w_class[cpt] = 4
		if val[0][0]>=0.5 and val[0][0]<0.6:
			w_class[cpt] = 5
		if val[0][0]>=0.6 and val[0][0]<0.7:
			w_class[cpt] = 6
		if val[0][0]>=0.7 and val[0][0]<0.8:
			w_class[cpt] = 7
		if val[0][0]>=0.8 and val[0][0]<0.9:
			w_class[cpt] = 8
		if val[0][0]>=0.9 and val[0][0]<1:
			w_class[cpt] = 9
	

		cpt+=1
	print(w_data)
	print(w_labels)
	print(w_class)
		
	w_data_train = w_data[0:nb_data-300]
	w_labels_train = w_labels[0:nb_data-300]
	w_class_train = w_class[0:nb_data-300]

	w_data_eval = w_data[nb_data-300:nb_data-100]
	w_labels_eval = w_labels[nb_data-300:nb_data-100]
	w_class_eval = w_class[nb_data-300:nb_data-100]
	eval_pool = Pool(data=w_data_eval, label=w_labels_eval)
	eval_class = Pool(data=w_data_eval, label=w_class_eval)

	w_data_test = w_data[nb_data-100:nb_data]
	w_labels_test = w_labels[nb_data-100:nb_data]
	w_class_test = w_class[nb_data-100:nb_data]
	
	model_regressor = CatBoostRegressor(iterations=30000,
							  learning_rate=0.03)
	model_classifier = CatBoostClassifier(iterations=30000,
							  learning_rate=0.03,eval_metric='AUC',loss_function='MultiClass')
							

	# Fit model
	#for r in range(0,20):
	#	elem = np.random.randint(nb_data-200,size=(200))
	#	w_data_train_batch=w_data_train[elem]
	#	w_labels_train_batch=w_labels_train[elem]
	#	w_class_train_batch=w_class_train[elem]
	#	model_regressor.fit(w_data_train_batch, w_labels_train_batch,eval_set=eval_pool, early_stopping_rounds=100)
	#	model_classifier.fit(w_data_train_batch, w_class_train_batch,eval_set=eval_class, early_stopping_rounds=100)
	# Get predictions
	model_regressor.fit(w_data_train, w_labels_train,eval_set=eval_pool, early_stopping_rounds=500)
	model_classifier.fit(w_data_train, w_class_train,eval_set=eval_class, early_stopping_rounds=500)


	preds = model_regressor.predict(w_data_test)
	preds_class = model_classifier.predict(w_data_test)

	model_classifier.save_model('../../../experiments/catboost_models/model_optim_classifier.cbm', format='cbm')
	model_regressor.save_model('../../../experiments/catboost_models/model_optim_regressor.cbm', format='cbm')

	x_indices = np.argsort(w_labels_test)
	x_class = np.argsort(w_class_test)

	sorted_preds=preds[x_indices]
	sorted_labels=w_labels_test[x_indices]
	x_ind = np.zeros(100)
	for i in range(0,100):
		x_ind[i]=i
	plt.scatter(x_ind,sorted_labels,zorder=2,s=1)
	plt.scatter(x_ind,sorted_preds,zorder=1,s=1)
	plt.title('Generalization prediction/labels')
	plt.savefig('TEST-Generalization_prediction_catboost_optim.pdf')
	plt.clf()

	class_sorted_preds=preds_class[x_class]
	class_sorted_labels=w_class_test[x_class]
	x_ind = np.zeros(100)
	for i in range(0,100):
		x_ind[i]=i
	plt.scatter(x_ind,class_sorted_labels,zorder=2,s=1)
	plt.scatter(x_ind,class_sorted_preds,zorder=1,s=1)
	plt.title('Generalization prediction/labels')
	plt.savefig('TEST-Generalization_prediction_catboost_optim_classifier.pdf')
	plt.clf()


	# Get predictions
	preds = model_regressor.predict(w_data_train)
	preds_class = model_classifier.predict(w_data_train)

	x_indices = np.argsort(w_labels_train)
	x_class = np.argsort(w_class_train)

	sorted_preds=preds[x_indices]
	sorted_labels=w_labels_train[x_indices]
	x_ind = np.zeros(nb_data-300)
	for i in range(0,nb_data-300):
		x_ind[i]=i
	plt.scatter(x_ind,sorted_labels,zorder=2,s=1)
	plt.scatter(x_ind,sorted_preds,zorder=1,s=1)
	plt.title('Generalization prediction/labels')
	plt.savefig('TRAIN-Generalization_prediction_catboost_optim.pdf')
	plt.clf()

	class_sorted_preds=preds_class[x_class]
	class_sorted_labels=w_class_train[x_class]

	x_ind = np.zeros(nb_data-300)
	for i in range(0,nb_data-300):
		x_ind[i]=i
	plt.scatter(x_ind,class_sorted_labels,zorder=2,s=1)
	plt.scatter(x_ind,class_sorted_preds,zorder=1,s=1)
	plt.title('Generalization prediction/labels')
	plt.savefig('TRAIN-Generalization_prediction_catboost_optim_classifier.pdf')
	plt.clf()


def main():
	random.seed(123456)
	np.random.seed(123456)
	
	generate_metrics()

if __name__ == "__main__":
	main()
