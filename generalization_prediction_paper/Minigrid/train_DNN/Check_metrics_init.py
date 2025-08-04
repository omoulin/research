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

			

def generate_metrics():

	siz = 0
	nb_data = 1255

	w_data = np.zeros((21,nb_data))
	y_val=np.zeros(nb_data)

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

	i=0
	correlation_score = np.array(21)
	for cpt in ra:
		y_val[i] = np.load('../../../experiments/Models_init/Generalization_after/Average_reward'+str(cpt)+'.npy')
		print("Environment: ",i)
		print("Loading weights.")
		weights_EXT_1=np.load('../../../experiments/Models_init/Weights_after/Weights_EXT_1_'+str(cpt)+'.npy').flatten()
		weights_EXT_2=np.load('../../../experiments/Models_init/Weights_after/Weights_EXT_2_'+str(cpt)+'.npy').flatten()
		weights_ACT=np.load('../../../experiments/Models_init/Weights_after/Weights_ACT_'+str(cpt)+'.npy').flatten()
		print("Generating metrict.")

		w_data[0,i]=np.average(weights_EXT_1)
		w_data[1,i]=np.var(weights_EXT_1)
		w_data[2,i]=np.percentile(weights_EXT_1,0)
		w_data[3,i]=np.percentile(weights_EXT_1,25)
		w_data[4,i]=np.percentile(weights_EXT_1,50)
		w_data[5,i]=np.percentile(weights_EXT_1,75)
		w_data[6,i]=np.percentile(weights_EXT_1,100)
		w_data[7,i]=np.average(weights_EXT_2)
		w_data[8,i]=np.var(weights_EXT_2)
		w_data[9,i]=np.percentile(weights_EXT_2,0)
		w_data[10,i]=np.percentile(weights_EXT_2,25)
		w_data[11,i]=np.percentile(weights_EXT_2,50)
		w_data[12,i]=np.percentile(weights_EXT_2,75)
		w_data[13,i]=np.percentile(weights_EXT_2,100)
		w_data[14,i]=np.average(weights_ACT)
		w_data[15,i]=np.var(weights_ACT)
		w_data[16,i]=np.percentile(weights_ACT,0)
		w_data[17,i]=np.percentile(weights_ACT,25)
		w_data[18,i]=np.percentile(weights_ACT,50)
		w_data[19,i]=np.percentile(weights_ACT,75)
		w_data[20,i]=np.percentile(weights_ACT,100)
		i+=1
	
	x_indices = np.argsort(y_val)

	plt.scatter(w_data[0,x_indices], y_val[x_indices],s=1)
	plt.xlabel('Weights average')
	plt.ylabel('Generalization score')
	plt.title('Weights average - Layer 1')
	plt.savefig('../../../experiments/Models_init/Analysis/average-layer1.pdf')
	plt.clf()
	plt.scatter(w_data[1,x_indices], y_val[x_indices],s=1)
	plt.xlabel('Weights variance')
	plt.ylabel('Generalization score')
	plt.title('Weights variance - Layer 1')
	plt.savefig('../../../experiments/Models_init/Analysis/variance-layer1.pdf')
	plt.clf()
	plt.scatter(w_data[2,x_indices], y_val[x_indices],s=1)
	plt.xlabel('Weights percentile 0')
	plt.ylabel('Generalization score')
	plt.title('Weights percentile 0 - Layer 1')
	plt.savefig('../../../experiments/Models_init/Analysis/percentile0-layer1.pdf')
	plt.clf()
	plt.scatter(w_data[3,x_indices], y_val[x_indices],s=1)
	plt.xlabel('Weights percentile 25')
	plt.ylabel('Generalization score')
	plt.title('Weights percentile 25 - Layer 1')
	plt.savefig('../../../experiments/Models_init/Analysis/percentile25-layer1.pdf')
	plt.clf()
	plt.scatter(w_data[4,x_indices], y_val[x_indices],s=1)
	plt.xlabel('Weights percentile 50')
	plt.ylabel('Generalization score')
	plt.title('Weights percentile 50 - Layer 1')
	plt.savefig('../../../experiments/Models_init/Analysis/percentile50-layer1.pdf')
	plt.clf()
	plt.scatter(w_data[5,x_indices], y_val[x_indices],s=1)
	plt.xlabel('Weights percentile 75')
	plt.ylabel('Generalization score')
	plt.title('Weights percentile 75 - Layer 1')
	plt.savefig('../../../experiments/Models_init/Analysis/percentile75-layer1.pdf')
	plt.clf()
	plt.scatter(w_data[6,x_indices], y_val[x_indices],s=1)
	plt.xlabel('Weights percentile 100')
	plt.ylabel('Generalization score')
	plt.title('Weights percentile 100 - Layer 1')
	plt.savefig('../../../experiments/Models_init/Analysis/percentile100-layer1.pdf')
	plt.clf()

	plt.scatter(w_data[7,x_indices], y_val[x_indices],s=1)
	plt.xlabel('Weights average')
	plt.ylabel('Generalization score')
	plt.title('Weights average - Layer 2')
	plt.savefig('../../../experiments/Models_init/Analysis/average-layer2.pdf')
	plt.clf()
	plt.scatter(w_data[8,x_indices], y_val[x_indices],s=1)
	plt.xlabel('Weights variance')
	plt.ylabel('Generalization score')
	plt.title('Weights variance - Layer 2')
	plt.savefig('../../../experiments/Models_init/Analysis/variance-layer2.pdf')
	plt.clf()
	plt.scatter(w_data[9,x_indices], y_val[x_indices],s=1)
	plt.xlabel('Weights percentile 0')
	plt.ylabel('Generalization score')
	plt.title('Weights percentile 0 - Layer 2')
	plt.savefig('../../../experiments/Models_init/Analysis/percentile0-layer2.pdf')
	plt.clf()
	plt.scatter(w_data[10,x_indices], y_val[x_indices],s=1)
	plt.xlabel('Weights percentile 25')
	plt.ylabel('Generalization score')
	plt.title('Weights percentile 25 - Layer 2')
	plt.savefig('../../../experiments/Models_init/Analysis/percentile25-layer2.pdf')
	plt.clf()
	plt.scatter(w_data[11,x_indices], y_val[x_indices],s=1)
	plt.xlabel('Weights percentile 50')
	plt.ylabel('Generalization score')
	plt.title('Weights percentile 50 - Layer 2')
	plt.savefig('../../../experiments/Models_init/Analysis/percentile50-layer2.pdf')
	plt.clf()
	plt.scatter(w_data[12,x_indices], y_val[x_indices],s=1)
	plt.xlabel('Weights percentile 75')
	plt.ylabel('Generalization score')
	plt.title('Weights percentile 75 - Layer 2')
	plt.savefig('../../../experiments/Models_init/Analysis/percentile75-layer2.pdf')
	plt.clf()
	plt.scatter(w_data[13,x_indices], y_val[x_indices],s=1)
	plt.xlabel('Weights percentile 100')
	plt.ylabel('Generalization score')
	plt.title('Weights percentile 100 - Layer 2')
	plt.savefig('../../../experiments/Models_init/Analysis/percentile100-layer2.pdf')
	plt.clf()

	plt.scatter(w_data[14,x_indices], y_val[x_indices],s=1)
	plt.xlabel('Weights average')
	plt.ylabel('Generalization score')
	plt.title('Weights average - Layer 3')
	plt.savefig('../../../experiments/Models_init/Analysis/average-layer3.pdf')
	plt.clf()
	plt.scatter(w_data[15,x_indices], y_val[x_indices],s=1)
	plt.xlabel('Weights variance')
	plt.ylabel('Generalization score')
	plt.title('Weights variance - Layer 3')
	plt.savefig('../../../experiments/Models_init/Analysis/variance-layer3.pdf')
	plt.clf()
	plt.scatter(w_data[16,x_indices], y_val[x_indices],s=1)
	plt.xlabel('Weights percentile 0')
	plt.ylabel('Generalization score')
	plt.title('Weights percentile 0 - Layer 3')
	plt.savefig('../../../experiments/Models_init/Analysis/percentile0-layer3.pdf')
	plt.clf()
	plt.scatter(w_data[17,x_indices], y_val[x_indices],s=1)
	plt.xlabel('Weights percentile 25')
	plt.ylabel('Generalization score')
	plt.title('Weights percentile 25 - Layer 3')
	plt.savefig('../../../experiments/Models_init/Analysis/percentile25-layer3.pdf')
	plt.clf()
	plt.scatter(w_data[18,x_indices], y_val[x_indices],s=1)
	plt.xlabel('Weights percentile 50')
	plt.ylabel('Generalization score')
	plt.title('Weights percentile 50 - Layer 3')
	plt.savefig('../../../experiments/Models_init/Analysis/percentile50-layer3.pdf')
	plt.clf()
	plt.scatter(w_data[19,x_indices], y_val[x_indices],s=1)
	plt.xlabel('Weights percentile 75')
	plt.ylabel('Generalization score')
	plt.title('Weights percentile 75 - Layer 3')
	plt.savefig('../../../experiments/Models_init/Analysis/percentile75-layer3.pdf')
	plt.clf()
	plt.scatter(w_data[20,x_indices], y_val[x_indices],s=1)
	plt.xlabel('Weights percentile 100')
	plt.ylabel('Generalization score')
	plt.title('Weights percentile 100 - Layer 3')
	plt.savefig('../../../experiments/Models_init/Analysis/percentile100-layer3.pdf')
	plt.clf()
	correlation_score = []
	x_indices = []
	cut = []
	for n in range(0,21):
		correlation_matrix = np.corrcoef(w_data[n], y_val)
		print(correlation_matrix)
		correlation_score.append(correlation_matrix[0, 1])
		x_indices.append(n)
		cut.append(0.25)
	labels = ['Avg. L1','Var. L1','Per.0 L1','Per.25 L1','Per.50 L1','Per.75 L1','Per.100 L1','Avg. L2','Var. L2','Per.0 L2','Per.25 L2','Per.50 L2','Per.75 L2','Per.100 L2','Avg. L3','Var. L3','Per.0 L3','Per.25 L3','Per.50 L3','Per.75 L3','Per.100 L3']
	correlation_score=np.abs(correlation_score)
	plt.scatter(x_indices,correlation_score,s=2)
	plt.plot(x_indices, cut, color='red', linestyle='--', label='Threeshold')
	plt.xticks(ticks=x_indices, labels=labels,rotation=90)
	plt.tight_layout()
	plt.subplots_adjust(left=0.15, right=0.95, top=0.9, bottom=0.25)
#	for i, label in enumerate(labels):
#		plt.text(x_indices[i], correlation_score[i], label, fontsize=12, ha='right')
	plt.xlabel('Statistical element')
	plt.ylabel('Absolute correlation score')
	plt.title('Pearson correlation of statistical indicators')
	plt.savefig('../../../experiments/Models_init/Analysis/pearson_correlation.pdf')
	plt.clf()

def main():
	random.seed(123456)
	np.random.seed(123456)
	
	generate_metrics()

if __name__ == "__main__":
	main()
