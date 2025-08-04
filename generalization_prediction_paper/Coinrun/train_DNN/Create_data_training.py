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

	for i in range(1000,3800):
		w_data = np.zeros(21)
		w_data_optim = np.zeros(9)
		w_data_dnn = np.zeros(14)
		print("Environment: ",i)
		print("Loading weights.")
		weights_EXT_1=np.load('../../../experiments/Coinrun/weights/Weights_EXT_1_'+str(i)+'.npy').flatten()
		weights_EXT_2=np.load('../../../experiments/Coinrun/weights/Weights_EXT_2_'+str(i)+'.npy').flatten()
		weights_ACT=np.load('../../../experiments/Coinrun/weights/Weights_ACT_'+str(i)+'.npy').flatten()
		print("Generating metrict.")
		w_data[0]=np.average(weights_EXT_1)
		w_data[1]=np.var(weights_EXT_1)
		w_data[2]=np.percentile(weights_EXT_1,0)
		w_data[3]=np.percentile(weights_EXT_1,25)
		w_data[4]=np.percentile(weights_EXT_1,50)
		w_data[5]=np.percentile(weights_EXT_1,75)
		w_data[6]=np.percentile(weights_EXT_1,100)


		w_data[7]=np.average(weights_EXT_2)
		w_data[8]=np.var(weights_EXT_2)
		w_data_optim[0]=w_data[8]
		w_data[9]=np.percentile(weights_EXT_2,0)
		w_data_optim[1]=w_data[9]
		w_data[10]=np.percentile(weights_EXT_2,25)
		w_data_optim[2]=w_data[10]
		w_data[11]=np.percentile(weights_EXT_2,50)
		w_data[12]=np.percentile(weights_EXT_2,75)
		w_data_optim[3]=w_data[12]
		w_data[13]=np.percentile(weights_EXT_2,100)


		w_data[14]=np.average(weights_ACT)
		w_data[15]=np.var(weights_ACT)
		w_data_optim[4]=w_data[15]
		w_data[16]=np.percentile(weights_ACT,0)
		w_data_optim[5]=w_data[16]
		w_data[17]=np.percentile(weights_ACT,25)
		w_data_optim[6]=w_data[17]
		w_data[18]=np.percentile(weights_ACT,50)
		w_data[19]=np.percentile(weights_ACT,75)
		w_data_optim[7]=w_data[19]
		w_data[20]=np.percentile(weights_ACT,100)
		w_data_optim[8]=w_data[2]


		np.save('../../../experiments/Coinrun/vector/data_vector_'+str(i)+'.npy',w_data)
		np.save('../../../experiments/Coinrun/vector/data_vector_optim_'+str(i)+'.npy',w_data)

def main():
	random.seed(123456)
	np.random.seed(123456)
	
	generate_metrics()

if __name__ == "__main__":
	main()
