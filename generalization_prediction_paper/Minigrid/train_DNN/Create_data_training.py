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

	nb_data = 700
	w_data = np.zeros(56)
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

	for i in ra:
		w_data = np.zeros(21)
		w_data_middle = np.zeros(7)
		w_data_end = np.zeros(14)
		w_data_last = np.zeros(7)
		w_data_start = np.zeros(14)
		w_data_optim = np.zeros(11)
		print("Environment: ",i)
		print("Loading weights.")
		weights_EXT_1=np.load('../../../experiments/Models_init/Weights_after/Weights_EXT_1_'+str(i)+'.npy').flatten()
		weights_EXT_2=np.load('../../../experiments/Models_init/Weights_after/Weights_EXT_2_'+str(i)+'.npy').flatten()
		weights_ACT=np.load('../../../experiments/Models_init/Weights_after/Weights_ACT_'+str(i)+'.npy').flatten()
		print("Generating metrict.")
		w_data[0]=np.average(weights_EXT_1)
		w_data_start[0]=w_data[0]
		w_data[1]=np.var(weights_EXT_1)
		w_data_start[1]=w_data[1]
		w_data[2]=np.percentile(weights_EXT_1,0)
		w_data_start[2]=w_data[2]
		w_data_optim[0]=w_data[2]
		w_data[3]=np.percentile(weights_EXT_1,25)
		w_data_start[3]=w_data[3]
		w_data[4]=np.percentile(weights_EXT_1,50)
		w_data_start[4]=w_data[4]
		w_data[5]=np.percentile(weights_EXT_1,75)
		w_data_start[5]=w_data[5]
		w_data_optim[1]=w_data[5]
		w_data[6]=np.percentile(weights_EXT_1,100)
		w_data_start[6]=w_data[6]
		w_data_optim[2]=w_data[6]
		w_data[7]=np.average(weights_EXT_2)
		w_data_start[7]=w_data[7]
		w_data_middle[0]=w_data[7]
		w_data_end[0]=w_data[7]
		w_data[8]=np.var(weights_EXT_2)
		w_data_start[8]=w_data[8]
		w_data_middle[1]=w_data[8]
		w_data_end[1]=w_data[8]
		w_data_optim[3]=w_data[8]
		w_data[9]=np.percentile(weights_EXT_2,0)
		w_data_start[9]=w_data[9]
		w_data_middle[2]=w_data[9]
		w_data_end[2]=w_data[9]
		w_data_optim[4]=w_data[9]
		w_data[10]=np.percentile(weights_EXT_2,25)
		w_data_start[10]=w_data[10]
		w_data_middle[3]=w_data[10]
		w_data_end[3]=w_data[10]
		w_data_optim[5]=w_data[10]
		w_data[11]=np.percentile(weights_EXT_2,50)
		w_data_start[11]=w_data[11]
		w_data_middle[4]=w_data[11]
		w_data_end[4]=w_data[11]
		w_data[12]=np.percentile(weights_EXT_2,75)
		w_data_start[12]=w_data[12]
		w_data_middle[5]=w_data[12]
		w_data_end[5]=w_data[12]
		w_data_optim[6]=w_data[12]
		w_data[13]=np.percentile(weights_EXT_2,100)
		w_data_start[13]=w_data[13]
		w_data_middle[6]=w_data[13]
		w_data_end[6]=w_data[13]
		w_data_optim[7]=w_data[13]
		w_data[14]=np.average(weights_ACT)
		w_data_end[7]=w_data[14]
		w_data_last[0]=w_data[14]
		w_data[15]=np.var(weights_ACT)
		w_data_end[8]=w_data[15]
		w_data_last[1]=w_data[15]
		w_data_optim[8]=w_data[15]
		w_data[16]=np.percentile(weights_ACT,0)
		w_data_end[9]=w_data[16]
		w_data_last[2]=w_data[16]
		w_data_optim[9]=w_data[16]
		w_data[17]=np.percentile(weights_ACT,25)
		w_data_end[10]=w_data[17]
		w_data_last[3]=w_data[17]
		w_data[18]=np.percentile(weights_ACT,50)
		w_data_end[11]=w_data[18]
		w_data_last[4]=w_data[18]
		w_data[19]=np.percentile(weights_ACT,75)
		w_data_end[12]=w_data[19]
		w_data_last[5]=w_data[19]
		w_data[20]=np.percentile(weights_ACT,100)
		w_data_end[13]=w_data[20]
		w_data_last[6]=w_data[20]
		w_data_optim[10]=w_data[20]
		np.save('../../../experiments/Models_init/Model_training/data_vector_'+str(i)+'.npy',w_data)
		np.save('../../../experiments/Models_init/Model_training/data_vector_middle_'+str(i)+'.npy',w_data_middle)
		np.save('../../../experiments/Models_init/Model_training/data_vector_end_'+str(i)+'.npy',w_data_end)
		np.save('../../../experiments/Models_init/Model_training/data_vector_last_'+str(i)+'.npy',w_data_last)
		np.save('../../../experiments/Models_init/Model_training/data_vector_start_'+str(i)+'.npy',w_data_start)
		np.save('../../../experiments/Models_init/Model_training/data_vector_optim_'+str(i)+'.npy',w_data_start)

def main():
	random.seed(123456)
	np.random.seed(123456)
	
	generate_metrics()

if __name__ == "__main__":
	main()
