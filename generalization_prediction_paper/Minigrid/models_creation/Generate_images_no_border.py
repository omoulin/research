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


def generate_models():
	for i in range(0,11000):
		print("Environment: ",i)
		weights_npy_EXT_1=np.load('../../../Experiments/Weights/Weights_EXT_1_'+str(i)+'.npy')
		weights_npy_EXT_2=np.load('../../../Experiments/Weights/Weights_EXT_2_'+str(i)+'.npy')
		weights_npy_ACT=np.load('../../../Experiments/Weights/Weights_ACT_'+str(i)+'.npy')
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
		print("Saving image (numpy format).")
		np.save('../../../Experiments/Images_no_border/Image_'+str(i)+'.npy',image)
		print("Saving image (JPG format).")
		img = Image.fromarray(image)
		img.save('../../../Experiments/Images_no_border/Image_'+str(i)+'.jpeg')	


def main():
	random.seed(123456)
	np.random.seed(123456)
	
	generate_models()

if __name__ == "__main__":
	main()
