import numpy as np
import copy
from datetime import datetime
from scipy.special import softmax
import matplotlib.pyplot as plt
import os, sys, time, datetime, json, random
import gym
from PIL import Image
import time


def generate_models():
   coef =0.5 
   nb_agent = 25

   x_gen=[]
   baselines = np.zeros((nb_agent,20))
   upgraded = np.zeros((nb_agent,20))
   for i in range (0,21):
      x_gen.append(i*50000)
   x_gen=np.array(x_gen)
   for i in range(50000,50000+nb_agent):
      tmp_bs=np.load('../../../experiments/Coinrun/Training_upgraded/baselines_training_gen_'+str(i)+'.npy')
      tmp_up=np.load('../../../experiments/Coinrun/Training_upgraded/upgraded_training_gen_'+str(coef)+'_'+str(i)+'.npy')
      for j in range(0,20):
         baselines[i-50000][j]=tmp_bs[j]
         upgraded[i-50000][j]=tmp_up[j]

   avg_baselines= np.zeros(20)
   avg_upgraded= np.zeros(20)
   for i in range(0,20):
      for j in range(0,nb_agent):
         avg_baselines[i]+=baselines[j][i]
         avg_upgraded[i]+=upgraded[j][i]
      avg_baselines[i]=avg_baselines[i]/nb_agent
      avg_upgraded[i]=avg_upgraded[i]/nb_agent

   std_err_baselines= np.zeros(20)
   std_err_upgraded= np.zeros(20)
   for i in range(0,20):
      all_baselines = np.zeros(nb_agent)
      all_upgraded = np.zeros(nb_agent)      
      for j in range(0,nb_agent):
         all_baselines[j]=baselines[j][i]
         all_upgraded[j]=upgraded[j][i]
      std_err_baselines[i]=np.std(all_baselines)
      std_err_upgraded[i]=np.std(all_upgraded)
      std_err_baselines[i]=std_err_baselines[i]/np.sqrt(nb_agent)
      std_err_upgraded[i]=std_err_upgraded[i]/np.sqrt(nb_agent)

   avg_baselines=np.insert(avg_baselines,0,0.1)
   avg_upgraded=np.insert(avg_upgraded,0,0.1)
   std_err_baselines=np.insert(std_err_baselines,0,0)
   std_err_upgraded=np.insert(std_err_upgraded,0,0) 
   print(avg_baselines) 
   plt.plot(x_gen,avg_baselines,label="Baseline training",color="green")
   plt.fill_between(x_gen, avg_baselines - std_err_baselines, avg_baselines + std_err_baselines,facecolor="green",color='green',alpha=0.2)  
   plt.plot(x_gen,avg_upgraded,label="Upgraded training",color="purple")
   plt.fill_between(x_gen, avg_upgraded - std_err_upgraded, avg_upgraded + std_err_upgraded,facecolor="purple",color='purple',alpha=0.2)  
   plt.legend()
   plt.title('Coinrun - Upgraded PPO - Gen. loss coef '+str(coef))
   plt.savefig('Coinrun_UpgradedPPOvsBaseline_'+str(coef)+'.pdf')
   plt.clf()



def main():
   random.seed(123456)
   np.random.seed(123456)
   
   generate_models()

if __name__ == "__main__":
   main()
