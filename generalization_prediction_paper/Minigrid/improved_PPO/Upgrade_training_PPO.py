import numpy as np
import copy
from datetime import datetime
from scipy.special import softmax
import matplotlib.pyplot as plt
import os, sys, time, datetime, json, random
import gym
from PIL import Image
import torch as th
import tensorflow as tf
from PPO  import PPO as PPOUpgraded
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from catboost import CatBoostRegressor, Pool, CatBoostClassifier


import time
from gym_minigrid.wrappers import *
from gym.core import ObservationWrapper

class ReseedingWrapper(gym.Wrapper):	
   def __init__(self, env, seed=None):
      super().__init__(env)
      self.seed = seed
      env.reset(seed=self.seed)
   def reset(self, **kwargs):
      return self.env.reset(seed=self.seed)
      
      
class NoisyObsWrapper(ObservationWrapper):
      noise_map = []
      
      def __init__(self, env):
         super().__init__(env)
               
      def seed(self,seed):
         super().seed(seed)
         self.noise_map=np.load('../../../experiments/Generalization_noisy/Noise_map'+str(seed)+'.npy')
      
      def observation(self, obs):
         noisy_obs=obs
         pcol = self.agent_pos[0]
         prow = self.agent_pos[1]
         tcol, trow = self.width, self.height
         ncol, nrow = obs.shape[0],obs.shape[1]
         if(pcol+ncol)>tcol:
            ncol=tcol-pcol
         if(prow+nrow)>trow:
            nrow=trow-prow		
         for i in range(0,ncol):
            for j in range(0,nrow):		
               noisy_obs[i][j]+=self.noise_map[pcol+i][prow+j]	
         return obs
      

class Agent():
         def print_environment(self,env):
            pe = env.unwrapped.grid.encode()
            pe[env.agent_pos[0],env.agent_pos[1]]=10
            pe = pe[:,:,0]
            for i in range(0,9):
               line =""
               for j in range(0,9):
                  if pe[i,j]==1:
                     line=line+" "
                  if pe[i,j]==2:
                     line = line + "#"
                  if pe[i,j]==8:
                     line=line+"X"
                  if pe[i,j]==10:
                     line=line+"@"
               print(line)
         
         def __init__ (self,env_nb,upg = 0,noise=0):
            self.env = gym.make('MiniGrid-SimpleCrossingS9N2-v0')
            self.env = ImgObsWrapper(self.env)
            if noise==1:
               self.env = NoisyObsWrapper(self.env)
            self.envlink = self.env
            self.env = DummyVecEnv([lambda: self.env])		
            self.seed=env_nb
            self.env.seed(self.seed)
            self.env.reset()
            if upg==0:
               self.model = PPO("MlpPolicy",self.env,seed=env_nb,verbose=0)
            else:
               self.model = PPOUpgraded("MlpPolicy",self.env,seed=env_nb,verbose=0)               
            self.print_environment(self.envlink)
         
         def set_noise(self):
            self.env = gym.make('MiniGrid-SimpleCrossingS9N2-v0')
            self.env = ImgObsWrapper(self.env)
            self.env = NoisyObsWrapper(self.env)
            self.envlink = self.env
            self.env = DummyVecEnv([lambda: self.env])		
            self.env.seed(self.seed)
            self.env.reset()   

         def unset_noise(self):
            self.env = gym.make('MiniGrid-SimpleCrossingS9N2-v0')
            self.env = ImgObsWrapper(self.env)
            self.envlink = self.env
            self.env = DummyVecEnv([lambda: self.env])		
            self.env.seed(self.seed)
            self.env.reset()   
            
         def change_environment(self,nb_env):
            self.seed=nb_env
            self.env.seed(self.seed)
            self.env.reset()
         
         def check(self):
            step=0 
            self.env.seed(self.seed)
            state  = self.env.reset()
            total_rew =0
            done = False
            while done == False:
               action, _states = self.model.predict(state,deterministic=True)
               state, reward, done, _ = self.env.step(action)
               total_rew+=reward
               step+=1
               if step>100:
                  done = True
            return total_rew
         
         def solve(self,mr):
            training_frame = 0
            solved = False
            while solved==False:
               self.model.learn(total_timesteps=2000000)
               training_frame+=2000000
               chk = self.check()
               print("Check for stopping training : ",chk)
               if chk>=0.8:
                  solved = True
               self.print_gen_loss(mr)
               sys.stdout.flush()
            return training_frame

         def solve_cycle(self,cycle,mr):
            self.model.learn(total_timesteps=cycle)
            self.print_gen_loss(mr)
            training_frame=cycle
            return training_frame

         def save(self,fld,i):
            self.model.save(fld+'Model_'+str(i)+'.zip')
            
         def load(self,fld,i,e):
            self.model = PPO.load(fld+'Model_'+str(i)+'.zip',e)

         def print_gen_loss(self,mr):
            weights_EXT_1 = self.model.get_parameters()['policy']["mlp_extractor.policy_net.0.weight"]
            weights_EXT_2 = self.model.get_parameters()['policy']["mlp_extractor.policy_net.2.weight"]
            weights_ACT = self.model.get_parameters()['policy']["action_net.weight"]
            weights_EXT_1=np.array(weights_EXT_1)
            weights_EXT_2=np.array(weights_EXT_2)
            weights_ACT=np.array(weights_ACT)
            w_data = np.zeros(11)
            
            w_data[0]=np.percentile(weights_EXT_1,0)
            w_data[1]=np.percentile(weights_EXT_1,75)
            w_data[2]=np.percentile(weights_EXT_1,100)
            w_data[3]=np.var(weights_EXT_2)
            w_data[4]=np.percentile(weights_EXT_2,0)
            w_data[5]=np.percentile(weights_EXT_2,25)
            w_data[6]=np.percentile(weights_EXT_2,75)
            w_data[7]=np.percentile(weights_EXT_2,100)
            w_data[8]=np.var(weights_ACT)
            w_data[9]=np.percentile(weights_ACT,0)
            w_data[10]=np.percentile(weights_ACT,100)
            gen_loss=0.8-mr.predict(w_data) 
            print("Gen loss : ",gen_loss)


def generate_models():
   gen = 0.0
   avg_rew = 0.0
   rew_train = 0.0
   steps = 0.0
   baseline_steps = 0.0
   coef= 1

   x_gen = []
   for i in range (20,70):
      x_gen.append(i)
   mr = CatBoostRegressor(iterations=30000,learning_rate=0.03)
   mr.load_model('../../../experiments/catboost_models/model_optim_regressor.cbm')
   for i in range(50000,50030):
      print("Environment: ",i)
      gen_baselines = []
      gen_upgraded = []
      print("Upgraded Training")
      ag=Agent(i,upg=1)
      for k in range(0,20):
         ag.unset_noise()
         ag.change_environment(i)
         steps = ag.solve_cycle(50000,mr)
         rew_train = ag.check()
         print("Testing generalization.")
         total_reward=0
         total_ok=0
         ag.change_environment(20000)
         ag.set_noise()
         for j in range(20000,20500):
            ag.change_environment(j)
            r=ag.check()
            total_reward+=r
            if r>=0.8:
               total_ok+=1
         gen=total_ok/1000
         avg_rew=total_reward/1000
         gen_upgraded.append(avg_rew)
      gen_upgraded=np.array(gen_upgraded)    
      np.save('../../../experiments/Training_upgraded/upgraded_training_gen_'+str(coef)+'_'+str(i)+'.npy',gen_upgraded)
      print("Basic Training")
      
      #ag=Agent(i)
      #for k in range(0,20):
      #   ag.unset_noise()
      #   ag.change_environment(i)
      #   baseline_steps = ag.solve_cycle(50000,mr)
      #   baseline_rew_train = ag.check()
      #   print("Testing generalization.")
      #   total_reward=0
      #   total_ok=0
      #   ag.change_environment(20000)
      #   ag.set_noise()
      #   for j in range(20000,20500):
      #      ag.change_environment(j)
      #      r=ag.check()
      #      total_reward+=r
      #      if r>=0.8:
      #         total_ok+=1
      #   baseline_gen=total_ok/1000
      #   baseline_avg_rew=total_reward/1000
      #   gen_baselines.append(baseline_avg_rew)
      #gen_baselines=np.array(gen_baselines)
      #np.save('../../../experiments/Training_upgraded/baselines_training_gen_'+str(i)+'.npy',gen_baselines)
      #print("Average_reward : ",np.average(gen_upgraded)," / baseline : ",np.average(gen_baselines))





def main():
   random.seed(123456)
   np.random.seed(123456)
   
   generate_models()

if __name__ == "__main__":
   main()
