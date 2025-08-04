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
import time
from stable_baselines3.common.vec_env import DummyVecEnv, VecFrameStack, VecMonitor, VecNormalize
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import CheckpointCallback
import procgen
import time
from gym_minigrid.wrappers import *

class ReseedingWrapper(gym.Wrapper):
   def __init__(self, env,seed=None):
      super().__init__(env)
      self.fixseed = seed
      env.seed(self.fixseed)
      env.reset()
      self.env =env
   def reseed(self,seed):
      self.fixseed=seed
      self.env.seed(self.fixseed)
      self.env.reset()
   def reset(self, **kwargs):
      self.env.seed(self.fixseed)
      return self.env.reset()

class SeedWrapper(gym.Wrapper):
   def seed(self, seed=None):
      # Use Gym reset's seeding API
      return [seed]
      
      

class Agent():
   def __init__ (self,seed,n_stack=4,upg=0):
      self.seed = seed
      env0 = gym.make("procgen-coinrun-v0",num_levels=20,start_level=self.seed,distribution_mode="hard")
      env0 = SeedWrapper(env0)
      env0.reset()
      self.base_env = env0
      vec = DummyVecEnv([lambda: env0])
      vec = VecMonitor(vec)
      vec = VecNormalize(vec, norm_obs=True, norm_reward=True)
      self.env = vec
      if (upg ==0):
         self.model = PPO(
            policy="MlpPolicy",
            env=self.env,
            seed=123456,
            verbose=0,
         )
      else:
         self.model = PPOUpgraded(
            policy="MlpPolicy",
            env=self.env,
            seed=123456,
            verbose=0,
         )

   def change_environment(self,new_seed):
      self.seed = new_seed
      self.base_env.start_level = new_seed
      self.base_env.num_levels  = 1
      self.env.reset()

   def check(self):
      step=0
      self.base_env.num_levels  = 1
      state  = self.env.reset()
      total_rew =0
      done = False
      while done == False:
         action, _states = self.model.predict(state,deterministic=True)
         state, reward, done, _ = self.env.step(action)
         total_rew+=reward
      return total_rew

   def solve(self):
      training_frame = 0
      solved = False
      self.base_env.num_levels  = 20
      for i in range(0,5):
         self.model.learn(total_timesteps=30000)
         training_frame+=100000
         chk = self.check()
         print("Reward after training steps : ",chk)
         sys.stdout.flush()
      return training_frame

   def save(self,fld,i):
      self.model.save(fld+'Model_'+str(i)+'.zip')

   def load(self,fld,i,e):
      self.model = PPO.load(fld+'Model_'+str(i)+'.zip',e)

   def solve_cycle(self,cycle):
      self.base_env.num_levels = 20 
      self.model.learn(total_timesteps=cycle)
      training_frame=cycle
      return training_frame


def generate_models():
   gen = 0.0
   avg_rew = 0.0
   rew_train = 0.0
   steps = 0.0
   baseline_steps = 0.0
   coef= 0.5 

   for i in range(50000,50030):
      print("Environment: ",i)
      gen_baselines = []
      gen_upgraded = []
      print("Upgraded Training")
      ag=Agent(i,upg=1)
      print("Testing generalization.")
      total_reward=0
      total_ok=0
      ag.change_environment(20000)
      for j in range(20000,20500):
         ag.change_environment(j)
         r=ag.check()
         total_reward+=r
      avg_rew=total_reward/1000
      gen_upgraded.append(avg_rew)
      print("Basic Training")
      ag=Agent(i)
      print("Testing generalization.")
      total_reward=0
      total_ok=0
      ag.change_environment(20000)
      for j in range(20000,20500):
         ag.change_environment(j)
         r=ag.check()
         total_reward+=r
      baseline_avg_rew=total_reward/1000
      gen_baselines.append(baseline_avg_rew)
      print("Rew :",avg_rew," bas:",baseline_avg_rew)
   gen_upgraded=np.array(gen_upgraded) 
   gen_baselines=np.array(gen_baselines)
   print("Average_reward : ",np.average(gen_upgraded)," / baseline : ",np.average(gen_baselines))





def main():
   random.seed(123456)
   np.random.seed(123456)
   
   generate_models()

if __name__ == "__main__":
   main()
