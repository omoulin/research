import numpy as np
import copy
from datetime import datetime
from scipy.special import softmax
import matplotlib.pyplot as plt
import os, sys, time, datetime, json, random
#import gymnasium
#sys.modules["gym"] = gymnasium
import gym
from PIL import Image
import torch as th

from stable_baselines3 import PPO
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

	def __init__ (self,seed,n_stack=4):
		self.seed = seed
		env0 = gym.make("procgen-coinrun-v0",num_levels=20,start_level=self.seed,distribution_mode="hard")
		env0 = SeedWrapper(env0)
		env0.reset()
		self.base_env = env0
		vec = DummyVecEnv([lambda: env0])
		vec = VecMonitor(vec)
		vec = VecNormalize(vec, norm_obs=True, norm_reward=True)
		self.env = vec
		self.model = PPO(
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

def generate_models(start = 0):
	gen = []
	common_agent=Agent(0)
	print("Agent NN weights structure :")
	print("Extractor policy NN(0) weight shape :",common_agent.model.get_parameters()['policy']["mlp_extractor.policy_net.0.weight"].shape)
	print("Extractor policy NN(0) bias shape :",common_agent.model.get_parameters()['policy']["mlp_extractor.policy_net.0.bias"].shape)
	print("Extractor policy NN(2) weight shape :",common_agent.model.get_parameters()['policy']["mlp_extractor.policy_net.2.weight"].shape)
	print("Extractor policy NN(2) bias shape :",common_agent.model.get_parameters()['policy']["mlp_extractor.policy_net.2.bias"].shape)
	print("Extractor value NN(0) weight shape :",common_agent.model.get_parameters()['policy']["mlp_extractor.value_net.0.weight"].shape)
	print("Extractor value NN(0) bias shape :",common_agent.model.get_parameters()['policy']["mlp_extractor.value_net.0.bias"].shape)
	print("Extractor value NN(2) weight shape :",common_agent.model.get_parameters()['policy']["mlp_extractor.value_net.2.weight"].shape)
	print("Extractor value NN(2) bias shape :",common_agent.model.get_parameters()['policy']["mlp_extractor.value_net.2.bias"].shape)
	print("Action NN weight shape :",common_agent.model.get_parameters()['policy']["action_net.weight"].shape)
	print("Action NN bias shape :",common_agent.model.get_parameters()['policy']["action_net.bias"].shape)
	print("Value NN weight shape :",common_agent.model.get_parameters()['policy']["value_net.weight"].shape)
	print("Value NN bias shape :",common_agent.model.get_parameters()['policy']["value_net.bias"].shape)

	for i in range(start,start+200):
		print("Environment: ",i)
		ag=Agent(i)
		ag.change_environment(i)
		ag.solve()
		print("Trained on 1 000 000 steps.")
		r = ag.check()
		print("Final reward : ",r)
		print("Saving model")
		ag.model.save('../../../experiments/Coinrun/models/Model_'+str(i)+'.zip')
		weights_EXT_1 = ag.model.get_parameters()['policy']["mlp_extractor.policy_net.0.weight"]
		weights_EXT_2 = ag.model.get_parameters()['policy']["mlp_extractor.policy_net.2.weight"]
		weights_ACT = ag.model.get_parameters()['policy']["action_net.weight"]
		weights_npy_EXT_1=np.array(weights_EXT_1.cpu())
		weights_npy_EXT_2=np.array(weights_EXT_2.cpu())
		weights_npy_ACT=np.array(weights_ACT.cpu())
		print("Saving weights")
		np.save('../../../experiments/Coinrun/weights/Weights_EXT_1_'+str(i)+'.npy', weights_npy_EXT_1)
		np.save('../../../experiments/Coinrun/weights/Weights_EXT_2_'+str(i)+'.npy', weights_npy_EXT_2)
		np.save('../../../experiments/Coinrun/weights/Weights_ACT_'+str(i)+'.npy', weights_npy_ACT)
		total_ok=0.0
		for j in range(20000,20500):
			ag.change_environment(j)
			r = ag.check()
			total_ok+=r
		total_ok=total_ok/500
		gen = []
		gen.append(total_ok)
		print("Average reward : ",total_ok)
		gen_npy = np.array(gen)	
		np.save('../../../experiments/Coinrun/generalization/Average_reward'+str(i)+'.npy', gen_npy)


def main():
	random.seed(123456)
	np.random.seed(123456)
	param = sys.argv
	start = 0
	if len(param) > 1:
		start = int(param[1])
	generate_models(start)

if __name__ == "__main__":
	main()
