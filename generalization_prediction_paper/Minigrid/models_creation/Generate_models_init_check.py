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
from stable_baselines3.common.vec_env import DummyVecEnv
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

	def __init__ (self,env_nb):
		self.env = gym.make('MiniGrid-SimpleCrossingS9N2-v0')
		#self.env = RGBImgObsWrapper(self.env)
		#self.env = FullyObsWrapper(self.env)
		self.env = ImgObsWrapper(self.env)
		#self.env=ReseedingWrapper(self.env)
		self.envlink = self.env
		self.env = DummyVecEnv([lambda: self.env])		
		self.seed=env_nb
		#self.envlink.reseed(self.seed)
		self.env.seed(self.seed)
		self.env.reset()
		self.model = PPO("MlpPolicy",self.env,seed=123456,verbose=0)
		self.print_environment(self.envlink)
	
	def change_environment(self,nb_env):
		#self.env = gymnasium.make('MiniGrid-FourRooms-v0')#,render_mode='human')
		#self.env = ImgObsWrapper(self.env)
		#self.env = ReseedingWrapper(self.env, seed=nb_env)	
		self.seed=nb_env
		#self.envlink.reseed(self.seed)
		self.env.seed(self.seed)
		self.env.reset()
		#self.model.set_env(self.env)

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

	def solve(self):
		training_frame = 0
		solved = False
		for i in range(0,20):
			self.model.learn(total_timesteps=100000)
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
		ag.model.set_env(ag.model.env)
		print("Saving model BEFORE")
		ag.model.save('../../../experiments/Models_init/Models_before/Model_'+str(i)+'.zip')
		weights_EXT_1 = ag.model.get_parameters()['policy']["mlp_extractor.policy_net.0.weight"]
		weights_EXT_2 = ag.model.get_parameters()['policy']["mlp_extractor.policy_net.2.weight"]
		weights_ACT = ag.model.get_parameters()['policy']["action_net.weight"]
		weights_npy_EXT_1=np.array(weights_EXT_1.cpu())
		weights_npy_EXT_2=np.array(weights_EXT_2.cpu())
		weights_npy_ACT=np.array(weights_ACT.cpu())
		print("Saving weights BEFORE")
		np.save('../../../experiments/Models_init/Weights_before/Weights_EXT_1_'+str(i)+'.npy', weights_npy_EXT_1)
		np.save('../../../experiments/Models_init/Weights_before/Weights_EXT_2_'+str(i)+'.npy', weights_npy_EXT_2)
		np.save('../../../experiments/Models_init/Weights_before/Weights_ACT_'+str(i)+'.npy', weights_npy_ACT)
		total_ok=0.0
		for j in range(20000,21000):
			ag.change_environment(j)
			r = ag.check()
			total_ok+=r
		total_ok=total_ok/1000
		gen = []
		gen.append(total_ok)
		print("Average reward BEFORE : ",total_ok)
		gen_npy = np.array(gen)	
		np.save('../../../experiments/Models_init/Generalization_before/Average_reward'+str(i)+'.npy', gen_npy)
		ag.change_environment(i)
		ag.solve()
		print("Trained on 1 000 000 steps.")
		r = ag.check()
		print("Final reward : ",r)
		print("Saving model AFTER")
		ag.model.save('../../../experiments/Models_init/Models_after/Model_'+str(i)+'.zip')
		weights_EXT_1 = ag.model.get_parameters()['policy']["mlp_extractor.policy_net.0.weight"]
		weights_EXT_2 = ag.model.get_parameters()['policy']["mlp_extractor.policy_net.2.weight"]
		weights_ACT = ag.model.get_parameters()['policy']["action_net.weight"]
		#weights_npy_EXT_1=np.array(weights_EXT_1.cpu())
		#weights_npy_EXT_2=np.array(weights_EXT_2.cpu())
		#weights_npy_ACT=np.array(weights_ACT.cpu())
		print("Saving weights AFTER")
		np.save('../../../experiments/Models_init/Weights_after/Weights_EXT_1_'+str(i)+'.npy', weights_npy_EXT_1)
		np.save('../../../experiments/Models_init/Weights_after/Weights_EXT_2_'+str(i)+'.npy', weights_npy_EXT_2)
		np.save('../../../experiments/Models_init/Weights_after/Weights_ACT_'+str(i)+'.npy', weights_npy_ACT)
		total_ok=0.0
		for j in range(20000,21000):
			ag.change_environment(j)
			r = ag.check()
			total_ok+=r
		total_ok=total_ok/1000
		gen = []
		gen.append(total_ok)
		print("Average reward AFTER : ",total_ok)
		gen_npy = np.array(gen)	
		np.save('../../../experiments/Models_init/Generalization_after/Average_reward'+str(i)+'.npy', gen_npy)


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
