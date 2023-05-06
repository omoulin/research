# Eco-system paper - (c) 2021 Olivier Moulin, Amsterdam Vrije Universiteit 

# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, version 3 of the License.

# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.

# You should have received a copy of the GNU General Public License
# along with this program.  If not, see http://www.gnu.org/licenses/.

import numpy as np
import copy
from datetime import datetime
from scipy.special import softmax
import matplotlib.pyplot as plt
import os, sys, time, datetime, json, random
import gym
from PIL import Image

from stable_baselines3 import PPO

import time
from gym_minigrid.wrappers import *


class Agent():
	def print_environment(self):
		pe = self.env.unwrapped.grid.encode()
		pe[self.env.agent_pos[0],self.env.agent_pos[1]]=10
		pe = pe[:,:,0]
		for i in range(0,19):
			line =""
			for j in range(0,19):
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
		self.env = gym.make('MiniGrid-FourRooms-v0')
		#self.env = RGBImgPartialObsWrapper(self.env)
		self.env = ImgObsWrapper(self.env)
		self.env.seed(env_nb)	
		self.env.reset()	
		self.print_environment()
		self.model = PPO("MlpPolicy",self.env,verbose=0)
		
	def change_environment(self,nb_env):
		self.env.seed(nb_env)

	def check(self):
		state = self.env.reset()
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
		while solved==False:
			self.model.learn(total_timesteps=100000)
			training_frame+=100000
			chk = self.check()
			print("Check for stopping training : ",chk)
			if chk>=0.80:
				solved = True
			sys.stdout.flush()
		return training_frame
			
	def save(self,i,it):
		self.model.save('Agent_'+str(i)+'_'+str(it)+'.mdl')
	
	def load(self,i,it):
		self.model = PPO.load('Agent_'+str(i)+'_'+str(it)+'.mdl')
			

	def get_model(self):
		return self.model

def generate_results_MAOT():
	training_time=[]
	forget = []
	general=[]
	access=[]
	training_frame=0
	Agent_grid=[]
	for i in range(0,1001):
		print("Run : ",i)
		ag = Agent(i)
		frm = ag.solve()
		grid_ag = (ag,np.array([i]))
		Agent_grid.append(grid_ag)
		training_frame+=frm
		if i % 20 ==0:
			count_tested = 0.0
			count_ok_average = 0.0
			count_ok_voting = 0.0
			total_res_average = 0
			total_res_voting = 0
			for j in range(0,30):
				count_tested+=1.0
				#Average
				test_env = gym.make('MiniGrid-FourRooms-v0')
				test_env = ImgObsWrapper(test_env)
				test_env.seed(np.random.randint(65000))	
				state = test_env.reset()
				total_rew =0
				action_options = 0
				done = False
				while done == False:
					action_options=0
					for cpt_agent in range(0,len(Agent_grid)):
						action, _states = Agent_grid[cpt_agent][0].model.predict(state,deterministic=True)
						action_options+=action
					action=round(action_options/len(Agent_grid))
					state, reward, done, _ = test_env.step(action)
					total_rew+=reward
				res=total_rew 
				if res>=0.8:
					count_ok_average+=1.0
				if res<0:
					res=0	
				total_res_average+=res
				#Voting
				state = test_env.reset()
				total_rew =0
				action_options = 0
				done = False
				while done == False:
					action_options=[0,0,0,0,0,0,0,0,0]
					for cpt_agent in range(0,len(Agent_grid)):
						action, _states = Agent_grid[cpt_agent][0].model.predict(state,deterministic=True)
						action_options[action]+=1
					action=np.argmax(action_options)
					state, reward, done, _ = test_env.step(action)
					total_rew+=reward
				res=total_rew 
				if res>=0.8:
					count_ok_voting+=1.0
				if res<0:
					res=0	
				total_res_voting+=res
			print("*******************************Generalizability test average : ",count_ok_average/count_tested,"*****************")
			print("*******************************Generalizability test voting : ",count_ok_voting/count_tested,"*****************")
			print("Total steps used :",training_frame)
			print("Average return average:",total_res_average/30)
			print("Average return voting:",total_res_voting/30)			


def main():
	random.seed(345678)
	np.random.seed(345678)
	
	generate_results_MAOT()

if __name__ == "__main__":
	main()
