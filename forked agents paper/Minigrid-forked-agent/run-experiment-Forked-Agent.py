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
		self.env.reset()
		self.model.set_env(self.env)

	def get_model(self):
		return copy.copy(self.model)
		
	def set_model(self,mdl):
		self.model = mdl
		
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
			self.model.learn(total_timesteps=50000)
			training_frame+=50000
			chk = self.check()
			print("Check for stopping training : ",chk)
			if chk>=0.8:
				solved = True
			sys.stdout.flush()
		return training_frame
			
	def save(self,i):
		self.model.save('Agent_'+str(i)+'.zip')
			
	def load(self,i,e):
		self.model = PPO.load('Agent_'+str(i)+'.zip',e)
			

class MultipleAgentsOneTraining():
	def __init__ (self):
		self.Agent_grid=[]
		i=0
		self.common_agent=Agent(0)

	def sort_list(self,e):
		return(e[1].size)

	def train(self,env_seed):
		training_frame = 0
		print("Training on environment ",env_seed)
		start_time = time.time()
		solved= False
		i = 0
		while solved==False and i<len(self.Agent_grid):
			ag_test = self.Agent_grid[i][0]
			ag_test.change_environment(env_seed)
			chk = ag_test.check()
			if chk>=0.8:
				solved=True
			else:
				i+=1
		if solved==False:
			print("Solved with training agent.")
			ag=Agent(env_seed)
			ag.model.set_parameters(self.common_agent.model.get_parameters())
			ag.model.set_env(ag.model.env)
			training_frame+=ag.solve()
			self.common_agent.model.set_parameters(ag.model.get_parameters())
			grid_ag = (ag,np.array([env_seed]))
			self.Agent_grid.append(grid_ag)
			i = len(self.Agent_grid)-1
			k = 0
			while k<len(self.Agent_grid):
				list_env = self.Agent_grid[k][1].tolist()
				can_solve_all=True
				for env_test in list_env:
					self.Agent_grid[i][0].change_environment(env_test)
					if self.Agent_grid[i][0].check()>=0.8:
						if env_test not in self.Agent_grid[i][1]:
							self.Agent_grid[i]=(self.Agent_grid[i][0],np.append(self.Agent_grid[i][1],env_test))
					else:
						can_solve_all=False
				if can_solve_all==True and k!=i:
					self.Agent_grid.pop(k)
					if k<i:
						i-=1
				else:
					k+=1
		else:
			print("Solved with pool of agents.")
			if env_seed not in self.Agent_grid[i][1]:
				self.Agent_grid[i]=(self.Agent_grid[i][0],np.append(self.Agent_grid[i][1],env_seed))
		self.Agent_grid.sort(key=self.sort_list,reverse=True)
		elapsed_time = time.time() - start_time
		print("Time elapsed: ",time.strftime("%H:%M:%S", time.gmtime(elapsed_time)))
		return elapsed_time,training_frame

	def print_architecture(self):
		print("Agent eco-system architecture")
		print("Number of active agents : ",len(self.Agent_grid))
		total_env = 0
		for i in range(0,len(self.Agent_grid)):
			total_env += self.Agent_grid[i][1].size
			print("Number of environments covered : ",total_env)
		print("Architecture")
		for i in range(0,len(self.Agent_grid)):
			print("Agent : ",i)
			print("Environments : ",self.Agent_grid[i][1])
	
		
	
	def test(self,nb_env):
		max_res=0
		found = False
		j=0
		while j<len(self.Agent_grid):
			self.Agent_grid[j][0].change_environment(nb_env)
			res=self.Agent_grid[j][0].check()
			if res>max_res:
				max_res=res
			j+=1
		return max_res

def generate_results_MAOT():
	MAOT=MultipleAgentsOneTraining()
	training_time=[]
	forget = []
	general=[]
	access=[]
	t_time=0
	training_frame=0
	for i in range(0,501):
		print("Iteration: ",i)
		tm,frm=MAOT.train(np.random.randint(65000))
		t_time+=tm
		training_frame+=frm
		if i % 50 ==0:
			count_tested = 0.0
			count_ok = 0.0
			total_res = 0
			tab_res=[]
			for j in range(0,100):
				count_tested+=1.0
				res =MAOT.test(np.random.randint(65000)) 
				if res>=0.8:
					count_ok+=1.0
				if res<0:
					res=0	
				total_res+=res
				tab_res.append(res)
			general.append(count_ok/count_tested)
			print("*******************************Generalizability test : ",count_ok/count_tested,"*****************")
			print("Iter: ",i," ###############################Average return: ",total_res/count_tested,"#################")			
			print("Number of agents: ",len(MAOT.Agent_grid))
			tab_res_npy = np.array(tab_res)
			np.save("Results_"+str(i)+".npy",tab_res_npy)
	g_npy=np.array(general)
	np.save('Multi_agent_one_training_general.npy',g_npy)
	plt.title('% generalization on new environments')
	plt.xticks([0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20],[0, 50, 100, 150, 200,250,300,350,400,450,500,550,600,650,700,750,800,850,900,950,1000],rotation=90)
	plt.plot(general, color='black', label='accuracy')
	plt.legend()
	plt.savefig('Multi_agent_one_training_general.png')
	plt.close()


def main():
	random.seed(123456)
	np.random.seed(123456)
	
	generate_results_MAOT()

if __name__ == "__main__":
	main()
