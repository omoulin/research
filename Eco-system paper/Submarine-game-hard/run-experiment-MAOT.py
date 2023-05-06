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
np.random.seed(122)

import tensorflow as tf
tf.random.set_seed(122)

from tensorflow.python.framework.ops import disable_eager_execution
disable_eager_execution()
from tensorflow.python.compiler.mlcompute import mlcompute
mlcompute.set_mlc_device(device_name='cpu')

from tensorflow.keras.models import Sequential, save_model, load_model, model_from_json
from tensorflow.keras.layers import Dense,PReLU,Input,Conv2D,Flatten
from tensorflow.keras.optimizers import Adam, RMSprop

from datetime import datetime
from scipy.special import softmax
import matplotlib.pyplot as plt
import os, sys, time, datetime, json, random
import gym
from PIL import Image
from deer.default_parser import process_args
from deer.agent import NeuralAgent
from deer.learning_algos.q_net_keras import MyQNetwork
import deer.experiment.base_controllers as bc
from deer.policies import EpsilonGreedyPolicy

import time

from deer.base_classes import Environment

class MarineEnv(Environment):
    def __init__(self,rng,seed,length):
        self._random_state = rng
        self._last_ponctual_observation = [0, 0, 0]
        self.seed = seed
        self.Marine=np.load('level/level-'+str(seed)+'.npy')
        self.X=0
        self.Y=5
        self.length=length
        self.Y_history = []
        self.Y_history.append(5)
        self.total_step=0
        self.display_submarine = [[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
                             [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
                             [0,0,0,0,0,0,0,0,1,1,1,1,0,0,0,0,0,0,0,0],
                             [0,0,0,0,0,0,0,0,1,1,1,1,0,0,0,0,0,0,0,0],
                             [0,0,0,0,0,0,0,0,1,1,0,0,0,0,0,0,0,0,0,0],
                             [0,0,0,0,0,0,0,0,1,1,0,0,0,0,0,0,0,0,0,0],
                             [0,0,0,0,0,0,0,0,1,1,1,1,0,0,0,0,0,0,0,0],
                             [0,0,0,0,0,0,0,0,1,1,1,1,0,0,0,0,0,0,0,0],
                             [0,0,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,0,0],
                             [0,0,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,0,0],
                             [1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1],
                             [1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1],
                             [1,1,1,1,0,0,1,1,0,0,1,1,0,0,1,1,0,0,1,1],
                             [1,1,1,1,0,0,1,1,0,0,1,1,0,0,1,1,0,0,1,1],
                             [1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1],
                             [1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1],
                             [0,0,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,0,0],
                             [0,0,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,0,0],
                             [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
                             [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
                            ]
        self.display_rock = []
        for i in range(0,20):
            line = [1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1]
            if i<5:
                for j in range(0,20):
                    line[j]=np.random.randint(0,2)
            else:
                for j in range(0,5):
                    line[j]=np.random.randint(0,2)
                for j in range(15,20):
                    line[j]=np.random.randint(0,2)
            self.display_rock.append(line)
        self.environment_access = 0

    def act(self, action):
        reward = 1
        self.total_step = self.total_step+1
        self.X=self.X+1
        if action==0:
            self.Y=self.Y-1
        if action==1:
            self.Y=self.Y+1
        if action==2:
            self.Y=self.Y
        if self.Marine[self.Y,self.X]==1:
            reward=-100
        else:
            if self.X>=self.length-1:
                reward=100
        self.Y_history.append(self.Y)
        return reward

    def reset(self,mode):
        self.X=0
        self.Y=5
        res=np.zeros((11*15+2,))
        res[0:11*15]=self.Marine[0:11,self.X:self.X+15].flatten()
        res[11*15]=self.Y
        res[11*15+1]=self.X
        state=res
        self.total_step=0
        self.Y_history=[]
        self.Y_history.append(5)
        return state

    def reset_access(self):
        self.environment_access=0

    def inputDimensions(self):
        res = []
        for i in range(0,11*15+2):
            res.append((1,))
        return res

    def nActions(self):
        return 3

    def inTerminalState(self):
        res=False
        if self.Marine[self.Y,self.X]==1:
            res= True
        else:
            if self.X>=self.length-1:
                res = True
        return res

    def observe(self):
        res=np.zeros((11*15+2,))
        res[0:11*15]=self.Marine[0:11,self.X:self.X+15].flatten()
        res[11*15]=self.Y
        res[11*15+1]=self.X
        state=res
        self.environment_access+=1
        return state

    def get_access(self):
        return self.environment_access

    def update_seed (self,seed):
        self.seed = seed
        self.Marine=np.load('level/level-'+str(seed)+'.npy')
        self.reset(0)

    def render(self):
        display_Marine = np.copy(self.Marine)
        display_Marine[self.Y,self.X]=8
        data = np.zeros((11*20,self.length*20, 3), dtype=np.uint8)
        for i in range(0,self.length):
            for j in range(0,11):
                if display_Marine[j,i]==8:
                    for l in range(0,20):
                        for m in range(0,20):
                            data[j*20+l,i*20+m]=[0,0,255]
                            if self.display_submarine[l][m]==1:
                                data[j*20+l,i*20+m]=[160,160,160]
                else:
                    for l in range(0,20):
                        for m in range(0,20):
                            if display_Marine[j,i]==0:
                                data[j*20+l,i*20+m]=[0,0,255]
                            if display_Marine[j,i]==1:
                                if self.display_rock[l][m]==1:
                                    data[j*20+l,i*20+m]=[88,41,0]
                                else:
                                    data[j*20+l,i*20+m]=[0,0,255]
                            if i==self.length-1:
                                data[j*20+l,i*20+m]=[0,255,0]
        img = Image.fromarray(data, 'RGB')
        display(img)

    def render_with_agent(self):
        display_Marine = np.copy(self.Marine)
        for i in range(0,len(self.Y_history)):
            display_Marine[self.Y_history[i],i]=8
        data = np.zeros((11*20,self.length*20, 3), dtype=np.uint8)
        for i in range(0,self.length):
            for j in range(0,11):
                if display_Marine[j,i]==8:
                    for l in range(0,20):
                        for m in range(0,20):
                            data[j*20+l,i*20+m]=[0,0,255]
                            if self.display_submarine[l][m]==1:
                                data[j*20+l,i*20+m]=[160,160,160]
                else:
                    for l in range(0,20):
                        for m in range(0,20):
                            if display_Marine[j,i]==0:
                                data[j*20+l,i*20+m]=[0,0,255]
                            if display_Marine[j,i]==1:
                                if self.display_rock[l][m]==1:
                                    data[j*20+l,i*20+m]=[88,41,0]
                                else:
                                    data[j*20+l,i*20+m]=[0,0,255]
                            if i==self.length-1:
                                data[j*20+l,i*20+m]=[0,255,0]
        img = Image.fromarray(data, 'RGB')
        display(img)

class Defaults:
    # ----------------------
    # Experiment Parameters
    # ----------------------
    STEPS_PER_EPOCH = 1000
    EPOCHS = 100
    STEPS_PER_TEST = 500
    PERIOD_BTW_SUMMARY_PERFS = 1

    # ----------------------
    # Environment Parameters
    # ----------------------
    FRAME_SKIP = 1

    # ----------------------
    # DQN Agent parameters:
    # ----------------------
    UPDATE_RULE = 'rmsprop'
    LEARNING_RATE = 0.005
    LEARNING_RATE_DECAY = 1.
    DISCOUNT = 0.9
    DISCOUNT_INC = 1.
    DISCOUNT_MAX = 0.99
    RMS_DECAY = 0.9
    RMS_EPSILON = 0.0001
    MOMENTUM = 0
    CLIP_NORM = 1.0
    EPSILON_START = 1.0
    EPSILON_MIN = .1
    EPSILON_DECAY = 10000
    UPDATE_FREQUENCY = 1
    REPLAY_MEMORY_SIZE = 1000000
    BATCH_SIZE = 32
    FREEZE_INTERVAL = 1000
    DETERMINISTIC = True

class SubMarineAgent():
    def __init__ (self,env_nb):
        self.parameters = Defaults()
        self.rng = np.random.RandomState(122)
        env = MarineEnv(self.rng,env_nb,20)
        self.qnetwork = MyQNetwork(
            env,
            self.parameters.RMS_DECAY,
            self.parameters.RMS_EPSILON,
            self.parameters.MOMENTUM,
            self.parameters.CLIP_NORM,
            self.parameters.FREEZE_INTERVAL,
            self.parameters.BATCH_SIZE,
            self.parameters.UPDATE_RULE,
            self.rng)
        self.train_policy = EpsilonGreedyPolicy(self.qnetwork, env.nActions(), self.rng, 0.1)
        self.test_policy = EpsilonGreedyPolicy(self.qnetwork, env.nActions(), self.rng, 0.)
        self.agent = NeuralAgent(
            env,
            self.qnetwork,
            self.parameters.REPLAY_MEMORY_SIZE,
            max(env.inputDimensions()[i][0] for i in range(len(env.inputDimensions()))),
            self.parameters.BATCH_SIZE,
            self.rng,
            train_policy=self.train_policy,
            test_policy=self.test_policy)
        self.agent.attach(bc.VerboseController(
            evaluate_on='epoch',
            periodicity=1))
        self.agent.attach(bc.TrainerController(
            evaluate_on='action',
            periodicity=self.parameters.UPDATE_FREQUENCY,
            show_episode_avg_V_value=False,
            show_avg_Bellman_residual=False))
        self.agent.attach(bc.LearningRateController(
            initial_learning_rate=self.parameters.LEARNING_RATE,
            learning_rate_decay=self.parameters.LEARNING_RATE_DECAY,
            periodicity=1))
        self.agent.attach(bc.DiscountFactorController(
            initial_discount_factor=self.parameters.DISCOUNT,
            discount_factor_growth=self.parameters.DISCOUNT_INC,
            discount_factor_max=self.parameters.DISCOUNT_MAX,
            periodicity=1))
        self.agent.attach(bc.EpsilonController(
            initial_e=self.parameters.EPSILON_START,
            e_decays=self.parameters.EPSILON_DECAY,
            e_min=self.parameters.EPSILON_MIN,
            evaluate_on='action',
            periodicity=1,
            reset_every='none'))
        self.agent.attach(bc.InterleavedTestEpochController(
            id=0,
            epoch_length=self.parameters.STEPS_PER_TEST,
            periodicity=1,
            show_score=True,
            summarize_every=self.parameters.PERIOD_BTW_SUMMARY_PERFS))
        self.nb_access = 0

    def change_environment(self,nb_env):
        self.agent._environment.update_seed(nb_env)

    def replay(self):
        self.agent._environment.reset(0)
        while self.agent._environment.inTerminalState()==False:
            action = self.agent._test_policy.action(self.agent._environment.observe())[0]
            self.agent._environment.act(action)
        self.agent._environment.render_with_agent()
        self.nb_access+=self.agent._environment.get_access()

    def check(self):
        self.agent._environment.reset(0)
        total_rew =0
        self.agent._environment.reset_access()
        while self.agent._environment.inTerminalState()==False:
            action = self.agent._test_policy.action(self.agent._environment.observe())[0]
            rew=self.agent._environment.act(action)
            total_rew+=rew
        self.nb_access+=self.agent._environment.get_access()
        return (total_rew)

    def generate_gif(self):
        self.agent._environment.reset(0)
        while self.agent._environment.inTerminalState()==False:
            action = self.agent._test_policy.action(self.agent._environment.observe())[0]
            self.agent._environment.act(action)
        self.agent._environment.save_gif_result()

    def reset_access(self):
        self.nb_access=0
        
    def get_access(self):
        return self.nb_access

    def solve(self):
        solved = False
        while solved==False:
            self.agent._environment.reset(0)
            self.agent._environment.reset_access()
            self.agent.run(1, self.parameters.STEPS_PER_EPOCH)
            self.nb_access+=self.agent._environment.get_access()
            self.agent._environment.reset_access()
            if self.check()>=100:
                solved = True
            self.nb_access+=self.agent._environment.get_access()

class MultipleAgentsOneTraining():
    def __init__ (self):
        self.Agent_grid=[]
        self.nb_access=0
        i=0

    def get_access(self):
        return self.nb_access

    def reset_access(self):
        self.nb_access=0

    def sort_list(self,e):
        return(e[1].size)

    def train(self,env_seed):
        training_frame = 0
        print("Training on environment ",env_seed)
        sys.stdout.flush()
        start_time = time.time()
        solved= False
        i = 0
        while solved==False and i<len(self.Agent_grid):
            ag_test = self.Agent_grid[i][0]
            ag_test.change_environment(env_seed)
            ag_test.reset_access()
            if ag_test.check()>=100:
                solved=True
            else:
                i+=1
            self.nb_access+=ag_test.get_access()
        if solved==False:
            ag=SubMarineAgent(env_seed)
            ag.reset_access()
            ag.solve()
            self.nb_access+=ag.get_access()
            grid_ag = (ag,np.array([env_seed]))
            self.Agent_grid.append(grid_ag)
            i = len(self.Agent_grid)-1
            k = 0
            while k<len(self.Agent_grid):
                list_env = self.Agent_grid[k][1].tolist()
                can_solve_all=True
                for env_test in list_env:
                    self.Agent_grid[i][0].change_environment(env_test)
                    self.Agent_grid[i][0].reset_access()
                    if self.Agent_grid[i][0].check()>=100:
                        if env_test not in self.Agent_grid[i][1]:
                            self.Agent_grid[i]=(self.Agent_grid[i][0],np.append(self.Agent_grid[i][1],env_test))
                    else:
                        can_solve_all=False
                    self.nb_access+=self.Agent_grid[i][0].get_access()
                if can_solve_all==True and k!=i:
                    self.Agent_grid.pop(k)
                    if k<i:
                        i-=1
                else:
                    k+=1
        else:
            if env_seed not in self.Agent_grid[i][1]:
                self.Agent_grid[i]=(self.Agent_grid[i][0],np.append(self.Agent_grid[i][1],env_seed))
        self.Agent_grid.sort(key=self.sort_list,reverse=True)
        elapsed_time = time.time() - start_time
        print("Time elapsed: ",time.strftime("%H:%M:%S", time.gmtime(elapsed_time)))
        sys.stdout.flush()
        return elapsed_time

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

    def checkpoint(self,it):
        for i in range(0,len(self.Agent_grid)):
            self.Agent_grid[i][0].save(i,it)
            np.save('Agent_covered_'+str(i)+'_'+str(it)+'.npy',self.Agent_grid[i][1])

    def restore(self,it):
        self.Agent_grid=[]
        for i in range(0,len(self.Agent_grid)):
            ag = SubMarineAgent(0)
            ag.load(i,it)
            env_covered = np.load('Agent_covered_'+str(i)+'_'+str(it)+'.npy')
            self.Agent_grid.append((ag,env_covered))
            
    def test(self,nb_env):
        max_res=0
        found = False
        j=0
        while found == False and j<len(self.Agent_grid):
            if nb_env in self.Agent_grid[j][1]:
                found = True
                self.Agent_grid[j][0].change_environment(nb_env)
                self.Agent_grid[j][0].reset_access()
                max_res = self.Agent_grid[j][0].check()
                self.nb_access+=self.Agent_grid[j][0].get_access()
            j+=1
        j=0
        while found == False and j<len(self.Agent_grid):
            self.Agent_grid[j][0].change_environment(nb_env)
            self.Agent_grid[j][0].reset_access()
            res=self.Agent_grid[j][0].check()
            self.nb_access+=self.Agent_grid[j][0].get_access()
            if res>max_res:
                max_res=res
            if res>=100:
                found = True
            j+=1
        return max_res
    
def generate_results_MAOT():
    for cpt in range(5):
        rnd = np.random.randint(650000)
        np.random.seed(rnd)
        tf.random.set_seed(rnd)    
        MAOT=MultipleAgentsOneTraining()
        training_time=[]
        forget = []
        general=[]
        access=[]
        t_time=0
        for i in range(0,301):
            t_time+=MAOT.train(i)
            if i%50==0:
                training_time.append(t_time)
                access.append(MAOT.get_access())
                nb_ok=0
                for j in range(0,i):
                    if MAOT.test(j)>=100:
                        nb_ok+=1
                if i>0:
                    forget.append(nb_ok/i)
                else:
                    forget.append(nb_ok)
                nb_ok=0
                for j in range(1000,2000):
                    if MAOT.test(j)>=100:
                        nb_ok+=1
                general.append(nb_ok/10)
        tt_npy = np.array(training_time)
        np.save('Multi_agent_one_training_time'+str(cpt)+'.npy',tt_npy)
        f_npy=np.array(forget)
        np.save('Multi_agent_one_training_forget'+str(cpt)+'.npy',f_npy)
        g_npy=np.array(general)
        np.save('Multi_agent_one_training_general'+str(cpt)+'.npy',g_npy)
        a_npy=np.array(access)
        np.save('Multi_agent_one_training_access'+str(cpt)+'.npy',a_npy)

def main():
    generate_results_MAOT()

if __name__ == "__main__":
    main()
