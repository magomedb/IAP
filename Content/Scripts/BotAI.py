import tensorflow as tf
import unreal_engine as ue
from TFPluginAPI import TFPluginAPI
import upypip as pip
#utility imports
from random import randint
import collections
import numpy as np

#part of structure taken from https://gist.github.com/arushir/04c58283d4fc00a4d6983dc92a3f1021
from dqn import DQN

class BotAI_API(TFPluginAPI):

	def onSetup(self):
		pass
    #setting up class and getting hyperparameter
	def setupModel(self, jsonInput):
		jsonArr = jsonInput.split(",")
		ue.log(str(jsonArr))
        #iteration counter. need this for saving reward over iteration and so on.
		self.iterations = 0

		DEFAULT_GAMMA = 0.99
		DEFAULT_REGULARIZATION = 0.001

		EPSILON = float(jsonArr[4])
		DECAY_RATE = float(jsonArr[5])
		EPSILON_MIN = float(jsonArr[6])

		MEMORY_CAPACITY = int(jsonArr[8])
		MINI_BATCH_SIZE = int(jsonArr[9])

		USE_DDQN = int(jsonArr[10])
		PRINT_OBS = int(jsonArr[11])
		PRINT_REWARD = int(jsonArr[12])
		USE_CNN = int(jsonArr[13])

		LEARNING_RATE = float(jsonArr[7])
		layer_amount = int(jsonArr[17])
		conv_layer_amount = int(jsonArr[17+(layer_amount+1)])
		IMAGE_WIDTH = int(jsonArr[17+(layer_amount+conv_layer_amount+2)])
		IMAGE_HEIGHT = int(jsonArr[17+(layer_amount+conv_layer_amount+3)])
		COLOR_CHANNELS = int(jsonArr[17+(layer_amount+conv_layer_amount+4)])
		hidden_layers = []
		conv_layers = []
		USE_MAXPOOLING = int(jsonArr[len(jsonArr)-1])
		start = len(jsonArr) - (layer_amount + conv_layer_amount + 5) #PASS PÅ DENNE!
		conv_start = len(jsonArr) - (conv_layer_amount + 4)

		for i in range(layer_amount):
				hidden_layers.append(int(jsonArr[start+i]))

		for n in range(conv_layer_amount):
		        arr = jsonArr[conv_start+n]
		        conv_layers.append(arr.split("-"))

		self.train_model = int(jsonArr[1])
		self.num_actions = int(jsonArr[3])

		self.means = []
		self.sd = []
		self.use_zscore = int(jsonArr[14])
		meansString = str(jsonArr[15])
		sdString = str(jsonArr[16])
		try:
			if self.use_zscore == 1:
				meansList = meansString.split("|")
				sdList = sdString.split("|")
                #need to convert array of string to array of floats
				self.means = [float(i) for i in meansList]
				self.sd = [float(i) for i in sdList]
		except:
			ue.log("You need to fill in means and standard deviations if you are going to use z-score normalizing")

		#prameters that we pass on to different classes
		self.cnn_params = {'lr': LEARNING_RATE, 'reg': DEFAULT_REGULARIZATION,'hidden_layers':hidden_layers, 'conv_layers': conv_layers, 'mini_batch_size': MINI_BATCH_SIZE,'use_cnn': USE_CNN, 'image_width':IMAGE_WIDTH, 'image_height':IMAGE_HEIGHT, 'color_channels':COLOR_CHANNELS, 'use_maxpooling':USE_MAXPOOLING}
		self.dqn_params = {'memory_capacity': MEMORY_CAPACITY,'epsilon': EPSILON,'gamma': DEFAULT_GAMMA,'mini_batch_size': MINI_BATCH_SIZE,'decay_rate': DECAY_RATE,'epsilon_min': EPSILON_MIN,'use_ddqn': USE_DDQN,'print_obs': PRINT_OBS,'print_reward': PRINT_REWARD}
		ue.log(str(self.dqn_params))
		ue.log(str(self.cnn_params))

		#use collections to manage a x frames buffer of input
		self.memory_capacity = 200
		self.inputQ = collections.deque(maxlen=self.memory_capacity)
		self.actionQ = collections.deque(maxlen=self.memory_capacity)

		null_input = np.zeros(int(jsonArr[2]))
		self.observation_shape = null_input.shape
		folder = jsonArr[0]
		self.model = DQN(self.num_actions, self.observation_shape, self.dqn_params, self.cnn_params, folder)
		self.mbs = MINI_BATCH_SIZE

		#fill our deque so our input size is always the same
		for x in range(0, self.memory_capacity):
			self.inputQ.append(null_input)
			self.actionQ.append(0)

		return {'model created':True}

	#parse input object and return a result object, which will be converted to json for UE4
	def onJsonInput(self, jsonInput):

		action = randint(0, self.num_actions-1)

		#make a 1D stack of current input
		observation = jsonInput['percept']

        #make observations into z-score
		if self.use_zscore == 1:
			for i in range(len(observation)):
				observation[i] = (observation[i]-self.means[i])/self.sd[i]

		reward = jsonInput['reward']

		lastAction = self.actionQ[self.memory_capacity-1]
		lastObservation = self.inputQ[self.memory_capacity-1]
		done = False
		
		# update the state 
		self.model.update_state(lastAction, lastObservation, observation, reward, done)

		# train step
		if(self.train_model == 1):
			self.model.train_step()
			#ue.log(str(observation))#Debug

		#append our stacked input to our deque
		self.inputQ.append(observation)

		action = self.model.select_action(observation, self.iterations)
		self.actionQ.append(action)
		
        #counting iterations to save when we hit our memory
		self.iterations += 1

        #Cant start training before we have enough data in memory
		if(self.iterations == self.mbs+1):
			self.model.startTraining = True  

        #Calls saveBatchReward when we are at a completly new batch for plotting
		if(self.iterations%1000 == 0):
			self.saveBatchReward()

		#return selected action
		return {'action':float(action)}


	def saveModel(self, jsonInput):
	    self.model.model.saveModel(self.inputQ, self.actionQ)
	    pass
	def saveBatchReward(self):
	    self.model.saveBatchReward(self.iterations)
	    pass

	#Start training your network
	def onBeginTraining(self):
		pass
	
#required function to get api
def getApi():
	#return CLASSNAME.getInstance()
	return BotAI_API.getInstance()