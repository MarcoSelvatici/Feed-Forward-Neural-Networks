import tensorflow as tf
import gym
from gym import wrappers
import numpy as np
import tflearn
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.estimator import regression
from statistics import mean, median
from collections import Counter

#plot the results
import plotly
import plotly.graph_objs as go
 
# hide warnings
import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
 
LR = 1e-3
 
env = gym.make("LunarLander-v2")
env.reset()

# Important parameters
goal_steps = 500
score_requirement = 20
initial_games = 500
final_games = 10
plot_data_before_training = []
plot_data_after_training = []

population_size = 100  # Population size
generation_limit = 100  # Max number of generations
sigma = 0.1  # Noise standard deviation
alpha = 0.00025  # Learning rate
STEPS_LIMIT = 255  # Perform the NULL_ACTION when step surpass, limit steps to enforce stopping early

RNG_SEED = 8
NULL_ACTION = 0 


def get_good_training_population():
	global plot_data_before_training
	env.seed(RNG_SEED)
	np.random.seed(RNG_SEED)

	input_size = env.observation_space.shape[0]
	output_size = env.action_space.n

	# Initial weights
	W = np.zeros((input_size, output_size))

	for gen in range(generation_limit):
		# Keep track of Returns
		R = np.zeros(population_size)
		
		# Generate noise
		N = np.random.randn(population_size, input_size, output_size)
		for j in range(population_size):
			W_ = W + sigma * N[j]
			R[j] = run_episode(env, W_)

	    # Update weights
	    # Summation of episode_weight * episode_reward
		weighted_weights = np.matmul(N.T, R).T
		new_W = W + alpha / (population_size * sigma) * weighted_weights

		gen_mean = np.mean(R)

		plot_data_before_training.append(gen_mean)

		if gen_mean >= score_requirement:
			break
		
		W = new_W
		
		print("Generation {}, Population Mean: {}".format(gen, gen_mean))
	
	print("Creating the data to train the network")
	training_data = initial_population(env, W)
	return training_data


def run_episode(environment, weight):
	global STEPS_LIMIT
	obs = environment.reset()
	episode_reward = 0
	done = False
	step = 0
	max_steps = STEPS_LIMIT
	while not done:
		if step < max_steps:
			action = np.matmul(weight.T, obs)
			move = np.argmax(action)
		else:
			move = NULL_ACTION
		obs, reward, done, info = environment.step(move)
		step += 1
		episode_reward += reward
	return episode_reward
 
# run several random games and collect training data, accepting only
# the episodes with a score higer than goal_requirment
def initial_population(env, weight):
	training_data = []
	scores = []
	accepted_scores = []
    # run all episodes
	for _ in range(initial_games):
		score = 0
		game_memory = []
		prev_observation = env.reset()
		# run each episdode
		for _ in range(goal_steps):
			action = np.matmul(weight.T, prev_observation)
			move = np.argmax(action)
			observation, reward, done, _ = env.step(move)
			
			if len(prev_observation) > 0:
				game_memory.append([prev_observation, move])

			prev_observation = observation
			score += reward
			if done:
				break

		# collect data for training from the episode just ended, if accepted
		if score >= score_requirement:
			# save the score (to compute average and median)
			accepted_scores.append(score)
			
			for data in game_memory:
				if data[1] == 0:
					output = [1,0,0,0]
				elif data[1] == 1:
					output = [0,1,0,0]
				elif data[1] == 2:
					output = [0,0,1,0]
				elif data[1] == 3:
					output = [0,0,0,1]
				training_data.append([data[0], output])
 
		# at the end of each game
		env.reset()
		scores.append(score)

	training_data_save = np.array(training_data)
 
	print('Average accepted score:', mean(accepted_scores))
	print('Median accepted score:', median(accepted_scores))
	print(Counter(accepted_scores))

	return training_data
 
# create a neural network model
def neural_network_model(input_size):
	# input layer, define the shape of the nodes
	network = input_data(shape=[None, input_size, 1], name='input')

	# 5 hidden layers (fully connected)
	# in this case it has 128 nodes and uses the rectified linear activation function
	network = fully_connected(network, 128, activation='relu')
	network = dropout(network, 0.8)

	network = fully_connected(network, 256, activation='relu')
	network = dropout(network, 0.8)

	network = fully_connected(network, 512, activation='relu')
	network = dropout(network, 0.8)

	network = fully_connected(network, 256, activation='relu')
	network = dropout(network, 0.8)

	network = fully_connected(network, 128, activation='relu')
	network = dropout(network, 0.8)

	# output layer: 4 nodes, softmax activation function
	network = fully_connected(network, 4, activation='softmax')

	# the regression model (how to modify weights and train the network)
	network = regression(network, optimizer='adam', learning_rate=LR, loss='categorical_crossentropy', name='targets')

	# create the model with the features we selected
	model = tflearn.DNN(network, tensorboard_dir='log')

	return model
 
# train a model with some trainig data
def train_model(training_data, EPOCH, model=False):
	# saving our training data differently:
	# x -> all the observations
	# y -> observation outcomes (action)
	x = np.array([i[0] for i in training_data]).reshape(-1, len(training_data[0][0]), 1)
	y = [i[1] for i in training_data]

	# if you do not pass any model, it will be created
	if not model:
		model = neural_network_model(input_size=len(x[0]))

	# train the network giving input data (x) and what they are expected to produce as output (y)
	# do that n_epoch times (NOTE: do not train too much a network with the same examples or it
	# will overfit ONLY the examples)
	model.fit({'input':x}, {'targets':y}, n_epoch=EPOCH, run_id='openaiLearning')

	return model
	 
# run the trained network
def run_game(scores, model, render = True):
	score = 0
	game_memory = []
	prev_observation = []
	env.reset()
	for _ in range(goal_steps):
		if(render):
			env.render()
		if len(prev_observation) == 0:
			action = env.action_space.sample()
		else:
            # choose the best output from the network
			a = prev_observation.reshape(-1, len(prev_observation), 1)
			action = np.argmax(model.predict(a)[0])

		new_observation, reward, done, _ = env.step(action)
		prev_observation = new_observation
		game_memory.append([new_observation, action])
		score += reward
		if done:
			#print("game {}: has ended with score {}".format(each_game,score))
			break
	scores.append(score)
 
def run_final_games(model):
	scores = []

	for i in range(final_games):
		print("episode "+str(i))
		run_game(scores, model, True)

	print('Average Score', sum(scores)/len(scores))

	global plot_data_after_training
	plot_data_after_training = scores

#create plot
def create_plot():
	plot_data = plot_data_before_training + plot_data_after_training
	#print plot_data
	trace = go.Scatter(
		x = np.linspace(0,1, initial_games/final_games + final_games),
		y = plot_data,
		mode = 'lines+markers',
		fill='tozeroy'
	)

	data = [trace]

	plotly.offline.plot({"data": data, "layout": go.Layout(title = "LunarLander")}, filename = "LunarLander_plot")


def save_model(model):
	print('do you want to save the model? [y,n] \nremember to use apexes')
	ans = str(input())
	if ans in ['y', 'Y']:
		print('select a name')
		name = str(input())
		name += '.model'
		model.save(name)
	
def main():
	print("select the number of epochs:")
	EPOCH = int(input())
	# create the training data and use them to train the model
	training_data = get_good_training_population()
	model = train_model(training_data, EPOCH, False)
	
	global env
	env.reset()
	#record video
	env = wrappers.Monitor(env, "gym-results", force=True)
	
	run_final_games(model)
	create_plot()
	save_model(model)
	
if __name__ == "__main__":
	main()

