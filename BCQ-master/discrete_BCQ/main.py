import argparse
import copy
import importlib
import json
import os

import pandas as pd
import numpy as np
import torch
 
import discrete_BCQ
import DQN
import utils
import constants

from viper.rl import *
from viper.dt import *
from viper.log  import *


def interact_with_environment(env, replay_buffer, is_atari, is_cartpole ,num_actions, state_dim, device, args, parameters):
	# For saving files
	if is_cartpole:
		setting = f"{args.env}{constants.ENV}_{args.seed}_{constants.MAX_EPISODE_STEPS}_{constants.POLE_SIZE}_{constants.BETA}_{constants.EPSILON}"
		settingLoad = f"{args.env}normal_{args.seed}_{constants.MAX_EPISODE_STEPS}_0.2_7_2"
		#settingLoad = f"{args.env}normal_{args.seed}_{constants.MAX_EPISODE_STEPS}_{constants.POLE_SIZE}_{constants.BETA}_{constants.EPSILON}" 
		buffer_name = f"{args.buffer_name}_{setting}"
	else:
		setting = f"{args.env}{constants.ENV}_{args.seed}_{constants.MAX_EPISODE_STEPS}_{constants.GRAVITY}_{constants.BETA}_{constants.EPSILON}"
		settingLoad = f"{args.env}{constants.ENV}_{args.seed}_{constants.MAX_EPISODE_STEPS}_{constants.GRAVITY}_{constants.BETA}_{constants.EPSILON}"
		#settingLoad = f"{args.env}normal_{args.seed}_{constants.MAX_EPISODE_STEPS}_{constants.GRAVITY}_{constants.BETA}_{constants.EPSILON}" 
		buffer_name = f"{args.buffer_name}_{setting}"  



	# Initialize and load policy
	policy = DQN.DQN(
		is_atari,
		is_cartpole,
		num_actions,
		state_dim,
		device,
		parameters["discount"],
		parameters["optimizer"],
		parameters["optimizer_parameters"],
		parameters["polyak_target_update"],
		parameters["target_update_freq"],
		parameters["tau"],
		parameters["initial_eps"],
		parameters["end_eps"],
		parameters["eps_decay_period"],
		parameters["eval_eps"],
	)

	if args.generate_buffer or args.grid or args.viper: policy.load(f"./models/behavioral_{settingLoad}")

	if args.viper:
		learn_dt(policy, env, setting)
		return None
	
	evaluations = []
	result = {}  # extract data from traning
	result_play = {} # extract data from playing
	i = 0
	j = 0

	state, done = env.reset(), False
	if is_cartpole:
		state = np.array([state[2],state[3]]) # 2dim
	episode_start = True
	episode_reward = 0
	episode_timesteps = 0
	episode_num = 0
	total_reward = 0
	
	low_noise_ep = np.random.uniform(0,1) < args.low_noise_p

	# For Policy Visualisation (just like "generate Buffer" but without any random noises) : 
	if args.grid and is_cartpole: 
		X = [- env.observation_space.high[2], env.observation_space.high[2]]
		j = 0		
		for x in np.linspace(min(X), max(X), 100):
			for y in np.linspace(-4, 4, 100):
				state = np.array([x,y])
				action, q_value_array = policy.select_action(state, eval=True) # Shrink the state Dim : state -> [state[2],state[3]]
				result[j] = {"Action": action, "Pole_Angle": state[0] , "Pole_Angular_Velocity": state[1], "Q_Value_0" : (q_value_array[0][0]).item() , "Q_Value_1" : (q_value_array[0][1]).item()}
				j += 1

		res = pd.DataFrame.from_dict(result, "index")
		res.to_csv(f"{setting}.csv")
		
		return None		

	if args.grid:
		X = [env.observation_space.low[0], env.observation_space.high[0]]
		Y = [env.observation_space.low[1], env.observation_space.high[1]]
		j = 0		
		for x in np.linspace(min(X), max(X), 100):
			for y in np.linspace(min(Y), max(Y), 100):
				state = np.array([x,y])
				action, q_value_array = policy.select_action(state, eval=True) # Shrink the state Dim : state -> [state[2],state[3]]
				result[j] = {"Action": action, "Car_Position": state[0] , "Car_Velocity": state[1], "Q_Value_0" : (q_value_array[0][0]).item() , "Q_Value_1" : (q_value_array[0][1]).item()} # for MC with two actions
				j += 1

		res = pd.DataFrame.from_dict(result, "index")
		res.to_csv(f"{setting}.csv")
		
		return None	

	# Interact with the environment for max_timesteps
	for t in range(int(args.max_timesteps)):

		episode_timesteps += 1
		#if episode_num % 50 == 0:
		#	env.render()

		# If generating the buffer, episode is low noise with p=low_noise_p.
		# If policy is low noise, we take random actions with p=eval_eps.
		# If the policy is high noise, we take random actions with p=rand_action_p.
		if args.generate_buffer:
			if not low_noise_ep and np.random.uniform(0,1) < args.rand_action_p - parameters["eval_eps"]:
				action = env.action_space.sample()
			else:
				action, q_value_array = policy.select_action(state, eval=True) # Shrink the state Dim : state -> [state[2],state[3]]
				if is_cartpole:
					result_play[j] = {"Episode": episode_num, "Action": action, "Pole_Angle": state[0] , "Pole_Angular_Velocity": state[1], "Q_Value_0" : (q_value_array[0][0]).item() , "Q_Value_1" : (q_value_array[0][1]).item()}
				else: 
					result_play[j] = {"Action": action, "Car_Position": state[0] , "Car_Velocity": state[1], "Q_Value_0" : (q_value_array[0][0]).item() , "Q_Value_1" : (q_value_array[0][1]).item()} # for MC with two actions 
				j += 1

		if args.train_behavioral:
			if t < parameters["start_timesteps"]:
				action = env.action_space.sample()
			else:
				action, _ = policy.select_action(state)

		# Perform action and log results
		next_state, reward, done, info = env.step(action)  # interact with env. this is the difference between my grid function and generate buffer. 
		if is_cartpole:
			next_state = np.array([next_state[2],next_state[3]]) # 2dim
		episode_reward += reward


		# Only consider "done" if episode terminates due to failure condition
		done_float = float(done) if episode_timesteps < env._max_episode_steps else 0

		# For atari, info[0] = clipped reward, info[1] = done_float
		if is_atari:
			reward = info[0]
			done_float = info[1]
			
		# Store data in replay buffer
		replay_buffer.add(state, action, next_state, reward, done_float, done, episode_start)
		state = copy.copy(next_state)
		episode_start = False

		# Train agent after collecting sufficient data
		if args.train_behavioral and t >= parameters["start_timesteps"] and (t+1) % parameters["train_freq"] == 0:
			policy.train(replay_buffer)

			
		if done:
			# +1 to account for 0 indexing. +0 on ep_timesteps since it will increment +1 even if done=True
			print(f"Total T: {t+1} Episode Num: {episode_num+1} Episode T: {episode_timesteps} Reward: {episode_reward:.3f} low_noise : {low_noise_ep} ")
			if is_cartpole:
				result[i] = {"Total T": t+1, "Episode": episode_num+1, "Reward": episode_reward}
			else:
				result[i] = {"Total T": t+1, "Episode": episode_num+1, "Position": state[0]  , "Velocity": state[1], "Reward": episode_reward}
			i += 1
			# Reset environment
			state, done = env.reset(), False
			if is_cartpole:
				state = np.array([state[2],state[3]]) # 2dim
			episode_start = True
			total_reward += episode_reward
			episode_reward = 0
			episode_timesteps = 0
			episode_num += 1
			low_noise_ep = np.random.uniform(0,1) < args.low_noise_p

		# Evaluate episode
		if args.train_behavioral and (t + 1) % parameters["eval_freq"] == 0:
			evaluations.append(eval_policy(policy, args.env, args.seed))
			np.save(f"./results/behavioral_{setting}", evaluations)
			policy.save(f"./models/behavioral_{setting}")

	# Save final policy
	if args.train_behavioral:
		policy.save(f"./models/behavioral_{setting}")

	# Save final buffer and performance
	else:
		evaluations.append(eval_policy(policy, args.env, args.seed))
		np.save(f"./results/buffer_performance_{setting}", evaluations)
		replay_buffer.save(f"./buffers/{buffer_name}")
	env.close()

	res = pd.DataFrame.from_dict(result, "index")
	if args.train_behavioral:
		res.to_csv(f"consoleOutput{setting}.csv")
	else:
		res.to_csv(f"consoleOutput{setting}_play.csv") 
		res_ply = pd.DataFrame.from_dict(result_play, "index")
		res_ply.to_csv(f"playOutput{setting}.csv")

	
	print(f"Average reward: {1.0 * total_reward / (episode_num + 1)}")



# Trains BCQ offline
def train_BCQ(env, replay_buffer, is_atari, is_cartpole, num_actions, state_dim, device, args, parameters):
	# For saving files
	setting = f"{args.env}_{args.seed}"
	buffer_name = f"{args.buffer_name}_{setting}"

	# Initialize and load policy
	policy = discrete_BCQ.discrete_BCQ(
		is_atari,
		is_cartpole,
		num_actions,
		state_dim,
		device,
		args.BCQ_threshold,
		parameters["discount"],
		parameters["optimizer"],
		parameters["optimizer_parameters"],
		parameters["polyak_target_update"],
		parameters["target_update_freq"],
		parameters["tau"],
		parameters["initial_eps"],
		parameters["end_eps"],
		parameters["eps_decay_period"],
		parameters["eval_eps"]
	)

	# Load replay buffer	
	replay_buffer.load(f"./buffers/{buffer_name}")
	
	evaluations = []
	episode_num = 0
	done = True 
	training_iters = 0
	
	while training_iters < args.max_timesteps: 
		
		for _ in range(int(parameters["eval_freq"])):
			policy.train(replay_buffer)

		evaluations.append(eval_policy(policy, args.env, args.seed))
		np.save(f"./results/BCQ_{setting}", evaluations)

		training_iters += int(parameters["eval_freq"])
		print(f"Training iterations: {training_iters}")


# Runs policy for X episodes and returns average reward
# A fixed seed is used for the eval environment
def eval_policy(policy, env_name, seed, eval_episodes=10):    # Default = 10 : Evaluation via averaging 10 previous values
	eval_env, _, _, _, _ = utils.make_env(env_name, atari_preprocessing)
	eval_env.seed(seed + 100)

	avg_reward = 0.
	for _ in range(eval_episodes):
		state, done = eval_env.reset(), False
		if policy.is_cartpole:
			state = np.array([state[2],state[3]]) # 2dim
		while not done:
			action, _ = policy.select_action(np.array(state), eval=True)
			state, reward, done, _ = eval_env.step(action)
			if policy.is_cartpole:
				state = np.array([state[2],state[3]]) # 2dim
			avg_reward += reward

	avg_reward /= eval_episodes  
	
	print("---------------------------------------")
	print(f"Evaluation over {eval_episodes} episodes: {avg_reward:.3f}")
	print("---------------------------------------")
	return avg_reward



# Insert Viper Algo 
# http://www.apache.org/licenses/LICENSE-2.0

def learn_dt(policy, env, setting):
    # Parameters
    log_fname = f"../{setting}_dt.log"
    max_depth = 12
    n_batch_rollouts = 10
    max_samples = 200000
    max_iters = 80
    train_frac = 0.8
    is_reweight = True
    n_test_rollouts = 50
    save_dirname = f"../tmp/{setting}"
    save_fname = f"dt_policy_{setting}.pk"
    save_viz_fname = f"dt_policy_{setting}.dot"
    is_train = True
    
    # Logging
    set_file(log_fname)
    
    # Data structures
    teacher = policy
    student = DTPolicy(max_depth)
    state_transformer = lambda x: x

    # Train student
    if is_train:
        student = train_dagger(env, teacher, student, state_transformer, max_iters, n_batch_rollouts, max_samples, train_frac, is_reweight, n_test_rollouts)
        save_dt_policy(student, save_dirname, save_fname)
        save_dt_policy_viz(student, save_dirname, save_viz_fname) # Labling Feature Name for Tree 
    else:
        student = load_dt_policy(save_dirname, save_fname)

    # Test student
    rew = test_policy(env, student, state_transformer, n_test_rollouts)
    log('Final reward: {}'.format(rew), INFO)
    log('Number of nodes: {}'.format(student.tree.tree_.node_count), INFO)

def bin_acts(policy, env, setting):
    # Parameters
    seq_len = 10
    n_rollouts = 10
    log_fname = f"{setting}_options.log"
    
    # Logging
    set_file(log_fname)
    
    # Data structures
    teacher = policy

    # Action sequences
    seqs = get_action_sequences(env, teacher, seq_len, n_rollouts)

    for seq, count in seqs:
        log('{}: {}'.format(seq, count), INFO)

def print_size():
    # Parameters
    dirname = 'results/run9'
    fname = 'dt_policy.pk'

    # Load decision tree
    dt = load_dt_policy(dirname, fname)

    # Size
    print(dt.tree.tree_.node_count)


if __name__ == "__main__":

	# Atari Specific
	atari_preprocessing = {
		"frame_skip": 4,
		"frame_size": 84,
		"state_history": 4,
		"done_on_life_loss": False,
		"reward_clipping": True,
		"max_episode_timesteps": 27e3
	}

	atari_parameters = {
		# Exploration
		"start_timesteps": 2e4,
		"initial_eps": 1,
		"end_eps": 1e-2,
		"eps_decay_period": 25e4,
		# Evaluation
		"eval_freq": 5e4,
		"eval_eps": 1e-3,
		# Learning
		"discount": 0.99,
		"buffer_size": 1e6,
		"batch_size": 32,
		"optimizer": "Adam",
		"optimizer_parameters": {
			"lr": 0.0000625,
			"eps": 0.00015
		},
		"train_freq": 4,
		"polyak_target_update": False,
		"target_update_freq": 8e3,
		"tau": 1
	}

	regular_parameters = {
		# Exploration
		"start_timesteps": 10000,  # execute some episodes with initial random actions and not with epsilon greedy policy. This helps a lot with exploration and faster learning
		"initial_eps": 1,          # initial epsilon in epsilon greedy policy
		"end_eps": 0.05,           # final epsilon in epsilon greedy policy 
		"eps_decay_period": 0.002, # how fast epsilon changes in iterations
		# Evaluation
		"eval_freq": 10000,        # every eval_freq steps we run the policy with epsilon = 0 to see how good it is without exploration (this is how we would apply it in the real world" 
		"eval_eps": 0.0,           # no exploration during evaluation (a.k.a real world execution)
		# Learning
		"discount": 0.95,
		"buffer_size": 500000,     # how many samples to keep in the DQN replay buffer. For CartPole, keep everything
		"batch_size": 32,          # mini-batch size. 32 for DQN is a reasonable choice (based on MANY experimental observations in several different environments
		"optimizer": "Adam",
		"optimizer_parameters": {
			"lr": 3e-4
		},
		"train_freq": 1,
		"polyak_target_update": True,
		"target_update_freq": 1,  # how often we update the parameters of the target network. If we use polyak averaging, for CartPole we can update the parameters in every iteration
		"tau": 0.005              # how the evaluation network and target network weights are mixed. I have never seen any other value ere.
	}

	# Load parameters
	parser = argparse.ArgumentParser()
	parser.add_argument("--env", default="CartPole-v0")     # OpenAI gym environment name  # Change Name if you play other gym Env. "MountainCar-v0", "CartPole-v0"
	parser.add_argument("--seed", default=0, type=int)             # Sets Gym, PyTorch and Numpy seeds
	parser.add_argument("--buffer_name", default="Default")        # Prepends name to filename
	parser.add_argument("--max_timesteps", default=5e4, type=int)  # Max time steps to run environment or train for   # for cartpole default=5e4, for mountain car 2e5
	parser.add_argument("--BCQ_threshold", default=0.3, type=float)# Threshold hyper-parameter for BCQ
	parser.add_argument("--low_noise_p", default=0.2, type=float)  # Probability of a low noise episode when generating buffer
	parser.add_argument("--rand_action_p", default=0.2, type=float)# Probability of taking a random action when generating buffer, during non-low noise episode
	parser.add_argument("--train_behavioral", action="store_true") # If true, train behavioral policy (If you read )
	parser.add_argument("--generate_buffer", action="store_false")  # If true, generate buffer
	parser.add_argument("--grid", action="store_true") # For Visualisation CartPole : If true, generate grid file as csv form 
	parser.add_argument("--viper", action="store_true") # For Viper Algorithm
	args = parser.parse_args()
	
	print("---------------------------------------")	
	if args.train_behavioral:
		print(f"Setting: Training behavioral, Env: {args.env}, Seed: {args.seed}")
	elif args.generate_buffer:
		print(f"Setting: Generating buffer, Env: {args.env}, Seed: {args.seed}")
	else:
		print(f"Setting: Training BCQ, Env: {args.env}, Seed: {args.seed}")
	print("---------------------------------------")

	if args.train_behavioral and args.generate_buffer:
		print("Train_behavioral and generate_buffer cannot both be true.")
		exit()

	if not os.path.exists("./results"):
		os.makedirs("./results")

	if not os.path.exists("./models"):
		os.makedirs("./models")

	if not os.path.exists("./buffers"):
		os.makedirs("./buffers")

	# Make env and determine properties
	env, is_atari, is_cartpole, state_dim, num_actions = utils.make_env(args.env, atari_preprocessing)
	if is_cartpole:
		env.length = constants.POLE_SIZE
	else:
		env.gravity = constants.GRAVITY
	env._max_episode_steps = constants.MAX_EPISODE_STEPS
	

	
	# Use 2 state dimensions, e.g. Pole Angle, Pole Angular
	if is_cartpole:
		state_dim = 2
	parameters = atari_parameters if is_atari else regular_parameters

	env.seed(args.seed)
	torch.manual_seed(args.seed)
	np.random.seed(args.seed)

	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

	# Initialize buffer
	replay_buffer = utils.ReplayBuffer(state_dim, is_atari, atari_preprocessing, parameters["batch_size"], parameters["buffer_size"], device)

	if args.train_behavioral or args.generate_buffer or args.grid or args.viper:
		interact_with_environment(env, replay_buffer, is_atari, is_cartpole, num_actions, state_dim, device, args, parameters)
	else:
		train_BCQ(env, replay_buffer, is_atari, is_cartpole, num_actions, state_dim, device, args, parameters)



