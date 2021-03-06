import copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import constants


# Used for Atari
class Conv_Q(nn.Module):
	def __init__(self, frames, num_actions):
		super(Conv_Q, self).__init__()
		self.c1 = nn.Conv2d(frames, 32, kernel_size=8, stride=4)
		self.c2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
		self.c3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)
		self.l1 = nn.Linear(3136, 512)
		self.l2 = nn.Linear(512, num_actions)


	def forward(self, state):
		q = F.relu(self.c1(state))
		q = F.relu(self.c2(q))
		q = F.relu(self.c3(q))
		q = F.relu(self.l1(q.reshape(-1, 3136)))
		return self.l2(q)


# This is a HUGE network, required for BCQ and offline RL. Let's reduce it to 256x256 and see how it goes

# Used for Box2D / Toy problems
class FC_Q(nn.Module):
	def __init__(self, state_dim, num_actions):
		super(FC_Q, self).__init__()
		self.l1 = nn.Linear(state_dim, 256)
		# self.l2 = nn.Linear(256, 256)  
		self.l3 = nn.Linear(256, num_actions)


	def forward(self, state):
		q = F.relu(self.l1(state))
		#  q = F.relu(self.l2(q))  
		return self.l3(q)


class DQN(object):
	def __init__(
		self, 
		is_atari,
		is_cartpole,
		num_actions,
		state_dim,
		device,
		discount=0.99,
		optimizer="Adam",
		optimizer_parameters={},
		polyak_target_update=False,
		target_update_frequency=8e3,
		tau=0.005,
		initial_eps = 1,
		end_eps = 0.001,
		eps_decay_period = 25e4,
		eval_eps=0.001,
	):
	
		self.device = device

		# Determine network type
		self.Q = Conv_Q(state_dim[0], num_actions).to(self.device) if is_atari else FC_Q(state_dim, num_actions).to(self.device)
		self.Q_target = copy.deepcopy(self.Q)
		self.Q_optimizer = getattr(torch.optim, optimizer)(self.Q.parameters(), **optimizer_parameters)
		self.is_cartpole = is_cartpole

		self.discount = discount

		# Target update rule
		self.maybe_update_target = self.polyak_target_update if polyak_target_update else self.copy_target_update
		self.target_update_frequency = target_update_frequency
		self.tau = tau

		# Decay for eps
		self.initial_eps = initial_eps
		self.end_eps = end_eps
		self.slope = (self.end_eps - self.initial_eps) / eps_decay_period

		# Evaluation hyper-parameters
		self.state_shape = (-1,) + state_dim if is_atari else (-1, state_dim)
		self.eval_eps = eval_eps
		self.num_actions = num_actions

		# Number of training iterations
		self.iterations = 0


	def select_action(self, state, eval=False):
		eps = self.eval_eps if eval \
			else max(self.slope * self.iterations + self.initial_eps, self.end_eps)

		# Select action according to policy with probability (1-eps)
		# otherwise, select random action
		if np.random.uniform(0,1) > eps:
			with torch.no_grad():
				state = torch.FloatTensor(state).reshape(self.state_shape).to(self.device)
				q_array_given_state = self.Q(state)
				chosen_action =q_array_given_state.argmax(1)
				return int(chosen_action), q_array_given_state
		else:
			return np.random.randint(self.num_actions), None

	# Local Lipschitzness Function

	def train(self, replay_buffer, epsilon=constants.EPSILON, perturb_steps=10, beta=constants.BETA, step_size=0.003):
		# Sample replay buffer
		state, action, next_state, reward, done = replay_buffer.sample()
		criterion_kl = nn.KLDivLoss(reduction='sum')

		# Compute the target Q value
		with torch.no_grad():
			q_value = self.Q_target(next_state).max(1, keepdim=True)[0]
			target_Q = reward + done * self.discount * q_value

		# Get current Q estimate
		current_Q = self.Q(state).gather(1, action) 

		# Apply Local Lipschitzness start:
		if beta > 0: # if beta = 0 local lip will not be computed so we can save time
			
			batch_size = len(current_Q)

			total_loss = 0

			step_size_unnorm = [0,0]
			step_size_unnorm[0] = step_size * (constants.XMAX - constants.XMIN)
			step_size_unnorm[1] = step_size * (constants.YMAX - constants.YMIN)

			epsilon_unnorm = [0,0]
			epsilon_unnorm[0] = epsilon * (constants.XMAX - constants.XMIN)
			epsilon_unnorm[1] = epsilon * (constants.YMAX - constants.YMIN)

			for i in range(batch_size):
				stateSingle = state[i]
				actionSingle = action[i]
				rnd = (0.001 * torch.randn(stateSingle.shape).to(self.device))
				rnd[0] = rnd[0] * (constants.XMAX - constants.XMIN)
				rnd[1] = rnd[1] * (constants.YMAX - constants.YMIN)
				x_adv = stateSingle.detach() + rnd
				#x_adv[0] = x_adv[0] + (0.001 * torch.randn(1).to(self.device) * (constants.XMAX - constants.XMIN) + constants.XMIN)
				#x_adv[1] = x_adv[1] + (0.001 * torch.randn(1).to(self.device) * (constants.YMAX - constants.YMIN) + constants.YMIN)

				for _ in range(perturb_steps):
					x_adv.requires_grad_(True)
					with torch.enable_grad():
						loss = criterion_kl(F.log_softmax(self.Q(x_adv)), F.softmax(self.Q(stateSingle)))
						#loss = self.local_lip(stateSingle, x_adv, actionSingle , 1, np.inf)
					grad = torch.autograd.grad(loss, [x_adv])[0]
					# renorming gradient
					eta = torch.sign(grad.detach())
					eta[0] = step_size_unnorm[0] * eta[0]
					eta[1] = step_size_unnorm[1] * eta[1] 
					x_adv = x_adv.data.detach() + eta.detach()
					x_adv[0] = torch.min(torch.max(x_adv[0], stateSingle[0] - epsilon_unnorm[0]), stateSingle[0] + epsilon_unnorm[0])
					x_adv[1] = torch.min(torch.max(x_adv[1], stateSingle[1] - epsilon_unnorm[1]), stateSingle[1] + epsilon_unnorm[1])
					x_adv[0] = torch.clamp(x_adv[0], constants.XMIN, constants.XMAX) 
					x_adv[1] = torch.clamp(x_adv[1], constants.YMIN, constants.YMAX) 

				v = Variable(x_adv, requires_grad=False)
				total_loss += criterion_kl(F.log_softmax(self.Q(x_adv)), F.softmax(self.Q(stateSingle)))

				# calculate robust loss
			loss_natural = F.smooth_l1_loss(current_Q, target_Q)
			loss_robust = total_loss / batch_size

			# Compute Q loss
			Q_loss = loss_natural + beta * loss_robust
		
		else:
			Q_loss = F.smooth_l1_loss(current_Q, target_Q) 
			
		# Apply Local Lipschitzness End

		# Optimize the Q
		self.Q_optimizer.zero_grad()
		Q_loss.backward()
		self.Q_optimizer.step()

		# Update target network by polyak or full copy every X iterations.
		self.iterations += 1
		self.maybe_update_target()

	
	def polyak_target_update(self):
		for param, target_param in zip(self.Q.parameters(), self.Q_target.parameters()):
		   target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)


	def copy_target_update(self):
		if self.iterations % self.target_update_frequency == 0:
			 self.Q_target.load_state_dict(self.Q.state_dict())


	def save(self, filename):
		torch.save(self.Q.state_dict(), filename + "_Q")
		torch.save(self.Q_optimizer.state_dict(), filename + "_optimizer")


	def load(self, filename):
		self.Q.load_state_dict(torch.load(filename + "_Q"))
		self.Q_target = copy.deepcopy(self.Q)
		self.Q_optimizer.load_state_dict(torch.load(filename + "_optimizer"))
