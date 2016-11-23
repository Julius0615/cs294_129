
# coding: utf-8

# # Deep Q-Learning
#
# In this notebook, you will implement a deep Q-Learning reinforcement algorithm. The implementation borrows ideas from both the original DeepMind Nature paper and the more recent asynchronous version:<br/>
# [1] "Human-Level Control through Deep Reinforcement Learning" by Mnih et al. 2015<br/>
# [2] "Asynchronous Methods for Deep Reinforcement Learning" by Mnih et al. 2016.<br/>
#
# In particular:
# * We use separate target and Q-functions estimators with periodic updates to the target estimator.
# * We use several concurrent "threads" rather than experience replay to generate less biased gradient updates.
# * Threads are actually synchronized so we start each one at a random number of moves.
# * We use an epsilon-greedy policy that blends random moves with policy moves.
# * We taper the random action parameter (epsilon) and the learning rate to zero during training.
#
# This gives a simple and reasonably fast general-purpose RL algorithm. We use it here for the Cartpole environment from OpenAI Gym, but it can easily be adapted to others. For this notebook, you will implement 4 steps:
#
# 1. The backward step for the Q-estimator
# 2. The $\epsilon$-greedy policy
# 3. "asynchronous" initialization
# 4. The Q-learning algorithm
#
# To get started, we import some prerequisites.

# In[1]:

# get_ipython().magic(u'matplotlib inline')

import gym
import numpy as np
import sys
import matplotlib.pyplot as plt
import time
import pickle


# The block below lists some parameters you can tune. They should be self-explanatory. They are currently set to train CartPole-V0 to a "solved" score (> 195) most of the time.

# In[2]:

nsteps = 10001                       # Number of steps to run (game actions per environment)
npar = 16                            # Number of parallel environments
target_window = 200                  # Interval to update target estimator from q-estimator
discount_factor = 0.99               # Reward discount factor
printsteps = 1000                    # Number of steps between printouts
render = False                       # Whether to render an environment while training

epsilon_start = 1.0                  # Parameters for epsilon-greedy policy: initial epsilon
epsilon_end = 0.0                    # Final epsilon
neps = int(0.8*nsteps)               # Number of steps to decay epsilon

learning_rate = 2e-3                 # Initial learning rate
lr_end = 0                           # Final learning rate
nlr = neps                           # Steps to decay learning rate
decay_rate = 0.99                    # Decay factor for RMSProp

nhidden = 200                        # Number of hidden layers for estimators

init_moves = 2000                    # Upper bound on random number of moves to take initially
nwindow = 2                          # Sensing window = last n images in a state


# Below are environment-specific parameters. The function "preprocess" should process an observation returned by the environment into a vector for training. For CartPole we simply append a 1 to implement bias in the first layer.
#
# For visual environments you would typically crop, downsample to 80x80, set color to a single bit (foreground/background), and flatten to a vector. That transformation is already implemented in the Policy Gradient code.
#
# *nfeats* is the dimension of the vector output by *preprocess*.

# In[3]:

game_type="CartPole-v0"                 # Model type and action definitions
VALID_ACTIONS = [0, 1]
nactions = len(VALID_ACTIONS)
nfeats = 5                              # There are four state features plus the constant we add

def preprocess(I):                      # preprocess each observation
    """Just append a 1 to the end"""
    return np.append(I.astype(float),1) # Add a constant feature for bias


# Here is the Q-estimator class. We use two instances of this class, one for the target estimator, and one for the Q-estimator. The Q function is normally represented as a scalar $Q(x,a)$ where $x$ is the state and $a$ is an action. For ease of implementation, we actually estimate a vector-valued function $Q(x,.)$ which returns the estimated reward for every action. The model here has just a single hidden layer:
#
# <pre>
# Input Layer (nfeats) => FC Layer => RELU => FC Layer => Output (naction values)
# </pre>
#
# ## 1. Implement Q-estimator gradient
# Your first task is to implement the
# <pre>Estimator.gradient(s, a, y)</pre>
# method for this class. **gradient** should compute the gradients wrt weight arrays W1 and W2 into
# <pre>self.grad['W1']
# self.grad['W2']</pre>
# respectively. Both <code>a</code> and <code>y</code> are vectors. Be sure to update only the output layer weights corresponding to the given action vector.

# In[4]:

class Estimator():

    def __init__(self, ninputs, nhidden, nactions):
        """ Create model matrices, and gradient and squared gradient buffers"""
        model = {}
        model['W1'] = np.random.randn(nhidden, ninputs) / np.sqrt(ninputs)   # "Xavier" initialization
        model['W2'] = np.random.randn(nactions, nhidden) / np.sqrt(nhidden)
        self.model = model
        self.grad = { k : np.zeros_like(v) for k,v in model.iteritems() }
        self.gradsq = { k : np.zeros_like(v) for k,v in model.iteritems() }


    def forward(self, s):
        """ Run the model forward given a state as input.
        returns action predictions and the hidden state"""
        h = np.dot(self.model['W1'], s)
        h[h<0] = 0 # ReLU nonlinearity
        rew = np.dot(self.model['W2'], h)
        return rew, h


    def predict(self, s):
        """ Predict the action rewards from a given input state"""
        rew, h = self.forward(s)
        return rew


    def gradient(self, s, a, y):
        """ Given a state s, action a and target y, compute the model gradients"""
        ##################################################################################
        ##                                                                              ##
        ## TODO: Compute gradients and return a scalar loss on a minibatch of size npar ##
        ##    s is the input state matrix (ninputs x npar).                             ##
        ##    a is an action vector (npar,).                                            ##
        ##    y is a vector of target values (npar,) corresponding to those actions.    ##
        ##    return: the loss per sample (npar,).                                      ##
        ##                                                                              ##
        ## Notes:                                                                       ##
        ##    * If the action is ai in [0,...,nactions-1], backprop only through the    ##
        ##      ai'th output.                                                           ##
        ##    * loss should be L2, and we recommend you normalize it to a per-input     ##
        ##      value, i.e. return L2(target,predition)/sqrt(npar).                     ##
        ##    * save the gradients in self.grad['W1'] and self.grad['W2'].              ##
        ##    * update self.grad['W1'] and self.grad['W2'] by adding the gradients, so  ##
        ##      that multiple gradient steps can be used beteween updates.              ##
        ##                                                                              ##
        ##################################################################################
        N = s.shape[0]
        y_, h = self.forward(s)
        loss = (y_[a, xrange(npar)] - y) ** 2 / np.sqrt(npar)
        d_out = np.zeros_like(y_)  # n_actions x npar
        d_out[a, xrange(npar)] = 0.5 * (y_[a, xrange(npar)] - y) / np.sqrt(npar)
        self.grad['W2'] = -np.dot(d_out, h.T)
        d_h = np.dot(self.model['W2'].T, d_out)
        d_h[h <= 0] = 0
        self.grad['W1'] = -np.dot(d_h, s.T)

        return loss


    def rmsprop(self, learning_rate, decay_rate):
        """ Perform model updates from the gradients using RMSprop"""
        for k in self.model:
            g = self.grad[k]
            self.gradsq[k] = decay_rate * self.gradsq[k] + (1 - decay_rate) * g*g
            self.model[k] += learning_rate * g / (np.sqrt(self.gradsq[k]) + 1e-5)
            self.grad[k].fill(0.0)
            
            


# ## 2. Implement $\epsilon$-Greedy Policy
#
# An $\epsilon$-Greedy policy should:
# * with probability $\epsilon$ take a uniformly-random action.
# * otherwise choose the best action according to the estimator from the given state.
#
# The function below should implement this policy. It should return a matrix A of size (nactions, npar) such that A[i,j] is the probability of taking action i on input j. The probabilities of non-optimal actions should be $\epsilon/{\rm nactions}$ and the probability of the best action should be $1-\epsilon+\epsilon/{\rm nactions}$.
#
# Since the function processes batches of states, the input <code>state</code> is a <code>ninputs x npar</code> matrix, and the returned value should be a <code>nactions x npar</code> matrix.

# In[5]:

def policy(estimator, state, epsilon):
    """ Take an estimator and state and predict the best action.
    For each input state, return a vector of action probabilities according to an epsilon-greedy policy"""
    ##################################################################################
    ##                                                                              ##
    ## TODO: Implement an epsilon-greedy policy                                     ##
    ##       estimator: is the estimator to use (instance of Estimator)             ##
    ##       state is an (ninputs x npar) state matrix                              ##
    ##       epsilon is the scalar policy parameter                                 ##
    ## return: an (nactions x npar) matrix A where A[i,j] is the probability of     ##
    ##       taking action i on input j.                                            ##
    ##                                                                              ##
    ## Use the definition of epsilon-greedy from the cell above.                    ##
    ##                                                                              ##
    ##################################################################################

    A = np.ones((nactions, npar)) * epsilon / nactions
    rew = estimator.predict(state)
    actions = np.argmax(rew, axis=0)
    A[actions, xrange(npar)] += 1 - epsilon
    A = A / np.sum(A, axis=0).reshape(1, -1)
    return A

def sample_actions(action_probs):
    actions = []
    for i in xrange(npar):
        try:
            actions.append(np.random.choice(nactions, p=action_probs[:, i]))
        except ValueError:
            import pdb; pdb.set_trace()
    return np.array(actions, dtype=int)

def expected_rewards(estimator, state, epsilon):
    A = policy(estimator, state, epsilon)
    rewards = estimator.predict(state)
    return np.sum(A * rewards, axis=0)


# This routine copies the state of one estimator into another. Its used to update the target estimator from the Q-estimator.

# In[6]:

def update_estimator(to_estimator, from_estimator, window, istep):
    """ every <window> steps, Copy model state from from_estimator into to_estimator"""
    if (istep % window == 0):
        for k in from_estimator.model:
            np.copyto(to_estimator.model[k], from_estimator.model[k])


# ## 3. Implement "Asynchronous Threads"
#
# Don't try that in Python!! Actually all we do here is create an array of environments and advance each one a random number of steps, using random actions at each step. Later on we will make *synchronous* updates to all the environments, but the environments (and their gradient updates) should remain uncorrelated. This serves the same goal as asynchronous updates in paper [2], or experience replay in paper [1].

# In[7]:

block_reward = 0.0;
total_epochs = 0;

# Create estimators
q_estimator = Estimator(nfeats*nwindow, nhidden, nactions)
target_estimator = Estimator(nfeats*nwindow, nhidden, nactions)

# The epsilon and learning rate decay schedules
epsilons = np.linspace(epsilon_start, epsilon_end, neps)
learning_rates = np.linspace(learning_rate, lr_end, nlr)

# Initialize the games
print("Initializing games..."); sys.stdout.flush()
envs = np.empty(npar, dtype=object)
state = np.zeros([nfeats * nwindow, npar], dtype=float)
rewards = np.zeros([npar], dtype=float)
dones = np.empty(npar, dtype=int)
actions = np.zeros([npar], dtype=int)


def append_state(all_states, obs, env_idx):
    all_states = all_states.copy()
    env_states = all_states[:, env_idx].reshape(nfeats, nwindow)
    env_states[:, :nwindow - 1] = env_states[:, 1:]
    env_states[:, nwindow - 1] = obs
    all_states[:, env_idx] = env_states.flatten()
    return all_states

def reset_state(all_states, env_idx):
    all_states = all_states.copy()
    all_states[:, env_idx] = 0
    return all_states


for i in range(npar):
    envs[i] = gym.make(game_type)
    for j in xrange(np.random.randint(nwindow, init_moves + 1)):
        act = np.random.randint(0, nactions)
        actions[i] = act
        obs, reward, done, info = envs[i].step(act)
        obs = preprocess(obs)
        block_reward += reward
        state = append_state(state, obs, i)
        if done:
            obs = envs[i].reset()
            obs = preprocess(obs)
            state = reset_state(state, i)
            state = append_state(state, obs, i)
            total_epochs += 1
    dones[i] = False



    ##################################################################################
    ##                                                                              ##
    ## TODO: Advance each environment by a random number of steps, where the number ##
    ##       of steps is sampled uniformly from [nwindow, init_moves].              ##
    ##       Use random steps to advance.                                           ##
    ##                                                                              ##
    ## Update the total reward and total epochs variables as you go.                ##
    ## If an environment returns done=True, reset it and increment the epoch count. ##
    ##                                                                              ##
    ##################################################################################



# ## 4. Implement Deep Q-Learning
# In this cell you actually implement the algorithm. We've given you comments to define all the steps. You should also add book-keeping steps to keep track of the loss, reward and number of epochs (where env.step() returns done = true).

# In[9]:

t0 = time.time()
block_loss = 0.0
last_epochs=0

for istep in np.arange(nsteps):
    if (render): envs[0].render()

    #########################################################################
    ## TODO: Implement Q-Learning                                          ##
    ##                                                                     ##
    ## At high level, your code should:                                    ##
    ## * Update epsilon and learning rate.                                 ##
    ## * Update target estimator from Q-estimator if needed.               ##
    ## * Get the next action probabilities for the minibatch by running    ##
    ##   the policy on the current state with the Q-estimator.             ##
    ## * Then for each environment:                                        ##
    ##     ** Pick an action according to the action probabilities.        ##
    ##     ** Step in the gym with that action.                            ##
    ##     ** Process the observation and concat it to the last nwindow-1  ##
    ##        processed observations to form a new state.                  ##
    ## Then for all environments (vectorized):                             ##
    ## * Predict Q-scores for the new state using the target estimator.    ##
    ## * Compute new expected rewards using those Q-scores.                ##
    ## * Using those expected rewards as a target, compute gradients and   ##
    ##   update the Q-estimator.                                           ##
    ## * Step to the new state.                                            ##
    ##                                                                     ##
    #########################################################################
    current_eps = epsilons[int(1.0 * istep / nsteps * neps)]
    current_lr = learning_rates[int(1.0 * istep / nsteps * nlr)]

    update_estimator(target_estimator, q_estimator, target_window, istep)

    old_state = state.copy()

    action_probs = policy(q_estimator, state, current_eps)
    actions = sample_actions(action_probs)

    for i in range(npar):
        if dones[i]:
            obs = envs[i].reset()
            obs = preprocess(obs)
            state = reset_state(state, i)
            state = append_state(state, obs, i)
            total_epochs += 1
            rewards[i] = 0

        obs, reward, done, info = envs[i].step(actions[i])
        dones[i] = done

        rewards[i] = reward

        obs = preprocess(obs)
        state = append_state(state, obs, i)

#     import pdb; pdb.set_trace()
    expected_new_rewards = expected_rewards(target_estimator, state, current_eps)
    expected_new_rewards = expected_new_rewards * np.logical_not(done)
    target_q = expected_new_rewards * discount_factor + rewards
    loss = q_estimator.gradient(old_state, actions, target_q)
    block_loss = np.sum(loss)
    q_estimator.rmsprop(current_lr, decay_rate)
    block_reward = np.sum(rewards)





    t = time.time() - t0
    if (istep % printsteps == 0):
        print("step {:0d}, time {:.1f}, loss {:.8f}, epochs {:0d}, reward/epoch {:.5f}".format(
                istep, t, block_loss/printsteps, total_epochs, block_reward/np.maximum(1,total_epochs-last_epochs)))
        last_epochs = total_epochs
        block_reward = 0.0
        block_loss = 0.0



# exit(0)

# Let's save the model now.

# In[12]:

pickle.dump(q_estimator.model, open("cartpole_q_estimator.p", "wb"))


# You can reload the model later if needed:

# In[13]:

test_estimator = Estimator(nfeats*nwindow, nhidden, nactions)
test_estimator.model = pickle.load(open("cartpole_q_estimator.p", "rb"))


# And animate the model's performance.

# In[14]:

state0 = state[:,0]
for i in np.arange(200):
    envs[0].render()
    preds = test_estimator.predict(state0)
    iaction = np.argmax(preds)
    obs, _, done0, _ = envs[0].step(VALID_ACTIONS[iaction])
    state0 = np.concatenate((state0[nfeats:], preprocess(obs)))
    if (done0): envs[0].reset()



# So there we have it. Simple 1-step Q-Learning can solve easy problems very fast. Note that environments that produce images will be much slower to train on than environments (like CartPole) which return an observation of the state of the system. But this model can still train on those image-based games - like Atari games. It will take hours-days however. It you try training on visual environments, we recommend you run the most expensive step - rmsprop - less often (e.g. every 10 iterations). This gives about a 3x speedup.

# ## Optional
# Do **one** of the following tasks:
# * Adapt the DQN algorithm to another environment - it can use direct state observations.  Call <code>env.get_action_meanings()</code> to find out what actions are allowed. Summarize training performance: your final average reward/epoch, the number of steps required to train, and any modifications to the model or its parameters that you made.
# * Try smarter schedules for epsilon and learning rate. Rewards for CartPole increase very sharply (several orders of magnitude) with better policies, especially as epsilon --> 0. Gradients will also change drastically, so the initial learning rate is probably not good later on. Try schedules for decreasing epsilon that allow the model to better adapt. Try other learning rate schedules, or setting learning rate based on average reward.
# * Try a fancier model. e.g. add another hidden layer, or try sigmoid non-linearities.
