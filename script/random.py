from torch import nn
import torch.nn.functional as F
import torch
from torch.distributions import Categorical, Normal
# import gymnasium as gym
# from gymnasium.spaces import Box, Discrete
import gym
from gym.spaces import Box, Discrete
from tqdm import tqdm
import numpy as np
import scipy
import wandb

import os
import datetime
import random
from gymnasium.wrappers.record_video import RecordVideo
from gymnasium.wrappers import NormalizeObservation

from environments import hopper,hopper_v3,walker2d,swimmer,half_cheetah,ant,humanoid
from util import compute_hv,compute_hypervolume,compute_sparsity,crowd_dist

import argparse

def discount_cumsum(x, discount):
    """
    magic from rllab for computing discounted cumulative sums of vectors.

    input: 
        vector x, 
        [x0, 
         x1, 
         x2]

    output:
        [x0 + discount * x1 + discount^2 * x2,  
         x1 + discount * x2,
         x2]
    """
    return scipy.signal.lfilter([1], [1, float(-discount)], x[::-1], axis=0)[::-1]


def combined_shape(length, shape=None):
    """
    Helper function makes sure the shape of experience is correct for the buffer

    Args:
        length (int): _description_
        shape (tuple[int,int], optional): _description_. Defaults to None.

    Returns:
        tuple[int,int]: correct shape
    """
    if shape is None:
        return (length,)
    return (length, shape) if np.isscalar(shape) else (length, *shape)

# TODO: This buffer cannot recompute GAE. Maybe change it in the future
class PPOBuffer():
    """
    A buffer to store the rollout experience from OpenAI spinningup
    """
    def __init__(self, observation_dim, action_dim, weight_dim, capacity, gamma, lam):
        self.obs_buf = np.zeros(combined_shape(capacity, observation_dim), dtype=np.float32)
        self.weight_buf = np.zeros(combined_shape(capacity, weight_dim), dtype=np.float32)
        self.act_buf = np.zeros(combined_shape(capacity, action_dim), dtype=np.float32)
        self.adv_buf = np.zeros(capacity, dtype=np.float32)
        self.rew_buf = np.zeros(capacity, dtype=np.float32)
        self.rtg_buf = np.zeros(capacity, dtype=np.float32)
        self.val_buf = np.zeros(capacity, dtype=np.float32)
        self.logp_buf = np.zeros(capacity, dtype=np.float32)
        self.capacity = capacity
        self.idx = 0
        self.path_idx = 0
        self.gamma = gamma
        self.lam = lam

    def push(self, obs, w, act, rew, val, logp):
        assert self.idx < self.capacity
        self.obs_buf[self.idx] = obs
        self.weight_buf[self.idx] = w
        self.act_buf[self.idx] = act
        self.rew_buf[self.idx] = rew
        self.val_buf[self.idx] = val
        self.logp_buf[self.idx] = logp

        self.idx += 1

    def GAE_cal(self, last_val):
        """Calculate the GAE when an episode is ended

        Args:
            last_val (int): last state value, it is zero when the episode is terminated.
            it's v(s_{t+1}) when the state truncate at t.
        """
        path_slice = slice(self.path_idx, self.idx)
        # to make the deltas the same dim
        rewards = np.append(self.rew_buf[path_slice], last_val)
        vals = np.append(self.val_buf[path_slice], last_val)

        deltas = rewards[:-1] + self.gamma * vals[1:] - vals[:-1]
        self.adv_buf[path_slice] = discount_cumsum(deltas, self.gamma * self.lam)

        ### OpenAI spinning up implemetation comment: No ideal, big value loss when episode rewards are large
        # self.rtg_buf[path_slice] = discount_cumsum(rewards, self.gamma)[:-1]

        ### OpenAI stable_baseline3 implementation
        ### in David Silver Lecture 4: https://www.youtube.com/watch?v=PnHCvfgC_ZA
        ### TD(lambda) estimator, see "Telescoping in TD(lambda)"
        self.rtg_buf[path_slice] = self.adv_buf[path_slice] + self.val_buf[path_slice]
        
        self.path_idx = self.idx

                
    def sample(self, minibatch_size, device):
        """This method sample a list of minibatches from the memory

        Args:
            minibatch_size (int): size of minibatch, usually 2^n
            device (object): CPU or GPU

        Returns:
            list: a list of minibatches
        """
        assert self.idx == self.capacity, f'The buffer is not full, \
              self.idx:{self.idx} and self.capacity:{self.capacity}'
        # normalise advantage
        self.adv_buf = (self.adv_buf - np.mean(self.adv_buf)) / (np.std(self.adv_buf) + 1e-8)
        
        inds = np.arange(self.capacity)
        
        np.random.shuffle(inds)
        
        data = []
        for start in range(0, self.capacity, minibatch_size):
            end = start + minibatch_size
            minibatch_inds = inds[start:end]
            minibatch = dict(obs=self.obs_buf[minibatch_inds], w=self.weight_buf[minibatch_inds], act=self.act_buf[minibatch_inds], \
                             rtg=self.rtg_buf[minibatch_inds], adv=self.adv_buf[minibatch_inds], \
                             logp=self.logp_buf[minibatch_inds])
            data.append({k: torch.as_tensor(v, dtype=torch.float32, device=device) for k,v in minibatch.items()})
        
        return data
    
    def reset(self):
        # reset the index
        self.idx, self.path_idx = 0, 0

def layer_init(layer, std=np.sqrt(2)):
    """Init the weights as the stable baseline3 so the performance is comparable.
       But it is not the ideal way to initialise the weights.

    Args:
        layer (_type_): layers
        std (_type_, optional): standard deviation. Defaults to np.sqrt(2).

    Returns:
        _type_: layers after init
    """
    nn.init.orthogonal_(layer.weight, std)
    nn.init.constant_(layer.bias, 0.0)
    return layer

class Actor_Net(nn.Module):
    def __init__(self, n_observations,n_weights, n_actions,  num_cells, continous_action, log_std_init=0.0):
        super(Actor_Net,self).__init__()
        self.factor = 1
        
        # self.layer1 = layer_init(nn.Linear(n_observations, num_cells))
        self.layer1 = layer_init(nn.Linear(n_observations+n_weights, num_cells))

        self.layer1_ = layer_init(nn.Linear(n_weights, num_cells))
        # self.layer2 = layer_init(nn.Linear(num_cells*2, num_cells*self.factor))
        self.layer2 = layer_init(nn.Linear(num_cells, num_cells*self.factor))

        # self.layer3 = layer_init(nn.Linear(num_cells*self.factor, num_cells*self.factor))

        self.layer4 = layer_init(nn.Linear(num_cells*self.factor+n_weights, n_actions), std=0.01)
        # self.layer4 = layer_init(nn.Linear(num_cells*self.factor+n_weights, n_actions), std=0.01)
        # self.layer4 = layer_init(nn.Linear(num_cells*self.factor+num_cells, n_actions), std=0.01)

        self.continous_action = continous_action
        self.action_dim = n_actions
        
        if self.continous_action:
            log_std = log_std_init * np.ones(self.action_dim, dtype=np.float32)
            # Add it to the list of parameters
            self.log_std = torch.nn.Parameter(torch.as_tensor(log_std), requires_grad=True)
            #
            ### https://iclr-blog-track.github.io/2022/03/25/ppo-implementation-details/  implementation
            # self.log_std = nn.Parameter(torch.zeros(1, self.action_dim))  

            ### Stable-baseline3 implementation
            # self.log_std = nn.Parameter(torch.ones(self.action_dim) * log_std_init, requires_grad=False)      

        

    def forward(self, x, w):
        # x = torch.cat((nn.Tanh()(self.layer1(x)), nn.Tanh()(self.layer1_(w))), dim=1)
        x = torch.cat((x,w),dim=1)
        x = nn.Tanh()(self.layer1(x))
        activation2 = nn.Tanh()(self.layer2(x))

        # activation3 = nn.Tanh()(self.layer3(activation2))

        residual = torch.cat((activation2, w), dim=1)
   
        activation4 = self.layer4(residual)

        # activation4 = self.layer4(activation2)
        return activation4
    
    def act(self, x, w):
        if self.continous_action:
            mu = self.forward(x, w)
            std = torch.exp(self.log_std)
            dist = Normal(mu, std)
        else:
            log_probs = F.log_softmax(self.forward(x, w), dim=1)
            dist = Categorical(log_probs)
    
        action = dist.sample()
        if self.continous_action:
            action_logprob = dist.log_prob(action).sum(axis=-1)
        else:
            action_logprob = dist.log_prob(action)

        return action, action_logprob
    
    def logprob_ent_from_state_acton(self, x, w, act):
        if self.continous_action:
            mu = self.forward(x, w)
            std = torch.exp(self.log_std)
            dist = Normal(mu, std)
            # sum term is crucial to reduce dimension, otherwise the ratio = torch.exp(logp - logp_old) will have wrong result with boardcasting
            act_logp = dist.log_prob(act).sum(axis=-1) 
        else:
            dist = Categorical(F.softmax(self.forward(x, w)))
            act_logp = dist.log_prob(act)
        entropy = dist.entropy()
        
        return entropy, act_logp
    
   
class Critic_Net(nn.Module):
    def __init__(self, n_observations, n_weights, num_cells):
        super(Critic_Net,self).__init__()
        self.factor = 1
        self.layer1 = layer_init(nn.Linear(n_observations+n_weights, num_cells))
        # self.layer1 = layer_init(nn.Linear(n_observations, num_cells))
        self.layer1_ = layer_init(nn.Linear(n_weights, num_cells))
        # self.layer2 = layer_init(nn.Linear(num_cells*2, num_cells*self.factor))
        self.layer2 = layer_init(nn.Linear(num_cells, num_cells*self.factor))

        # self.layer3 = layer_init(nn.Linear(num_cells*self.factor, num_cells*self.factor))

        self.layer4 = layer_init(nn.Linear(num_cells*self.factor+n_weights, 1), std=1.0)
        # self.layer4 = layer_init(nn.Linear(num_cells*self.factor+n_weights, 1), std=1.0)
        # self.layer4 = layer_init(nn.Linear(num_cells*self.factor+num_cells, 1), std=1.0)

    def forward(self, x, w):
        # x = torch.cat((nn.Tanh()(self.layer1(x)), nn.Tanh()(self.layer1_(w))), dim=1)
        x = torch.cat((x,w),dim=1)

        x = nn.Tanh()(self.layer1(x))
        activation2 = nn.Tanh()(self.layer2(x))
        # activation3 = self.layer3(activation2)

        residual = torch.cat((activation2, w), dim=1)
        activation4 = self.layer4(residual)

        # activation4 = self.layer4(activation2)

        return activation4


class Actor_Critic_net(nn.Module):
    def __init__(self, obs_dim, weight_dim, act_dim, hidden_dim, continous_action, parameters_hardshare, log_std_init=0.0):

        super(Actor_Critic_net, self).__init__()

        self.parameters_hardshare = parameters_hardshare
        self.continous_action = continous_action
        self.action_dim = act_dim
        self.weight_dim = weight_dim

        if self.parameters_hardshare:
            self.layer1 = layer_init(nn.Linear(obs_dim, hidden_dim))
            self.layer1_ = layer_init(nn.Linear(weight_dim, hidden_dim))
            self.layer2 = layer_init(nn.Linear(hidden_dim*2, hidden_dim))

            self.actor_head = layer_init(nn.Linear(hidden_dim+hidden_dim, act_dim), std=0.01)
            self.critic_head = layer_init(nn.Linear(hidden_dim, 1), std=1.0)
            if self.continous_action:
                log_std = log_std_init * np.ones(self.action_dim, dtype=np.float32)
                # Add it to the list of parameters
                self.log_std = torch.nn.Parameter(torch.as_tensor(log_std), requires_grad=True)
                #
                ### https://iclr-blog-track.github.io/2022/03/25/ppo-implementation-details/  implementation
                # self.log_std = nn.Parameter(torch.zeros(1, self.action_dim))  

                ### Stable-baseline3 implementation
                # self.log_std = nn.Parameter(torch.ones(self.act_dim) * log_std_init, requires_grad=False) 

        else:
            self.actor = Actor_Net(obs_dim, weight_dim, act_dim, hidden_dim, continous_action)
            self.critic = Critic_Net(obs_dim, weight_dim, hidden_dim)


    def forward(self, x, w):
        if self.parameters_hardshare:
            x = torch.cat((self.layer1(x), self.layer1_(w)), dim=1)
            
            activation2 = nn.Tanh()(self.layer2(x))
            residual = torch.cat((activation2, self.layer1_(w)), dim=1)
            actor_logits = self.actor_head(residual)
            value = self.critic_head(activation2)
        else:
            actor_logits = self.actor.forward(x,w)
            value = self.critic.forward(x,w)

        return actor_logits, value

    def get_value(self, x, w):
        return self.critic(x, w).item()

    
    def act(self, x, w):
        """act with a state

        Args:
            x (_type_): state from the environment

        Returns:
            action: action according to the state
            action_logprob: the log probability to sample the action
            value: the state value
        """
        if self.continous_action:
            mu, value = self.forward(x,w)
            log_std = self.log_std if self.parameters_hardshare else self.actor.log_std
            std = torch.exp(log_std)
            dist = Normal(mu, std)
        else:
            actor_logits, value = self.forward(x,w)
            log_probs = F.log_softmax(actor_logits, dim=1)
            dist = Categorical(log_probs)

        action = dist.sample()
        if self.continous_action:
            action_logprob = dist.log_prob(action).sum(axis=-1)
        else:
            action_logprob = dist.log_prob(action)
        

        return action, action_logprob, value  

    def logprob_ent_from_state_acton(self, x, w, action):
        """Return the entropy, log probability of the selected action and state value

        Args:
            x (_type_): state from the environment
            action (_type_): action

        Returns:
            entropy: entropy from the distribution that the action is sampled from
            action_logprob: the log probability to sample the action
            value: the state value
        """

        if self.continous_action:
            mu, value = self.forward(x, w)
            log_std = self.log_std if self.parameters_hardshare else self.actor.log_std
            std = torch.exp(log_std)
            dist = Normal(mu, std)
            ### sum in log space is equivalent to multiplication in probability space
            ### Pr(a_1, a_2) = Pr(a_1)*Pr(a_2) given a_1 and a_2 are independent sampled
            action_logp = dist.log_prob(action).sum(axis=-1) 
        else:
            actor_logits, value = self.forward(x, w)
            log_probs = F.log_softmax(actor_logits, dim=1)
            dist = Categorical(log_probs)
            action_logp = dist.log_prob(action)
        entropy = dist.entropy().sum(axis=-1)
        
        return entropy, action_logp, value
class PPO():
    def __init__(self, id ,gamma, lamb, eps_clip, K_epochs, \
                 observation_space, weight_dim,action_space, num_cells, \
                 actor_lr, critic_lr, memory_size , minibatch_size,\
                 max_training_iter, cal_total_loss, c1, c2, \
                 early_stop, kl_threshold, parameters_hardshare, \
                 max_grad_norm , device
                 ):
        """Init

        Args:
            gamma (float): discount factor of future value
            lamb (float): lambda factor from GAE from 0 to 1
            eps_clip (float): clip range, usually 0.2
            K_epochs (in): how many times learn from one batch
            action_space (tuple[int, int]): action space of environment
            num_cells (int): how many cells per hidden layer
            critic_lr (float): learning rate of the critic
            memory_size (int): the size of rollout buffer
            minibatch_size (int): minibatch size
            cal_total_loss (bool): add entropy loss to the actor loss or not
            c1 (float): coefficient for value loss
            c2 (float): coefficient for entropy loss
            kl_threshold (float): approx kl divergence, use for early stop
            parameters_hardshare (bool): whether to share the first two layers of actor and critic
            device (_type_): tf device

        """
        self.id = id
        self.gamma = gamma
        self.lamb = lamb
        self.eps_clip = eps_clip
        self.K_epochs = K_epochs
        self.max_training_iter = max_training_iter

        self.observation_space = observation_space
        self.weight_dim = weight_dim
        self.action_space = action_space
        self.memory_size = memory_size
        self.minibatch_size = minibatch_size
        
        self.cal_total_loss = cal_total_loss
        self.c1 = c1
        self.c2 = c2
        self.entropy_coef = self.c2
        self.actor_lr = actor_lr
        self.critic_lr = critic_lr
        self.early_stop = early_stop
        self.kl_threshold = kl_threshold

        self.parameters_hardshare = parameters_hardshare
        self.episode_count = 1
        self.max_grad_norm = max_grad_norm
        self.global_step = 0

        self._last_obs = None
        self._episode_reward = 0
        self._early_stop_count = 0

        if isinstance(action_space, Box):
            self.continous_action = True
        elif isinstance(action_space, Discrete):
            self.continous_action = False
        else:
            raise AssertionError(f"action space is not valid {action_space}")


        self.observtion_dim = observation_space.shape[0]

        self.actor_critic = Actor_Critic_net(self.observtion_dim, self.weight_dim,\
                               action_space.shape[0] if self.continous_action else action_space.n, \
                                  num_cells, self.continous_action, parameters_hardshare).to(device)

        if parameters_hardshare:
            ### eps=1e-5 follows stable-baseline3
            self.actor_critic_opt = torch.optim.Adam(self.actor_critic.parameters(), lr=actor_lr, eps=1e-5)
            
        else:
            self.actor_critic_opt = torch.optim.Adam([ 
                {'params': self.actor_critic.actor.parameters(), 'lr': actor_lr, 'eps' : 1e-5},
                {'params': self.actor_critic.critic.parameters(), 'lr': critic_lr, 'eps' : 1e-5} 
            ])


        self.memory = PPOBuffer(observation_space.shape, action_space.shape,weight_dim, memory_size, gamma, lamb)

        self.device = device
        
        # These two lines monitor the weights and gradients
        wandb.watch(self.actor_critic.actor, log='all', log_freq=1000, idx=self.id*2 + 1)
        wandb.watch(self.actor_critic.critic, log='all', log_freq=1000, idx=self.id*2 + 2)

    def roll_out(self, env,w):
        """rollout for experience

        Args:
            env (gymnasium.Env): environment from gymnasium
        """
        
        
        assert self._last_obs is not None, "No previous observation"
        
        action_shape = env.action_space.shape
        # Run the policy for T timestep
        for i in range(self.memory_size):
            truncated = False  
            with torch.no_grad():
                obs_tensor = torch.tensor(self._last_obs, \
                                        dtype=torch.float32, device=self.device).unsqueeze(0)
                w_tensor = torch.tensor(w, \
                                        dtype=torch.float32, device=self.device).unsqueeze(0)
            
                action, action_logprob, value = self.actor_critic.act(obs_tensor,w_tensor)
            
            action = action.cpu().numpy().reshape(action_shape)

            action_logprob = action_logprob.item()

            value = value.item()

            ### Clipping actions when they are reals is important
            clipped_action = action

            if self.continous_action:
                clipped_action = np.clip(action, self.action_space.low, self.action_space.high)

            # next_obs, reward, terminated, truncated, info = env.step(clipped_action)
            next_obs, reward_, terminated, obj = env.step(clipped_action)
            if weight_dim ==2:
                r_obj1,r_obj2,r_obj3,reward = obj['obj'][0],obj['obj'][1],-999,w[0]*obj['obj'][0]+w[1]*obj['obj'][1]
            elif weight_dim ==3:
                _obj1,r_obj2,r_obj3,reward = obj['obj'][0],obj['obj'][1],obj['obj'][2],w[0]*obj['obj'][0]+w[1]*obj['obj'][1]+w[2]*obj['obj'][2]

            

            self.global_step += 1

            self.memory.push(self._last_obs, w, action, reward, value, action_logprob)

            self._last_obs = next_obs

            self._episode_reward += reward

            self.episode_count += 1
            

            if self.episode_count%1000 == 0:
                terminated = True


            if terminated or truncated:
                self.episode_count = 0
                if truncated:
                    with torch.no_grad():
                        last_value = self.actor_critic.get_value(torch.tensor(next_obs, dtype=torch.float32, device=self.device).unsqueeze(0),torch.tensor(w, dtype=torch.float32, device=self.device).unsqueeze(0))
                else:
                    last_value = 0
                
                self.memory.GAE_cal(last_value)

                # gymnasium
                # self._last_obs, _ = env.reset()
                # old gym      
                self._last_obs= env.reset()
                
                wandb.define_metric(f'policy_{self.id}_episode_reward', step_metric="custom_step")
                log_dict = {
                f'policy_{self.id}_episode_reward': self._episode_reward,
                "custom_step": self.global_step
                }
                wandb.log(log_dict)
                # wandb.log({f'policy_{self.id}_episode_reward': self._episode_reward,"custom_step": self.global_step})
                # wandb.log({'episode_reward' : self._episode_reward}, step=self.global_step)

                self._episode_reward = 0

        with torch.no_grad():
            last_value = self.actor_critic.get_value(torch.tensor(next_obs, dtype=torch.float32, device=self.device).unsqueeze(0),torch.tensor(w, dtype=torch.float32, device=self.device).unsqueeze(0))
        self.memory.GAE_cal(last_value)


    def evaluate_recording(self, env, eval_weights):
        
        env_name = env.spec.id

        video_folder = os.path.join(wandb.run.dir, 'videos')

        env = RecordVideo(env, video_folder, name_prefix=env_name)
        for w in eval_weights:

            # gymnasium
            # obs, _ = env.reset()
            # old gym      
            obs= env.reset()

            done = False

            action_shape = env.action_space.shape

            

            while not done:
                obs_tensor = torch.tensor(obs, \
                                        dtype=torch.float32, device=self.device).unsqueeze(0)
                w_tensor = torch.tensor(w, \
                                        dtype=torch.float32, device=self.device).unsqueeze(0)
                with torch.no_grad():
                    action, _, _ = self.actor_critic.act(obs_tensor,w_tensor)

                action = action.cpu().numpy()
                action = action.reshape(action_shape)
                # next_obs, reward, terminated, truncated, _ = env.step(action)
                next_obs, reward, terminated, obj = env.step(action)
                done = terminated 
                obs = next_obs

            
            mp4_files = [file for file in os.listdir(video_folder) if file.endswith(".mp4")]

            for mp4_file in mp4_files:
                wandb.log({'Episode_recording': wandb.Video(os.path.join(video_folder, mp4_file))})

            env.close()
            

            


            

    def compute_loss(self, data):
        """compute the loss of state value, policy and entropy

        Args:
            data (List[Dict]): minibatch with experience

        Returns:
            actor_loss : policy loss
            critic_loss : value loss
            entropy_loss : mean entropy of action distribution
        """
        observations, weights, actions, logp_old = data['obs'], data['w'], data['act'], data['logp']
        advs, rtgs = data['adv'], data['rtg']

        # Calculate the pi_theta (a_t|s_t)
        entropy, logp, values = self.actor_critic.logprob_ent_from_state_acton(observations, weights, actions)
        entropy_loss = entropy.mean()

        ratio = torch.exp(logp - logp_old)
        # Kl approx according to http://joschu.net/blog/kl-approx.html
        kl_apx = ((ratio - 1) - (logp - logp_old)).mean()
    
        clip_advs = torch.clamp(ratio, 1-self.eps_clip, 1+self.eps_clip) * advs
        # Torch Adam implement tation mius the gradient, to plus the gradient, we need make the loss negative
        actor_loss = -(torch.min(ratio*advs, clip_advs)).mean()- self.entropy_coef * entropy_loss

        values = values.flatten() # I used squeeze before, maybe a mistake

        critic_loss = F.mse_loss(values, rtgs)
        # critic_loss = ((values - rtgs) ** 2).mean()

        

        

        return actor_loss, critic_loss, entropy_loss, kl_apx  
          
    def lr_decay(self):
        lr_a_now = self.actor_lr* (1 - self.global_step / self.max_training_iter)
        lr_c_now = self.critic_lr * (1 - self.global_step / self.max_training_iter)
        for param_group, new_lr in zip(self.actor_critic_opt.param_groups, [lr_a_now, lr_c_now]):
            param_group['lr'] = new_lr

    
    def entropy_coef_decay(self):
        entropy_coef_now = self.c2 * (1 - self.global_step / self.max_training_iter)
        self.entropy_coef = max(1e-5, entropy_coef_now)
        # print(f'entropy_coef：{self.entropy_coef}')

    def optimise(self):

        entropy_loss_list = []
        actor_loss_list = []
        critic_loss_list = []
        kl_approx_list = []
        
        # for _ in tnrange(self.K_epochs, desc=f"epochs", position=1, leave=False):
        for _ in range(self.K_epochs):
            
            # resample the minibatch every epochs
            data = self.memory.sample(self.minibatch_size, self.device)
            
            for minibatch in data:
            
                actor_loss, critic_loss, entropy_loss, kl_apx = self.compute_loss(minibatch)

                entropy_loss_list.append(entropy_loss.item())
                actor_loss_list.append(actor_loss.item())
                critic_loss_list.append(critic_loss.item())
                kl_approx_list.append(kl_apx.item())

                if self.cal_total_loss:
                    total_loss = actor_loss + self.c1 * critic_loss - self.c2 * entropy_loss

                ### If this update is too big, early stop and try next minibatch
                if self.early_stop and kl_apx > self.kl_threshold:
                    self._early_stop_count += 1
                    ### OpenAI spinning up uses break as they use fullbatch instead of minibatch
                    ### Stable-baseline3 uses break, which is questionable as they drop the rest
                    ### of minibatches.
                    continue
                
                self.actor_critic_opt.zero_grad()
                if self.cal_total_loss:
                    total_loss.backward()
                    # Used by stable-baseline3, maybe more important for RNN
                    torch.nn.utils.clip_grad_norm_(self.actor_critic.parameters(), self.max_grad_norm)
                    self.actor_critic_opt.step()

                else:
                    actor_loss.backward()
                    critic_loss.backward()
                    # Used by stable-baseline3, maybe more important for RNN
                    torch.nn.utils.clip_grad_norm_(self.actor_critic.parameters(), self.max_grad_norm)
                    self.actor_critic_opt.step()

        self.memory.reset()    

        self.lr_decay()
        self.entropy_coef_decay()

                 
    def train(self, env, w):
        # gymnasium
        # self._last_obs, _ = env.reset()
        # old gym      
        self._last_obs= env.reset()

        self.roll_out(env,w)
        self.optimise()
         

        # save the model to the wandb run folder
        # PATH = os.path.join(wandb.run.dir, "actor_critic.pt")
        # torch.save(self.actor_critic.state_dict(), PATH)


        # wandb.run.summary['total_episode'] = self.episode_count

def evaluate_policy_PF(_dict_ , i, agent,w, env, times):
 
    weight_dim = len(w)
    # _dict_ = {}
    # eval_weights = [[0.5,0.5]]
 
    evaluate_reward = 0
    evaluate_obj1_reward = 0
    evaluate_obj2_reward = 0
    evaluate_obj3_reward = 0

    for _ in range(times):
        # gymnasium
        # obs, _ = env.reset()
        # old gym
        obs = env.reset()



        done = False
        truncate = False
        action_shape = env.action_space.shape
        episode_reward = 0
        episode_obj1_reward =0
        episode_obj2_reward=0
        episode_obj3_reward=0
        step = 0
        while not done and step<1000:
            obs_tensor = torch.tensor(obs, \
                                    dtype=torch.float32, device=agent.device).unsqueeze(0)
            w_tensor = torch.tensor(w, \
                                    dtype=torch.float32, device=agent.device).unsqueeze(0)
            action, _, _ = agent.actor_critic.act(obs_tensor,w_tensor) # We use the deterministic policy during the evaluating
            action = action.cpu().numpy()
            action = action.reshape(action_shape)
            # next_obs, reward, terminated, truncated, obj = env.step(action)
            next_obs, reward, terminated, obj = env.step(action)
            
            done = terminated
            

            if weight_dim ==2:
                r_obj1,r_obj2,r_obj3,r = obj['obj'][0],obj['obj'][1],-999,w[0]*obj['obj'][0]+w[1]*obj['obj'][1]
            if weight_dim ==3:
                r_obj1,r_obj2,r_obj3,r = obj['obj'][0],obj['obj'][1],obj['obj'][2],w[0]*obj['obj'][0]+w[1]*obj['obj'][1]+w[2]*obj['obj'][2]




            episode_reward += r
            episode_obj1_reward+=r_obj1
            episode_obj2_reward+=r_obj2
            episode_obj3_reward+=r_obj3
            obs = next_obs
            step += 1

    evaluate_reward = episode_reward / times
    evaluate_obj1_reward = episode_obj1_reward/times
    evaluate_obj2_reward = episode_obj2_reward/times
    evaluate_obj3_reward = episode_obj3_reward/times
    
    if weight_dim ==2:
        print(f'evaluate_rwd_{agent.global_step}:{evaluate_reward},{evaluate_obj1_reward},{evaluate_obj2_reward}')
        _dict_[i] = [evaluate_obj1_reward,evaluate_obj2_reward]
    elif weight_dim ==3:
        print(f'evaluate_rwd_{agent.global_step}:{evaluate_reward},{evaluate_obj1_reward},{evaluate_obj2_reward},{evaluate_obj3_reward}')
        _dict_[i] = [evaluate_obj1_reward,evaluate_obj2_reward,evaluate_obj3_reward]



    if not os.path.exists(f'./model/{env_name}/dwc2_model'):
        # Create the directory (and any intermediate directories) if it doesn't exist
        os.makedirs(f'./model/{env_name}/dwc2_model')
    torch.save(agent.actor_critic.actor.state_dict(), f'./model/{env_name}/dwc2_model/{agent.id}_dwc_run{wandb.config.seed}_actor_model.pt')  
    torch.save(agent.actor_critic.critic.state_dict(), f'./model/{env_name}/dwc2_model/{agent.id}_dwc_run{wandb.config.seed}_critic_model.pt') 
    return _dict_


def pick_random_weights(data_points, policy_num):
    random_picks = []
    subsegment_size = int(len(data_points)/policy_num)
    for i in range(0, len(data_points), subsegment_size):
        # Determine the end index of the current subsegment, ensuring it doesn't exceed the list length
        end_idx = min(i + subsegment_size, len(data_points))
        
        # Select a random data point from the current subsegment
        random_idx = np.random.randint(i, end_idx)
        random_picks.append(data_points[random_idx])

    return random_picks

def main(config, env_name, env, recording_env, all_weights, eval_weights, weight_dim, debug=False):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if debug:
        run = wandb.init(
            project='PPO-test',
            # mode='disabled',
            # config = sweep_configuration
        )
        gamma = 0.99
        lamb = 0.95
        eps_clip = 0.2
        max_training_iter = 1_000_000
        k_epochs = 10
        num_cells = 64
        actor_lr = 3e-4 
        critic_lr = actor_lr
        memory_size = 2048
        minibatch_size = 64    
        c1 = 0.5
        c2 = 0
        kl_threshold = 0.15
        env_name = env_name
        parameters_hardshare = False
        early_stop = False
        cal_total_loss = False
        max_grad_norm = 0.5
        seed = 123456

        wandb.config.update(
            {
                'actor_lr' : actor_lr,
                'critic_lr' : critic_lr,
                'gamma' : gamma,
                'lambda' : lamb,
                'eps_clip' : eps_clip,
                'max_training_iter' : max_training_iter,
                'k_epochs' : k_epochs,
                'hidden_cell_dim' : num_cells,
                'memory_size' : memory_size,
                'minibatch_size' : minibatch_size,
                'c1' : c1,
                'c2' : c2,
                'kl_threshold' : kl_threshold,
                'env_name': env_name,
                'early_stop' : early_stop,
                'parameters_hardshare' : parameters_hardshare,
                'early_stop' : early_stop,
                'cal_total_loss' : cal_total_loss,
                'max_grad_norm' : max_grad_norm,
                'seed' : seed
            }
        )   
    else:
        run = wandb.init(
            project='MORL_RANODM',
            # mode='disabled',
            config = config
        )
        policy_num = wandb.config.policy_num
        gamma = wandb.config.gamma
        lamb = wandb.config.lam
        k_epochs = wandb.config.k_epochs
        actor_lr = wandb.config.actor_lr
        critic_lr = wandb.config.critic_lr
        memory_size = wandb.config.memory_size
        minibatch_size = wandb.config.minibatch_size
        c1 = wandb.config.c1
        c2 = wandb.config.c2
        kl_threshold = wandb.config.kl_threshold
        env_name = wandb.config.env_name
        parameters_hardshare = wandb.config.parameters_hardshare
        early_stop = wandb.config.early_stop
        cal_total_loss = wandb.config.cal_total_loss
        max_grad_norm = wandb.config.max_grad_norm
        seed = wandb.config.seed
        eps_clip = wandb.config.eps_clip
        num_cells = wandb.config.num_cells
        max_training_iter = wandb.config.max_training_iter
        seed = wandb.config.seed  


    wandb.config.update(
        {
            'num_cells' : num_cells,
            'max_training_iter': max_training_iter,
            'implementation': 'moppo'
        }
    )
    


    # Seeding for evaluation purpose
    env.np_random = np.random.default_rng(seed)
    env.action_space.seed(seed)
    env.observation_space.seed(seed)

    recording_env.np_random = np.random.default_rng(seed)
    recording_env.action_space.seed(seed)
    recording_env.observation_space.seed(seed)

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    # Deterministic operations for CuDNN, it may impact performances
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    agent_list = []
    scenarios_per_policy = 10
    for policy_idx in range(policy_num):

        my_ppo = PPO(policy_idx, gamma, lamb, eps_clip, k_epochs, env.observation_space, weight_dim, env.action_space, num_cells,\
                 actor_lr, critic_lr, memory_size, minibatch_size, max_training_iter, \
                 cal_total_loss, c1, c2, early_stop, kl_threshold, parameters_hardshare, max_grad_norm, device)
        
        agent_list.append(my_ppo)

    weights_to_train = pick_random_weights(eval_weights,policy_num)

    for i in tqdm(range(int(max_training_iter) // int(memory_size))):
        for agent,w in zip(agent_list,weights_to_train):
            agent.train(env,w)

        if i % 10 == 0:
            _dict_ = {}

            for j, agent in enumerate(agent_list):
                # Calculate the start and end index for the scenarios this policy will be evaluated on
                start_idx = j * scenarios_per_policy
                end_idx = start_idx + scenarios_per_policy
                
                # Ensure end_idx does not exceed the length of eval_weights
                end_idx = min(end_idx, len(eval_weights))
                
                # Evaluate the policy on its assigned scenarios
                for w, k in zip(eval_weights[start_idx:end_idx], range(start_idx, end_idx)):
                    _dict_= evaluate_policy_PF(_dict_, k, agent, w, recording_env, times=10)

   
              

            with open(f'./results/{env_name}/dwc2/last_pf_{wandb.config.seed}_{agent.global_step}.txt', 'w') as file:
                    file.write(str(_dict_))
            hv = compute_hypervolume(list(_dict_.values()))
            # sp = compute_sparsity(list(_dict_.values()))
            # hvs = hv-sp
            with open(f'./results/{env_name}/dwc2/hv{wandb.config.seed}.txt', 'a') as file:
                file.write(f'{str(hv)},')

    env.close()
    recording_env.close()
    run.finish()





if __name__ == '__main__':
    

    parser = argparse.ArgumentParser(description="all enumerate for morl")
    # Add the --seed argument
    # Note that specifying type=int will automatically convert the command-line string to an integer
    parser.add_argument('--eidx', type=int, default=4, help='The random seed for reproducibility')
    parser.add_argument('--sidx', type=int, default=0, help='The random seed for reproducibility')
    
    # Parse the command-line arguments
    args = parser.parse_args()
    
    env_list = ['Swimmer-v2', 'HalfCheetah-v2', 'Walker2d-v2','Ant-v2','Hopper-v2', 'Hopper-v3', 'Humanoid-v2']
    policy_num = 9
    eval_step_size = 0.01
    seeds = [123456,88756,17859,0,255,22,2333,23256,52,345]
    env_name = env_list[int(args.eidx)]
    seed = seeds[int(args.sidx)]

    all_weights = []
    eval_weights = []

    if env_name == 'Hopper-v3':

        env = hopper_v3.HopperEnv()
        env_evaluate = hopper_v3.HopperEnv()

        config={
        'policy_num':policy_num,
        'max_training_iter': 6e6,
        'num_cells': 64,
        'actor_lr': 3e-4,
        'critic_lr': 3e-4,
        'memory_size': 2048,
        'k_epochs': 10,
        'gamma': 0.99,
        'lam': 0.95,
        'early_stop': False,
        'cal_total_loss': False,
        'parameters_hardshare': False,
        'c1': 0.5,
        'c2': 0,
        'minibatch_size': 64,
        'kl_threshold': 0.15,
        'max_grad_norm': 0.5,
        'eps_clip': 0.2,
        'seed': 123456,  
        'env_name': 'Hopper-v3', 
                }

        step = 0.2
        for i in np.arange(step, 1.0, step):
            i = np.round(i, 2)
            for j in np.arange(step, 1.0, step):
                j = np.round(j, 2)
                for k in np.arange(step, 1.0, step):
                    k= np.round(k, 2)
                    if abs(i+j+k-1)<1e-5:
                        all_weights.append([i, j,k])
        all_weights = np.round(all_weights, 2)

        eval_step_size = 0.05
        for i in np.arange(eval_step_size, 1.0, eval_step_size):
            i = np.round(i, 2)
            for j in np.arange(eval_step_size, 1.0, eval_step_size):
                j = np.round(j, 2)
                for k in np.arange(eval_step_size, 1.0, eval_step_size):
                    k= np.round(k, 2)
                    if abs(i+j+k-1)<1e-5:
                        eval_weights.append([i, j,k])

        eval_weights = np.round(eval_weights, 2)

    elif env_name == 'Hopper-v2':

        env = hopper.HopperEnv()
        env_evaluate = hopper.HopperEnv()

        config={
        'policy_num':policy_num,
        'max_training_iter': 1e6,
        'num_cells': 64,
        'actor_lr': 3e-4,
        'critic_lr': 3e-4,
        'memory_size': 2048,
        'k_epochs': 10,
        'gamma': 0.99,
        'lam': 0.95,
        'early_stop': False,
        'cal_total_loss': False,
        'parameters_hardshare': False,
        'c1': 0.5,
        'c2': 0,
        'minibatch_size': 64,
        'kl_threshold': 0.15,
        'max_grad_norm': 0.5,
        'eps_clip': 0.2,
        'seed': 123456,  
        'env_name': 'Hopper-v2', 
                }
        
        step = 0.1
        for i in np.arange(step, 1.0, step):
            i = np.round(i, 2)
            for j in np.arange(step, 1.0, step):
                j = np.round(j, 2)
                if abs(i+j-1)<1e-5:
                    all_weights.append([i, j])

        all_weights = np.round(all_weights, 2)

        for i in np.arange(eval_step_size, 1.0, eval_step_size):
            i = np.round(i, 2)
            for j in np.arange(eval_step_size, 1.0, eval_step_size):
                j = np.round(j, 2)
                if abs(i+j-1)<1e-5:
                    eval_weights.append([i, j])

        eval_weights = np.round(eval_weights, 2)

    elif env_name == 'HalfCheetah-v2':

        env = half_cheetah.HalfCheetahEnv()
        env_evaluate = half_cheetah.HalfCheetahEnv()

        config={
        'policy_num':policy_num,
        'max_training_iter': 1e6,
        'num_cells': 64,
        'actor_lr': 3e-4,
        'critic_lr': 3e-4,
        'memory_size': 2048,
        'k_epochs': 10,
        'gamma': 0.99,
        'lam': 0.95,
        'early_stop': False,
        'cal_total_loss': False,
        'parameters_hardshare': False,
        'c1': 0.5,
        'c2': 0,
        'minibatch_size': 64,
        'kl_threshold': 0.15,
        'max_grad_norm': 0.5,
        'eps_clip': 0.2,
        'seed': 123456,  
        'env_name': 'HalfCheetah-v2', 
                }
        
        step = 0.1
        for i in np.arange(step, 1.0, step):
            i = np.round(i, 2)
            for j in np.arange(step, 1.0, step):
                j = np.round(j, 2)
                if abs(i+j-1)<1e-5:
                    all_weights.append([i, j])

        all_weights = np.round(all_weights, 2)

        for i in np.arange(eval_step_size, 1.0, eval_step_size):
            i = np.round(i, 2)
            for j in np.arange(eval_step_size, 1.0, eval_step_size):
                j = np.round(j, 2)
                if abs(i+j-1)<1e-5:
                    eval_weights.append([i, j])

        eval_weights = np.round(eval_weights, 2)

    elif env_name == 'Ant-v2':

        env = ant.AntEnv()
        env_evaluate = ant.AntEnv()

        config={
        'policy_num':policy_num,
        'max_training_iter': 1e6,
        'num_cells': 64,
        'actor_lr': 3e-4,
        'critic_lr': 3e-4,
        'memory_size': 2048,
        'k_epochs': 10,
        'gamma': 0.99,
        'lam': 0.95,
        'early_stop': False,
        'cal_total_loss': False,
        'parameters_hardshare': False,
        'c1': 0.5,
        'c2': 0,
        'minibatch_size': 64,
        'kl_threshold': 0.15,
        'max_grad_norm': 0.5,
        'eps_clip': 0.2,
        'seed': 123456,  
        'env_name': 'Ant-v2', 
                }

 

        step = 0.1
        all_weights = []
        for i in np.arange(step, 1.0, step):
            i = np.round(i, 2)
            for j in np.arange(step, 1.0, step):
                j = np.round(j, 2)
                if abs(i+j-1)<1e-5:
                    all_weights.append([i, j])

        all_weights = np.round(all_weights, 2)

        eval_weights = []
        for i in np.arange(eval_step_size, 1.0, eval_step_size):
            i = np.round(i, 2)
            for j in np.arange(eval_step_size, 1.0, eval_step_size):
                j = np.round(j, 2)
                if abs(i+j-1)<1e-5:
                    eval_weights.append([i, j])

        eval_weights = np.round(eval_weights, 2)

    elif env_name == 'Humanoid-v2':

        env = humanoid.HumanoidEnv()
        env_evaluate = humanoid.HumanoidEnv()

        config={
        'policy_num':policy_num,
        'max_training_iter': 1e7,
        'num_cells': 64,
        'actor_lr': 3e-4,
        'critic_lr': 3e-4,
        'memory_size': 2048,
        'k_epochs': 10,
        'gamma': 0.99,
        'lam': 0.95,
        'early_stop': False,
        'cal_total_loss': False,
        'parameters_hardshare': False,
        'c1': 0.5,
        'c2': 0,
        'minibatch_size': 64,
        'kl_threshold': 0.15,
        'max_grad_norm': 0.5,
        'eps_clip': 0.2,
        'seed': 123456,  
        'env_name': 'Humanoid-v2', 
                }

        step = 0.1
        all_weights = []
        for i in np.arange(step, 1.0, step):
            i = np.round(i, 2)
            for j in np.arange(step, 1.0, step):
                j = np.round(j, 2)
                if abs(i+j-1)<1e-5:
                    all_weights.append([i, j])

        all_weights = np.round(all_weights, 2)

        eval_weights = []
        for i in np.arange(eval_step_size, 1.0, eval_step_size):
            i = np.round(i, 2)
            for j in np.arange(eval_step_size, 1.0, eval_step_size):
                j = np.round(j, 2)
                if abs(i+j-1)<1e-5:
                    eval_weights.append([i, j])

        eval_weights = np.round(eval_weights, 2)

    elif env_name == 'Walker2d-v2':

        # env = gym.make('Walker2d-v4')
        # env_evaluate = gym.make('Walker2d-v4')

        env = walker2d.Walker2dEnv()
        env_evaluate = walker2d.Walker2dEnv()

        config={
        'policy_num':policy_num,
        'max_training_iter': 1e6,
        'num_cells': 64,
        'actor_lr': 3e-4,
        'critic_lr': 3e-4,
        'memory_size': 2048,
        'k_epochs': 10,
        'gamma': 0.99,
        'lam': 0.95,
        'early_stop': False,
        'cal_total_loss': False,
        'parameters_hardshare': False,
        'c1': 0.5,
        'c2': 0,
        'minibatch_size': 64,
        'kl_threshold': 0.15,
        'max_grad_norm': 0.5,
        'eps_clip': 0.2,
        'seed': 123456,  
        'env_name': 'Walker2d-v2', 
                }
        
        step = 0.1
        for i in np.arange(step, 1.0, step):
            i = np.round(i, 2)
            for j in np.arange(step, 1.0, step):
                j = np.round(j, 2)
                if abs(i+j-1)<1e-5:
                    all_weights.append([i, j])
        all_weights = np.round(all_weights, 2)

        for i in np.arange(eval_step_size, 1.0, eval_step_size):
            i = np.round(i, 2)
            for j in np.arange(eval_step_size, 1.0, eval_step_size):
                j = np.round(j, 2)
                if abs(i+j-1)<1e-5:
                    eval_weights.append([i, j])
        eval_weights = np.round(eval_weights, 2)

    elif env_name == 'Swimmer-v2':

        env = swimmer.SwimmerEnv()
        env_evaluate = swimmer.SwimmerEnv()

        config={
        'policy_num':policy_num,
        'max_training_iter': 1e6,
        'num_cells': 64,
        'actor_lr': 3e-4,
        'critic_lr': 3e-4,
        'memory_size': 2048,
        'k_epochs': 10,
        'gamma': 0.99,
        'lam': 0.95,
        'early_stop': False,
        'cal_total_loss': False,
        'parameters_hardshare': False,
        'c1': 0.5,
        'c2': 0,
        'minibatch_size': 64,
        'kl_threshold': 0.15,
        'max_grad_norm': 0.5,
        'eps_clip': 0.2,
        'seed': 123456,  
        'env_name': 'Swimmer-v2', 
                }
        
        step = 0.1
        for i in np.arange(step, 1.0, step):
            i = np.round(i, 2)
            for j in np.arange(step, 1.0, step):
                j = np.round(j, 2)
                if abs(i+j-1)<1e-5:
                    all_weights.append([i, j])

        all_weights = np.round(all_weights, 2)

        

        for i in np.arange(eval_step_size, 1.0, eval_step_size):
            i = np.round(i, 2)
            for j in np.arange(eval_step_size, 1.0, eval_step_size):
                j = np.round(j, 2)
                if abs(i+j-1)<1e-5:
                    eval_weights.append([i, j])

        eval_weights = np.round(eval_weights, 2)


    weight_dim = len(all_weights[0])

    current_time = datetime.datetime.now().strftime("%Y_%m_%d_%H_%M_%S")

    MODE =0
    
    if MODE  == 0:
        pf_list = []
        ws = []
        evals = []
        obj1_evals = []
        obj2_evals = []

        
        directory = f'./results/{env_name}/dwc2/'

        # Ensure the directory exists, if not create it
        os.makedirs(directory, exist_ok=True)

        # Now you can safely create the file
        with open(os.path.join(directory, 'seeds.txt'), 'w') as the_file:
            the_file.write(str(seeds))
            

        # with open(f'./results/{env_name}/dwc2/pf_{seed}.txt', 'w') as file:
        #     file.write(str(dict_))

        main(config,env_name, env, env_evaluate, all_weights, eval_weights, weight_dim, debug=False)
        # %env "WANDB_NOTEBOOK_NAME" "PPO_GYM"
        # sweep_id = wandb.sweep(sweep=sweep_configuration, project='PPO_Mujoco_Compare')
        # wandb.agent(sweep_id, function=main(env_name, env, env_evaluate, all_weights, eval_weights, weight_dim, debug=False))