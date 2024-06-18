import torch
import torch.nn.functional as F
from torch.utils.data.sampler import BatchSampler, SubsetRandomSampler
import torch.nn as nn
from torch.distributions import Beta, Normal
import torch.optim as optim
import numpy as np

device_global = torch.device("cuda:2" if torch.cuda.is_available() else "cpu")


# Trick 8: orthogonal initialization
def orthogonal_init(layer, gain=1.0):
    nn.init.orthogonal_(layer.weight, gain=gain)
    nn.init.constant_(layer.bias, 0)


class ValueNetwork(nn.Module):
    def __init__(self, num_features, hidden_size, learning_rate=.01, device=device_global):
        super(ValueNetwork, self).__init__()
        self.device = device
        self.learning_rate = learning_rate

        self.fc1 = nn.Linear(num_features, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, 1)

        orthogonal_init(self.fc1)
        orthogonal_init(self.fc2)
        orthogonal_init(self.fc3)

        self.optimizer = optim.Adam(self.parameters(), lr=learning_rate, eps=1e-5)

    def forward(self, s):
        s = s.to(self.device)
        s = F.tanh(self.fc1(s))
        s = F.tanh(self.fc2(s))
        v_s = self.fc3(s)
        return v_s


class Actor_Gaussian(nn.Module):
    def __init__(self, num_features, hidden_size, num_actions, epsilon=.2, learning_rate=3e-4, device=device_global):
        super(Actor_Gaussian, self).__init__()
        self.device = device
        self.learning_rate = learning_rate

        self.fc1 = nn.Linear(num_features, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.mean_layer = nn.Linear(hidden_size, num_actions)
        self.log_std = nn.Parameter(torch.zeros(1, num_actions))  # We use 'nn.Parameter' to train log_std automatically

        orthogonal_init(self.fc1)
        orthogonal_init(self.fc2)
        orthogonal_init(self.mean_layer, gain=0.01)

        self.optimizer = optim.Adam(self.parameters(), lr=learning_rate, eps=1e-5)

    def forward(self, s):
        s = s.to(self.device)
        s = F.tanh(self.fc1(s))
        s = F.tanh(self.fc2(s))
        mean = F.softmax(self.mean_layer(s), dim=1)  # 区别：要不要加这个tanh
        # mean = torch.tanh(self.mean_layer(s))
        return mean

    def get_dist(self, s):
        mean = self.forward(s)
        log_std = self.log_std.expand_as(mean)  # To make 'log_std' have the same dimension as 'mean'
        std = torch.exp(log_std)  # The reason we train the 'log_std' is to ensure std=exp(log_std)>0
        dist = Normal(mean, std)  # Get the Gaussian distribution
        return dist


class ReplayBuffer(object):
    def __init__(self, batch_size, num_features, action_dim):
        self.device = device_global

        self.batch_size = batch_size
        self.s = np.zeros((batch_size, num_features))
        self.a = np.zeros((batch_size, action_dim))
        self.a_logprob = np.zeros((batch_size, action_dim))
        self.r = np.zeros((batch_size, 1))
        self.s_ = np.zeros((batch_size, num_features))  # s_next
        # means dead or win, there is no next state s
        self.dw = np.zeros((batch_size, 1))
        # represents the terminal of an episode(dead or win or reaching the max_episode_steps)
        self.done = np.zeros((batch_size, 1))
        self.count = 0

    def store(self, s, a, a_logprob, r, s_, dw, done):
        self.s[self.count] = s
        self.a[self.count] = a
        self.a_logprob[self.count] = a_logprob
        self.r[self.count] = r
        self.s_[self.count] = s_
        self.dw[self.count] = dw
        self.done[self.count] = done
        self.count += 1

    def numpy_to_tensor(self):
        s = torch.tensor(self.s, dtype=torch.float).to(self.device)
        a = torch.tensor(self.a, dtype=torch.float).to(self.device)
        a_logprob = torch.tensor(self.a_logprob, dtype=torch.float).to(self.device)
        r = torch.tensor(self.r, dtype=torch.float).to(self.device)
        s_ = torch.tensor(self.s_, dtype=torch.float).to(self.device)
        dw = torch.tensor(self.dw, dtype=torch.float).to(self.device)
        done = torch.tensor(self.done, dtype=torch.float).to(self.device)
        return s, a, a_logprob, r, s_, dw, done


class AgentReplayBuffer(object):
    def __init__(self):
        self.device = device_global

        self.s = []
        self.a = []
        self.a_logprob = []
        self.r = []
        self.s_ = []  # s_next
        # means dead or win, there is no next state s
        self.dw = []
        # represents the terminal of an episode(dead or win or reaching the max_episode_steps)
        self.done = []
        self.batch_size = 0

    def numpy_to_tensor(self):
        s = torch.stack(self.s).to(self.device)
        # In discrete action space, a needs to be torch.long
        a = torch.tensor(self.a, dtype=torch.long).to(self.device)
        a_logprob = torch.tensor(self.a_logprob, dtype=torch.float).to(self.device)
        r = torch.tensor(self.r, dtype=torch.float).to(self.device)
        s_ = torch.stack(self.s_).to(self.device)
        dw = torch.tensor(self.dw, dtype=torch.float).to(self.device)
        done = torch.tensor(self.done, dtype=torch.float).to(self.device)
        return s, a, a_logprob, r, s_, dw, done

    def store_a_al_s(self, a, a_logprob, s):
        self.a.append(a)
        self.a_logprob.append(a_logprob)
        self.s.append(s)

    def store_r(self, r, s_, done, dw):
        self.r.append(r)
        self.s_.append(s_)
        self.done.append(done)
        self.dw.append(dw)
        self.batch_size += 1

    def reset(self):
        self.s = []
        self.a = []
        self.a_logprob = []
        self.r = []
        self.s_ = []  # s_next
        # means dead or win, there is no next state s
        self.dw = []
        # represents the terminal of an episode(dead or win or reaching the max_episode_steps)
        self.done = []
        self.batch_size = 0


class PPO_continuous():
    def __init__(self, num_features, hidden_size, num_actions, max_step):
        self.device = device_global

        print(self.device)
        self.max_step = max_step

        self.actor = Actor_Gaussian(num_features=num_features,
                                    hidden_size=hidden_size,
                                    num_actions=num_actions,
                                    device=self.device).to(self.device)
        self.critic = ValueNetwork(num_features=num_features,
                                   hidden_size=hidden_size,
                                   device=self.device).to(self.device)

    def evaluate(self, s):  # When evaluating the policy, we only use the mean
        s = torch.unsqueeze(torch.tensor(s, dtype=torch.float), 0)
        a = self.actor(s).detach().cpu().numpy().flatten()
        return a

    def choose_action(self, s):
        s = torch.unsqueeze(torch.tensor(s, dtype=torch.float), 0)
        with torch.no_grad():
            dist = self.actor.get_dist(s)
            a = dist.sample()  # Sample the action according to the probability distribution
            a = torch.clamp(a, -1, 1)  # [-max,max]
            a_logprob = dist.log_prob(a)  # The log probability density of the action
        return a.cpu().numpy().flatten(), a_logprob.cpu().numpy().flatten()

    def update(self, replay_buffer, total_steps, gamma=0.99, lamda=0.95, k_epochs=10, epsilon=0.2):
        s, a, a_logprob, r, s_, dw, done = replay_buffer.numpy_to_tensor()  # Get training data
        """
            Calculate the advantage using GAE
            'dw=True' means dead or win, there is no next state s'
            'done=True' represents the terminal of an episode(dead or win or reaching the max_episode_steps). When calculating the adv, if done=True, gae=0
        """

        batch_size = replay_buffer.batch_size
        mini_batch_size = 1 if batch_size < 2000 else batch_size // 32

        adv = []
        gae = 0
        with torch.no_grad():  # adv and v_target have no gradient
            vs = self.critic(s)
            vs_ = self.critic(s_)
            deltas = r + gamma * (1.0 - dw) * vs_ - vs
            for delta, d in zip(reversed(deltas.flatten().cpu().numpy()), reversed(done.flatten().cpu().numpy())):
                gae = delta + gamma * lamda * gae * (1.0 - d)
                adv.insert(0, gae)
            adv = torch.tensor(adv, dtype=torch.float).view(-1, 1).to(self.device)
            v_target = adv + vs
            # Trick 1:advantage normalization
            adv = ((adv - adv.mean()) / (adv.std() + 1e-5))

        # Optimize policy for K epochs:
        for _ in range(k_epochs):
            # Random sampling and no repetition. 'False' indicates that training will continue even if the number of samples in the last time is less than mini_batch_size
            for index in BatchSampler(SubsetRandomSampler(range(batch_size)), mini_batch_size, False):
                dist_now = self.actor.get_dist(s[index])
                dist_entropy = dist_now.entropy().sum(1, keepdim=True)  # shape(mini_batch_size X 1)
                a_logprob_now = dist_now.log_prob(a[index])
                # a/b=exp(log(a)-log(b))  In multi-dimensional continuous action space，we need to sum up the log_prob
                ratios = torch.exp(a_logprob_now.sum(1, keepdim=True) - a_logprob[index].sum(1,
                                                                                             keepdim=True))  # shape(mini_batch_size X 1)

                surr1 = ratios * adv[index]  # Only calculate the gradient of 'a_logprob_now' in ratios
                surr2 = torch.clamp(ratios, 1 - epsilon, 1 + epsilon) * adv[index]
                actor_loss = -torch.min(surr1, surr2) - 0.03 * dist_entropy  # Trick 5: policy entropy
                # Update actor
                self.actor.optimizer.zero_grad()
                actor_loss.mean().backward()
                # Trick 7: Gradient clip
                torch.nn.utils.clip_grad_norm_(self.actor.parameters(), 0.5)

                self.actor.optimizer.step()

                v_s = self.critic(s[index])
                critic_loss = F.mse_loss(v_target[index], v_s)
                # Update critic
                self.critic.optimizer.zero_grad()
                critic_loss.backward()

                # Trick 7: Gradient clip
                torch.nn.utils.clip_grad_norm_(self.critic.parameters(), 0.5)
                self.critic.optimizer.step()

        # Trick 6:learning rate Decay
        self.lr_decay(total_steps)

    def lr_decay(self, total_steps):
        lr_a_now = self.actor.learning_rate * (1 - total_steps / self.max_step)
        lr_c_now = self.critic.learning_rate * (1 - total_steps / self.max_step)
        for p in self.actor.optimizer.param_groups:
            p['lr'] = lr_a_now
        for p in self.critic.optimizer.param_groups:
            p['lr'] = lr_c_now
