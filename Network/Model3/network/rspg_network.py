import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from copy import deepcopy

class ActorNetwork(nn.Module):
    def __init__(self, obs_dim, action_dim, hidden_dim=128):
        super().__init__()
        self.fc1 = nn.Linear(obs_dim, hidden_dim)
        self.lstm = nn.LSTM(hidden_dim, hidden_dim, batch_first=True)
        self.fc_mean = nn.Linear(hidden_dim, action_dim)
        self.fc_log_std = nn.Linear(hidden_dim, action_dim)

    def forward(self, obs, hx=None):
        # obs: (B, T, obs_dim) or (B, obs_dim)
        if obs.dim() == 2:
            obs = obs.unsqueeze(1)
        x = F.relu(self.fc1(obs))
        x, hx = self.lstm(x, hx)
        mean = self.fc_mean(x)
        log_std = self.fc_log_std(x)
        log_std = torch.clamp(log_std, -20, 2)
        return mean, log_std, hx


class CriticNetwork(nn.Module):
    def __init__(self, obs_dim, action_dim, hidden_dim=128):
        super().__init__()
        self.fc1 = nn.Linear(obs_dim + action_dim, hidden_dim)
        self.lstm = nn.LSTM(hidden_dim, hidden_dim, batch_first=True)
        self.fc_q = nn.Linear(hidden_dim, 1)

    def forward(self, obs, action, hx=None):
        # obs: (B, T, obs_dim), action: (B, T, action_dim)
        if obs.dim() == 2:
            obs = obs.unsqueeze(1)
        if action.dim() == 2:
            action = action.unsqueeze(1)
        x = torch.cat([obs, action], dim=-1)
        x = F.relu(self.fc1(x))
        x, hx = self.lstm(x, hx)
        q = self.fc_q(x)
        return q, hx


def soft_update(target, source, tau):
    for t, s in zip(target.parameters(), source.parameters()):
        t.data.copy_(t.data * (1.0 - tau) + s.data * tau)


def hard_update(target, source):
    for t, s in zip(target.parameters(), source.parameters()):
        t.data.copy_(s.data)


class ReplayBuffer:
    def __init__(self, capacity, max_history=100):
        self.capacity = capacity
        self.max_history = max_history
        self.buffer = []
        self.pos = 0
        self.episode_id = 0

    def push(self, obs, action, reward, next_obs, done, hx=None):
        entry = {
            "obs": obs,
            "action": action,
            "reward": reward,
            "next_obs": next_obs,
            "done": done,
            "episode_id": self.episode_id,
        }
        if len(self.buffer) < self.capacity:
            self.buffer.append(entry)
        else:
            self.buffer[self.pos] = entry
        self.pos = (self.pos + 1) % self.capacity
        if done:
            self.episode_id += 1

    def sample(self, batch_size):
        valid_ends = [
            i for i, b in enumerate(self.buffer)
            if i > 0 and self.buffer[i - 1]["episode_id"] == b["episode_id"]
        ]
        if len(valid_ends) < batch_size:
            valid_ends = list(range(len(self.buffer)))

        end_indices = np.random.choice(valid_ends, batch_size, replace=False)
        sequences = []
        seq_len = 1
        for end in end_indices:
            ep = self.buffer[end]["episode_id"]
            start = end
            while (start > 0 and end - start + 1 < self.max_history
                   and self.buffer[start - 1]["episode_id"] == ep):
                start -= 1
            sequences.append((start, end))
            seq_len = max(seq_len, end - start + 1)

        def pad_sequence(key, dtype=torch.float32):
            rows = []
            for start, end in sequences:
                items = [self.buffer[i][key] for i in range(start, end + 1)]
                pad_count = seq_len - len(items)
                if pad_count:
                    items = [items[0]] * pad_count + items
                rows.append(torch.tensor(np.array(items), dtype=dtype))
            return torch.stack(rows)

        obs = pad_sequence("obs")
        action = pad_sequence("action")
        reward = pad_sequence("reward").unsqueeze(-1)
        next_obs = pad_sequence("next_obs")
        done = pad_sequence("done").unsqueeze(-1)
        return obs, action, reward, next_obs, done

    def __len__(self):
        return len(self.buffer)


class RSPGAgent:
    def __init__(self, obs_dim, action_dim, config):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.actor = ActorNetwork(obs_dim, action_dim).to(self.device)
        self.critic_1 = CriticNetwork(obs_dim, action_dim).to(self.device)
        self.critic_2 = CriticNetwork(obs_dim, action_dim).to(self.device)

        self.target_actor = ActorNetwork(obs_dim, action_dim).to(self.device)
        self.target_critic_1 = CriticNetwork(obs_dim, action_dim).to(self.device)
        self.target_critic_2 = CriticNetwork(obs_dim, action_dim).to(self.device)

        hard_update(self.target_actor, self.actor)
        hard_update(self.target_critic_1, self.critic_1)
        hard_update(self.target_critic_2, self.critic_2)

        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=config.RSPG_LR)
        self.critic_optimizer = torch.optim.Adam(
            list(self.critic_1.parameters()) + list(self.critic_2.parameters()), lr=config.RSPG_LR
        )

        self.log_alpha = torch.tensor(np.log(config.RSPG_ENTROPY_ALPHA_INIT), requires_grad=True, device=self.device)
        self.alpha_optimizer = torch.optim.Adam([self.log_alpha], lr=config.RSPG_ALPHA_LR)

        self.replay_buffer = ReplayBuffer(config.RSPG_REPLAY_BUFFER, config.RSPG_MAX_HISTORY)

        self.gamma = config.RSPG_GAMMA
        self.tau = config.RSPG_TAU
        self.batch_size = config.RSPG_BATCH_SIZE
        self.grad_clip = config.RSPG_GRAD_CLIP
        self.target_entropy = config.RSPG_TARGET_ENTROPY
        self.action_pos_max = config.ACTION_POS_MAX
        self.action_angle_max = np.pi

    @property
    def alpha(self):
        return torch.exp(self.log_alpha)

    def _scale_action(self, raw_action):
        squashed = torch.tanh(raw_action)
        pos = squashed[..., :3] * self.action_pos_max
        angles = squashed[..., 3:] * self.action_angle_max
        return torch.cat([pos, angles], dim=-1), squashed

    def _sample_policy_action(self, mean, log_std, deterministic=False):
        if deterministic:
            raw_action = mean
            action, _ = self._scale_action(raw_action)
            return action, None

        std = log_std.exp()
        dist = torch.distributions.Normal(mean, std)
        raw_action = dist.rsample()
        action, squashed = self._scale_action(raw_action)

        scale = torch.ones_like(action)
        scale[..., :3] *= self.action_pos_max
        scale[..., 3:] *= self.action_angle_max
        log_prob = dist.log_prob(raw_action) - torch.log(
            scale * (1.0 - squashed.pow(2)) + 1e-6
        )
        log_prob = log_prob.sum(-1, keepdim=True)
        return action, log_prob

    def select_action(self, obs, hx=None, deterministic=False):
        obs_tensor = torch.tensor(obs, dtype=torch.float32, device=self.device)
        with torch.no_grad():
            mean, log_std, hx_out = self.actor(obs_tensor.view(1, 1, -1), hx)
            action, _ = self._sample_policy_action(mean, log_std, deterministic)
        return action[0, -1].cpu().numpy(), hx_out

    def update(self):
        if len(self.replay_buffer) < self.batch_size:
            return

        obs, action, reward, next_obs, done = self.replay_buffer.sample(self.batch_size)
        obs = obs.to(self.device)
        action = action.to(self.device)
        reward = reward.to(self.device)
        next_obs = next_obs.to(self.device)
        done = done.to(self.device)

        # Critic update
        with torch.no_grad():
            next_mean, next_log_std, _ = self.target_actor(next_obs)
            next_action, next_log_prob = self._sample_policy_action(next_mean, next_log_std)

            q1_next, _ = self.target_critic_1(next_obs, next_action)
            q2_next, _ = self.target_critic_2(next_obs, next_action)
            q_next = torch.min(q1_next, q2_next)
            target_q = reward + (1 - done) * self.gamma * (q_next - self.alpha * next_log_prob)

        q1, _ = self.critic_1(obs, action)
        q2, _ = self.critic_2(obs, action)
        critic_loss = F.mse_loss(q1, target_q) + F.mse_loss(q2, target_q)

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.critic_1.parameters(), self.grad_clip)
        torch.nn.utils.clip_grad_norm_(self.critic_2.parameters(), self.grad_clip)
        self.critic_optimizer.step()

        # Actor update
        mean, log_std, _ = self.actor(obs)
        sampled_action, log_prob = self._sample_policy_action(mean, log_std)

        q1_pi, _ = self.critic_1(obs, sampled_action)
        q2_pi, _ = self.critic_2(obs, sampled_action)
        q_pi = torch.min(q1_pi, q2_pi)

        actor_loss = (self.alpha * log_prob - q_pi).mean()

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.actor.parameters(), self.grad_clip)
        self.actor_optimizer.step()

        # Alpha update
        alpha_loss = -(self.log_alpha * (log_prob + self.target_entropy).detach()).mean()
        self.alpha_optimizer.zero_grad()
        alpha_loss.backward()
        self.alpha_optimizer.step()

        # Soft update target networks
        soft_update(self.target_actor, self.actor, self.tau)
        soft_update(self.target_critic_1, self.critic_1, self.tau)
        soft_update(self.target_critic_2, self.critic_2, self.tau)

    def save(self, path):
        torch.save({
            "actor": self.actor.state_dict(),
            "critic_1": self.critic_1.state_dict(),
            "critic_2": self.critic_2.state_dict(),
            "target_actor": self.target_actor.state_dict(),
            "target_critic_1": self.target_critic_1.state_dict(),
            "target_critic_2": self.target_critic_2.state_dict(),
            "actor_optimizer": self.actor_optimizer.state_dict(),
            "critic_optimizer": self.critic_optimizer.state_dict(),
            "log_alpha": self.log_alpha,
        }, path)

    def load(self, path):
        checkpoint = torch.load(path, map_location=self.device)
        self.actor.load_state_dict(checkpoint["actor"])
        self.critic_1.load_state_dict(checkpoint["critic_1"])
        self.critic_2.load_state_dict(checkpoint["critic_2"])
        self.target_actor.load_state_dict(checkpoint["target_actor"])
        self.target_critic_1.load_state_dict(checkpoint["target_critic_1"])
        self.target_critic_2.load_state_dict(checkpoint["target_critic_2"])
        self.actor_optimizer.load_state_dict(checkpoint["actor_optimizer"])
        self.critic_optimizer.load_state_dict(checkpoint["critic_optimizer"])
        self.log_alpha = checkpoint["log_alpha"].to(self.device).detach().requires_grad_(True)
        self.alpha_optimizer = torch.optim.Adam([self.log_alpha], lr=self.alpha_optimizer.param_groups[0]["lr"])
