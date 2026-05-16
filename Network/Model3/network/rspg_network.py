import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


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
        x = x.reshape(-1, x.shape[-1])
        mean = self.fc_mean(x)
        log_std = torch.clamp(self.fc_log_std(x), -20, 2)
        return mean, log_std, hx


class CriticNetwork(nn.Module):
    def __init__(self, obs_dim, action_dim, hidden_dim=128):
        super().__init__()
        self.fc1 = nn.Linear(obs_dim + action_dim, hidden_dim)
        self.lstm = nn.LSTM(hidden_dim, hidden_dim, batch_first=True)
        self.fc_q = nn.Linear(hidden_dim, 1)

    def forward(self, obs, action, hx=None):
        if obs.dim() == 2:
            obs = obs.unsqueeze(1)
        if action.dim() == 2:
            action = action.unsqueeze(1)
        x = torch.cat([obs, action], dim=-1)
        x = F.relu(self.fc1(x))
        x, hx = self.lstm(x, hx)
        x = x.reshape(-1, x.shape[-1])
        q = self.fc_q(x)
        return q, hx


def soft_update(target, source, tau):
    for t_param, s_param in zip(target.parameters(), source.parameters()):
        t_param.data.copy_(t_param.data * (1.0 - tau) + s_param.data * tau)


def hard_update(target, source):
    for t_param, s_param in zip(target.parameters(), source.parameters()):
        t_param.data.copy_(s_param.data)


class ReplayBuffer:
    """
    Trajectory replay buffer for recurrent policy optimization.
    Stores whole episodes and samples contiguous history chunks.
    """

    def __init__(self, capacity, max_history=100):
        self.capacity = capacity
        self.max_history = max_history
        self.episodes = []
        self.current_episode = []
        self.num_transitions = 0

    def push(self, obs, action, reward, next_obs, done, hx=None):
        entry = {
            "obs": np.asarray(obs, dtype=np.float32),
            "action": np.asarray(action, dtype=np.float32),
            "reward": float(reward),
            "next_obs": np.asarray(next_obs, dtype=np.float32),
            "done": float(done),
        }
        self.current_episode.append(entry)

        if done:
            self._commit_current_episode()

    def _commit_current_episode(self):
        if not self.current_episode:
            return
        self.episodes.append(self.current_episode)
        self.num_transitions += len(self.current_episode)
        self.current_episode = []

        while self.num_transitions > self.capacity and self.episodes:
            removed = self.episodes.pop(0)
            self.num_transitions -= len(removed)

    def sample(self, batch_size, seq_len=None, burn_in=0):
        if not self.episodes:
            raise ValueError("Replay buffer is empty; no completed episodes available.")

        seq_len = seq_len or self.max_history
        burn_in = max(int(burn_in), 0)
        obs_dim = self.episodes[0][0]["obs"].shape[0]
        action_dim = self.episodes[0][0]["action"].shape[0]

        burn_in_obs = np.zeros((batch_size, burn_in, obs_dim), dtype=np.float32)
        burn_in_actions = np.zeros((batch_size, burn_in, action_dim), dtype=np.float32)
        burn_in_masks = np.zeros((batch_size, burn_in, 1), dtype=np.float32)
        obs = np.zeros((batch_size, seq_len, obs_dim), dtype=np.float32)
        actions = np.zeros((batch_size, seq_len, action_dim), dtype=np.float32)
        rewards = np.zeros((batch_size, seq_len, 1), dtype=np.float32)
        next_obs = np.zeros((batch_size, seq_len, obs_dim), dtype=np.float32)
        dones = np.zeros((batch_size, seq_len, 1), dtype=np.float32)
        masks = np.zeros((batch_size, seq_len, 1), dtype=np.float32)

        ep_indices = np.random.choice(len(self.episodes), size=batch_size, replace=True)
        for batch_idx, ep_idx in enumerate(ep_indices):
            episode = self.episodes[ep_idx]
            if len(episode) <= seq_len:
                train_start = 0
            else:
                train_start = np.random.randint(0, len(episode) - seq_len + 1)

            context_start = max(0, train_start - burn_in)
            prefix = episode[context_start:train_start]
            segment = episode[train_start : train_start + seq_len]

            for step_idx, transition in enumerate(prefix):
                burn_in_obs[batch_idx, step_idx] = transition["obs"]
                burn_in_actions[batch_idx, step_idx] = transition["action"]
                burn_in_masks[batch_idx, step_idx, 0] = 1.0

            for step_idx, transition in enumerate(segment):
                obs[batch_idx, step_idx] = transition["obs"]
                actions[batch_idx, step_idx] = transition["action"]
                rewards[batch_idx, step_idx, 0] = transition["reward"]
                next_obs[batch_idx, step_idx] = transition["next_obs"]
                dones[batch_idx, step_idx, 0] = transition["done"]
                masks[batch_idx, step_idx, 0] = 1.0

        return (
            torch.tensor(burn_in_obs, dtype=torch.float32),
            torch.tensor(burn_in_actions, dtype=torch.float32),
            torch.tensor(burn_in_masks, dtype=torch.float32),
            torch.tensor(obs, dtype=torch.float32),
            torch.tensor(actions, dtype=torch.float32),
            torch.tensor(rewards, dtype=torch.float32),
            torch.tensor(next_obs, dtype=torch.float32),
            torch.tensor(dones, dtype=torch.float32),
            torch.tensor(masks, dtype=torch.float32),
        )

    def __len__(self):
        return self.num_transitions


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
            list(self.critic_1.parameters()) + list(self.critic_2.parameters()),
            lr=config.RSPG_LR,
        )

        self.log_alpha = torch.tensor(
            np.log(config.RSPG_ENTROPY_ALPHA_INIT),
            requires_grad=True,
            device=self.device,
        )
        self.alpha_optimizer = torch.optim.Adam([self.log_alpha], lr=config.RSPG_ALPHA_LR)

        self.replay_buffer = ReplayBuffer(config.RSPG_REPLAY_BUFFER, config.RSPG_MAX_HISTORY)

        self.gamma = config.RSPG_GAMMA
        self.tau = config.RSPG_TAU
        self.batch_size = config.RSPG_BATCH_SIZE
        self.grad_clip = config.RSPG_GRAD_CLIP
        self.target_entropy = config.RSPG_TARGET_ENTROPY
        self.max_history = config.RSPG_MAX_HISTORY
        self.burn_in = min(getattr(config, "RSPG_BURN_IN", 0), max(self.max_history - 1, 0))
        self.train_seq_len = max(self.max_history - self.burn_in, 1)
        self.action_dim = action_dim

    @property
    def alpha(self):
        return torch.exp(self.log_alpha)

    def select_action(self, obs, hx=None, deterministic=False):
        obs_tensor = torch.tensor(obs, dtype=torch.float32, device=self.device)
        with torch.no_grad():
            mean, log_std, hx_out = self.actor(obs_tensor.unsqueeze(0), hx)
        if deterministic:
            action = mean
        else:
            std = log_std.exp()
            dist = torch.distributions.Normal(mean, std)
            action = dist.rsample()
        return action.squeeze(0).cpu().numpy(), hx_out

    def _hidden_size_for(self, network):
        return network.lstm.hidden_size

    def _zero_hidden(self, batch_size, hidden_size):
        h = torch.zeros(1, batch_size, hidden_size, device=self.device)
        c = torch.zeros(1, batch_size, hidden_size, device=self.device)
        return h, c

    def _compute_burn_in_hidden_actor(self, network, burn_in_obs, burn_in_mask):
        batch_size = burn_in_obs.shape[0]
        hidden_size = self._hidden_size_for(network)
        h_list = []
        c_list = []

        with torch.no_grad():
            for batch_idx in range(batch_size):
                valid = int(burn_in_mask[batch_idx].sum().item())
                if valid <= 0:
                    h_i, c_i = self._zero_hidden(1, hidden_size)
                else:
                    _, _, (h_i, c_i) = network(burn_in_obs[batch_idx : batch_idx + 1, :valid])
                h_list.append(h_i)
                c_list.append(c_i)

        return torch.cat(h_list, dim=1), torch.cat(c_list, dim=1)

    def _compute_burn_in_hidden_critic(self, network, burn_in_obs, burn_in_actions, burn_in_mask):
        batch_size = burn_in_obs.shape[0]
        hidden_size = self._hidden_size_for(network)
        h_list = []
        c_list = []

        with torch.no_grad():
            for batch_idx in range(batch_size):
                valid = int(burn_in_mask[batch_idx].sum().item())
                if valid <= 0:
                    h_i, c_i = self._zero_hidden(1, hidden_size)
                else:
                    _, (h_i, c_i) = network(
                        burn_in_obs[batch_idx : batch_idx + 1, :valid],
                        burn_in_actions[batch_idx : batch_idx + 1, :valid],
                    )
                h_list.append(h_i)
                c_list.append(c_i)

        return torch.cat(h_list, dim=1), torch.cat(c_list, dim=1)

    def _reshape_actor_outputs(self, mean, log_std, batch_size, seq_len):
        mean = mean.view(batch_size, seq_len, self.action_dim)
        log_std = log_std.view(batch_size, seq_len, self.action_dim)
        return mean, log_std

    def update(self):
        if len(self.replay_buffer) < self.batch_size:
            return

        (
            burn_in_obs,
            burn_in_actions,
            burn_in_mask,
            obs,
            action,
            reward,
            next_obs,
            done,
            mask,
        ) = self.replay_buffer.sample(
            self.batch_size,
            seq_len=self.train_seq_len,
            burn_in=self.burn_in,
        )
        burn_in_obs = burn_in_obs.to(self.device)
        burn_in_actions = burn_in_actions.to(self.device)
        burn_in_mask = burn_in_mask.to(self.device)
        obs = obs.to(self.device)
        action = action.to(self.device)
        reward = reward.to(self.device)
        next_obs = next_obs.to(self.device)
        done = done.to(self.device)
        mask = mask.to(self.device)

        normalizer = mask.sum().clamp_min(1.0)
        actor_hx = self._compute_burn_in_hidden_actor(self.actor, burn_in_obs, burn_in_mask)
        target_actor_hx = self._compute_burn_in_hidden_actor(self.target_actor, burn_in_obs, burn_in_mask)
        critic1_hx = self._compute_burn_in_hidden_critic(self.critic_1, burn_in_obs, burn_in_actions, burn_in_mask)
        critic2_hx = self._compute_burn_in_hidden_critic(self.critic_2, burn_in_obs, burn_in_actions, burn_in_mask)
        target_critic1_hx = self._compute_burn_in_hidden_critic(
            self.target_critic_1, burn_in_obs, burn_in_actions, burn_in_mask
        )
        target_critic2_hx = self._compute_burn_in_hidden_critic(
            self.target_critic_2, burn_in_obs, burn_in_actions, burn_in_mask
        )

        with torch.no_grad():
            next_mean, next_log_std, _ = self.target_actor(next_obs, target_actor_hx)
            next_mean, next_log_std = self._reshape_actor_outputs(
                next_mean, next_log_std, self.batch_size, self.train_seq_len
            )
            next_std = next_log_std.exp()
            next_dist = torch.distributions.Normal(next_mean, next_std)
            next_action = next_dist.rsample()
            next_log_prob = next_dist.log_prob(next_action).sum(-1, keepdim=True)

            q1_next, _ = self.target_critic_1(next_obs, next_action, target_critic1_hx)
            q2_next, _ = self.target_critic_2(next_obs, next_action, target_critic2_hx)

            q1_next = q1_next.view(self.batch_size, self.train_seq_len, 1)
            q2_next = q2_next.view(self.batch_size, self.train_seq_len, 1)
            next_log_prob = next_log_prob.view(self.batch_size, self.train_seq_len, 1)

            q_next = torch.min(q1_next, q2_next)
            target_q = reward + (1 - done) * self.gamma * (q_next - self.alpha.detach() * next_log_prob)

        q1, _ = self.critic_1(obs, action, critic1_hx)
        q2, _ = self.critic_2(obs, action, critic2_hx)
        q1 = q1.view(self.batch_size, self.train_seq_len, 1)
        q2 = q2.view(self.batch_size, self.train_seq_len, 1)

        critic_loss = (
            (((q1 - target_q) ** 2) + ((q2 - target_q) ** 2)) * mask
        ).sum() / normalizer

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.critic_1.parameters(), self.grad_clip)
        torch.nn.utils.clip_grad_norm_(self.critic_2.parameters(), self.grad_clip)
        self.critic_optimizer.step()

        mean, log_std, _ = self.actor(obs, actor_hx)
        mean, log_std = self._reshape_actor_outputs(
            mean, log_std, self.batch_size, self.train_seq_len
        )
        std = log_std.exp()
        dist = torch.distributions.Normal(mean, std)
        sampled_action = dist.rsample()
        log_prob = dist.log_prob(sampled_action).sum(-1, keepdim=True)

        q1_pi, _ = self.critic_1(obs, sampled_action, critic1_hx)
        q2_pi, _ = self.critic_2(obs, sampled_action, critic2_hx)
        q_pi = torch.min(q1_pi, q2_pi)

        q_pi = q_pi.view(self.batch_size, self.train_seq_len, 1)
        log_prob = log_prob.view(self.batch_size, self.train_seq_len, 1)

        actor_loss = ((self.alpha.detach() * log_prob - q_pi) * mask).sum() / normalizer

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.actor.parameters(), self.grad_clip)
        self.actor_optimizer.step()

        alpha_loss = (-(self.log_alpha * (log_prob + self.target_entropy).detach()) * mask).sum() / normalizer
        self.alpha_optimizer.zero_grad()
        alpha_loss.backward()
        self.alpha_optimizer.step()

        soft_update(self.target_actor, self.actor, self.tau)
        soft_update(self.target_critic_1, self.critic_1, self.tau)
        soft_update(self.target_critic_2, self.critic_2, self.tau)

    def save(self, path):
        torch.save(
            {
                "actor": self.actor.state_dict(),
                "critic_1": self.critic_1.state_dict(),
                "critic_2": self.critic_2.state_dict(),
                "target_actor": self.target_actor.state_dict(),
                "target_critic_1": self.target_critic_1.state_dict(),
                "target_critic_2": self.target_critic_2.state_dict(),
                "actor_optimizer": self.actor_optimizer.state_dict(),
                "critic_optimizer": self.critic_optimizer.state_dict(),
                "alpha_optimizer": self.alpha_optimizer.state_dict(),
                "log_alpha": self.log_alpha.detach().cpu(),
            },
            path,
        )

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
        alpha_lr = self.alpha_optimizer.param_groups[0]["lr"]
        log_alpha_value = checkpoint["log_alpha"].to(self.device)
        self.log_alpha = torch.tensor(log_alpha_value.item(), requires_grad=True, device=self.device)
        self.alpha_optimizer = torch.optim.Adam([self.log_alpha], lr=alpha_lr)
        if "alpha_optimizer" in checkpoint:
            self.alpha_optimizer.load_state_dict(checkpoint["alpha_optimizer"])
