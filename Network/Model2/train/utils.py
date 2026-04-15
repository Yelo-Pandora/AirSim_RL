# utils.py
import random
import numpy as np

class ReplayBuffer:
    def __init__(self, capacity=10000):
        self.capacity = capacity
        self.episodes = []

    def push_episode(self, episode):
        """
        episode: list of (img, state, action, reward, next_img, next_state, done)
        """
        if len(self.episodes) >= self.capacity:
            self.episodes.pop(0)
        self.episodes.append(episode)

    def sample_sequence(self, batch_size, seq_len):
        batch = []

        for _ in range(batch_size):
            ep = random.choice(self.episodes)

            if len(ep) < seq_len:
                continue

            start = random.randint(0, len(ep) - seq_len)
            seq = ep[start:start + seq_len]
            batch.append(seq)

        # 拆解
        images, states, actions, rewards, next_images, next_states, dones = [], [], [], [], [], [], []

        for seq in batch:
            img_seq, state_seq, action_seq, reward_seq, next_img_seq, next_state_seq, done_seq = zip(*seq)

            images.append(img_seq)
            states.append(state_seq)
            actions.append(action_seq)
            rewards.append(reward_seq)
            next_images.append(next_img_seq)
            next_states.append(next_state_seq)
            dones.append(done_seq)

        return (
            np.array(images),
            np.array(states),
            np.array(actions),
            np.array(rewards),
            np.array(next_images),
            np.array(next_states),
            np.array(dones),
        )

    def __len__(self):
        return len(self.episodes)