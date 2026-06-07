import gym
import gymnasium


def _convert_space(space):
    if isinstance(space, gymnasium.Space):
        return space
    if isinstance(space, gym.spaces.Box):
        return gymnasium.spaces.Box(
            low=space.low,
            high=space.high,
            shape=space.shape,
            dtype=space.dtype,
        )
    if isinstance(space, gym.spaces.Discrete):
        return gymnasium.spaces.Discrete(space.n, start=getattr(space, "start", 0))
    if isinstance(space, gym.spaces.MultiBinary):
        return gymnasium.spaces.MultiBinary(space.n)
    if isinstance(space, gym.spaces.MultiDiscrete):
        return gymnasium.spaces.MultiDiscrete(space.nvec)
    if isinstance(space, gym.spaces.Tuple):
        return gymnasium.spaces.Tuple(tuple(_convert_space(item) for item in space.spaces))
    if isinstance(space, gym.spaces.Dict):
        return gymnasium.spaces.Dict({
            key: _convert_space(value)
            for key, value in space.spaces.items()
        })
    raise NotImplementedError(f"Unsupported Gym space for local shimmy conversion: {space}")
