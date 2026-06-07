from . import openai_gym_compatibility


class GymV21CompatibilityV0:
    def __init__(self, env):
        self.env = env

    def __getattr__(self, name):
        return getattr(self.env, name)


class GymV26CompatibilityV0(GymV21CompatibilityV0):
    pass
