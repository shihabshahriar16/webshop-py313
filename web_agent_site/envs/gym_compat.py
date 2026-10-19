"""``gym`` if installed (the original py3.8 setup), else ``gymnasium``.

The text env only subclasses ``gym.Env`` and registers ids; both packages
provide those. gym itself is unmaintained and breaks on NumPy 2.
"""
try:
    import gym
    from gym.envs.registration import register
except ImportError:
    import gymnasium as gym
    from gymnasium.envs.registration import register
