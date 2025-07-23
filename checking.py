import gymnasium as gym
import ale_py # 0.9.0


gym.register_envs(ale_py)

games_available = gym.envs.registry.keys()
available = False
for game in games_available:
    if 'pong' in game.lower():
        available = True
        print(game)

if not available:
    print("download ale_py environment")
 