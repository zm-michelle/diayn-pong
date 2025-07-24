import gymnasium as gym
#import gym
from Brain import SACAgent
from Common import Play, Logger, get_params
import numpy as np
from tqdm import tqdm
import ale_py
import cv2 # 
#import mujoco_py


def concat_state_latent(s, z_, n, method=" "):
    z_one_hot = np.zeros(n)
    z_one_hot[z_] = 1 # (n, )
    if method == "flatten":
        s = s[0].astype(np.float32).flatten()  # flatten  
        return np.concatenate([s, z_one_hot])

    else:
        H, W = s.shape
        s = np.expand_dims(s, axis=0)
        z_maps = np.repeat(z_one_hot[:, None, None], H, axis=1)
        z_maps = np.repeat(z_maps, W, axis=2)  # shape: (n, H, W)
 
        out = np.concatenate([s, z_maps], axis=0) 
        return out

 


if __name__ == "__main__":
    params = get_params()
    gym.register_envs(ale_py)
    test_env = gym.make(params["env_name"], obs_type="grayscale")
    n_states = test_env.observation_space.shape[0] # continuous observation space
    n_actions = test_env.action_space.n #discrete
    #action_bounds = [test_env.action_space.low[0], test_env.action_space.high[0]] for continuous, 
    action_bounds = [0, test_env.action_space.n - 1] # chec k where or how is this checked

    params.update({"n_states": n_states,
                   "n_actions": n_actions,
                   "action_bounds": action_bounds})
    print("params:", params)
    test_env.close()
    del test_env, n_states, n_actions, action_bounds

    env = gym.make(params["env_name"], obs_type="grayscale")

    p_z = np.full(params["n_skills"], 1 / params["n_skills"])
    agent = SACAgent(p_z=p_z, **params)
    logger = Logger(agent, **params)

    if params["do_train"]:

        if not params["train_from_scratch"]:
            episode, last_logq_zs, np_rng_state, *env_rng_states, torch_rng_state, random_rng_state = logger.load_weights()
            agent.hard_update_target_network()
            min_episode = episode
            np.random.set_state(np_rng_state)
            env.np_random.set_state(env_rng_states[0])
            env.observation_space.np_random.set_state(env_rng_states[1])
            env.action_space.np_random.set_state(env_rng_states[2])
            agent.set_rng_states(torch_rng_state, random_rng_state) # necessary for reproducibility
            print("Keep training from previous run.")

        else:
            min_episode = 0
            last_logq_zs = 0
            np.random.seed(params["seed"])
            env.reset(seed=params["seed"])
            env.observation_space.seed(params["seed"])
            env.action_space.seed(params["seed"])
            print("Training from scratch.")

        logger.on()
        for episode in tqdm(range(1 + min_episode, params["max_n_episodes"] + 1)):
            z = np.random.choice(params["n_skills"], p=p_z)
            state, _ = env.reset() # shape: (210, 160, 3)
          
        
            state = concat_state_latent( state, z, params["n_skills"])
            episode_reward = 0
            logq_zses = []

            #max_n_steps = min(params["max_episode_len"], env.spec.max_episode_steps) # env.spec.max_episode_steps) is Nonetype for frameskip
            max_n_steps = params["max_episode_len"]
            for step in range(1, 1 + max_n_steps):

                action = agent.choose_action(state)
                next_state, reward, done, _ = env.step(action)
                next_state = concat_state_latent(next_state, z, params["n_skills"])
                agent.store(state, z, done, action, next_state)
                logq_zs = agent.train()
                if logq_zs is None:
                    logq_zses.append(last_logq_zs)
                else:
                    logq_zses.append(logq_zs)
                episode_reward += reward
                state = next_state
                if done:
                    break

            logger.log(episode,
                       episode_reward,
                       z,
                       sum(logq_zses) / len(logq_zses),
                       step,
                       np.random.get_state(),
                       env.np_random.get_state(),
                       env.observation_space.np_random.get_state(),
                       env.action_space.np_random.get_state(),
                       *agent.get_rng_states(),
                       )

    else:
        logger.load_weights()
        player = Play(env, agent, n_skills=params["n_skills"])
        player.evaluate()
