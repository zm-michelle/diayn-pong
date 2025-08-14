import gymnasium as gym
#import gym
from Brain import SACAgent
from Common import Play, Logger, get_params
import numpy as np
from tqdm import tqdm
import ale_py
import cv2 # 
#import mujoco_py
from torch.utils.tensorboard import SummaryWriter
def concat_state_latent(s, z_, n):
    s = s.flatten()
    #print("len s: ", len(s))
    z_one_hot = np.zeros(n)
    z_one_hot[z_] = 1
     
    result = np.concatenate([s, z_one_hot]) 
    return result

 
# action space: discrete(6)
# obs space: Box(0, 255, (210, 160), np.uint8)
if __name__ == "__main__":
    params = get_params()
    gym.register_envs(ale_py)
    test_env = gym.make(params["env_name"], obs_type="grayscale")
    print('Action space: ', test_env.action_space)
    print('Observation space: ', test_env.observation_space)
    #n_states = test_env.observation_space.shape[0] # continuous observation space
    H, W = test_env.observation_space.shape  #
    n_states = ( H * W)   #
    print("n states ", n_states)
    
    n_actions = test_env.action_space.n #discrete
    print("n_actions: ", n_actions)
    #action_bounds = [test_env.action_space.low[0], test_env.action_space.high[0]] for continuous, 
  # chec k where or how is this checked
    print("action_bounds: ", n_actions)
    params.update({"n_states": n_states,
                   "n_actions": n_actions,
                   })
    print("params:", params)
    test_env.close()
    del test_env, n_states, n_actions,  
    writer = SummaryWriter()
    env = gym.make(params["env_name"], obs_type="grayscale")

    p_z = np.full(params["n_skills"], 1 / params["n_skills"])
    print("p_z: ",p_z)
    print(f"   Sum: {p_z.sum():.6f} (should be 1.0)")
    agent = SACAgent(p_z=p_z, **params)
    print("=== DISCRIMINATOR SIZE CHECK ===")
    import torch 
    test_state = torch.randn(1, params['n_states']).to(agent.device)
    disc_output = agent.discriminator(test_state)
    print(f"Discriminator output shape: {disc_output.shape}")
    print(f"Should be: [1, {params['n_skills']}]")
    print(f"Actual output size: {disc_output.shape[-1]}")
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
            print("\n\n")
            # Skill chosen
            z = np.random.choice(params["n_skills"], p=p_z)
            print("Training skill (z): " , z)
            assert 0 <= z < params["n_skills"], f"Skill {z} out of bounds [0, {params['n_skills']})"
           
            state, _ = env.reset() 
                
            state = concat_state_latent( state, z, params["n_skills"])
            print(f"Concatenated state shape: {state.shape}")
            print(f"Concatenated state sum: {state.sum()}")
            print(f"state dtype: {state.dtype}")
            episode_reward = 0
            logq_zses = []

            #max_n_steps = min(params["max_episode_len"], env.spec.max_episode_steps) # env.spec.max_episode_steps) is Nonetype for frameskip
            max_n_steps = params["max_episode_len"]
            for step in range(1, 1 + max_n_steps):
                print("\n\n")
                action = agent.choose_action(state)
                print(f"action for step {step}, for skill {z}: {action}")
                next_state, reward, done, truncated, _ = env.step(action)
                next_state = concat_state_latent(next_state, z, params["n_skills"])
                agent.store(state, z, done, action, next_state)
                logq_zs = agent.train()
                if logq_zs is None:
                    if last_logq_zs is not None:
                        logq_zses.append(last_logq_zs)
                    # Or handle the None case differently
                else:
                    logq_zses.append(logq_zs)
                    last_logq_zs = logq_zs
                episode_reward += reward
                
                state = next_state
                
                writer.add_scalar(f"Reward skill {z}/step", reward, step)
                if done or truncated:
                    break
                    
            writer.add_scalar(f"Reward/episode, skill{z}", episode_reward, episode)


            def get_random_state(random_generator):
                """Get random state in a way compatible with both old and new NumPy versions"""
                try:
                    return random_generator.get_state()
                except AttributeError:
                    # For newer NumPy versions that use Generator instead of RandomState
                    if hasattr(random_generator, 'bit_generator'):
                        return random_generator.bit_generator.state
                    else:
                        # Fallback - return a simple state representation
                        return {'generator': str(random_generator)}
             
            logger.log(episode,
                       episode_reward,
                       z,
                       sum(logq_zses) / len(logq_zses),
                       step,
                       get_random_state(np.random),  # Changed
                        get_random_state(env.np_random),  # Changed
                        get_random_state(env.observation_space.np_random),  # Changed
                        get_random_state(env.action_space.np_random),  # Changed
                       *agent.get_rng_states(),
                       )
        writer.flush()
    else:
        logger.load_weights()
        player = Play(env, agent, n_skills=params["n_skills"])
        player.evaluate()
