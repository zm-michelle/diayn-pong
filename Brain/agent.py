import numpy as np
from .model import PolicyNetwork, QvalueNetwork, ValueNetwork, Discriminator
import torch
from .replay_memory import Memory, Transition
from torch import from_numpy
from torch.optim.adam import Adam
from torch.nn.functional import log_softmax
from torch.nn import functional as F
import time
from torch.utils.tensorboard import SummaryWriter

class SACAgent:
    def __init__(self,
                 p_z,
                 **config):
        self.config = config
        self.n_states = self.config["n_states"]
        self.n_skills = self.config["n_skills"]
        self.batch_size = self.config["batch_size"]
        self.p_z = np.tile(p_z, self.batch_size).reshape(self.batch_size, self.n_skills).astype(np.float32)
        #self.memory = Memory(self.config["mem_size"], self.config["seed"])
        self.memory = Memory(self.config["mem_size"], self.config["seed"])
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print("Device: ", self.device)
        time.sleep(15)

        torch.manual_seed(self.config["seed"])
        print("\n\nn_actions: ", self.config["n_actions"])
        self.policy_network = PolicyNetwork(n_states=self.n_states + self.n_skills,
                                            n_actions=self.config["n_actions"],
                                            #action_bounds=self.config["action_bounds"],
                                            n_hidden_filters=self.config["n_hiddens"]).to(self.device)

        self.q_value_network1 = QvalueNetwork(n_states=self.n_states + self.n_skills,
                                              n_actions=self.config["n_actions"],
                                              n_hidden_filters=self.config["n_hiddens"]).to(self.device)

        self.q_value_network2 = QvalueNetwork(n_states=self.n_states + self.n_skills,
                                              n_actions=self.config["n_actions"],
                                              n_hidden_filters=self.config["n_hiddens"]).to(self.device)

        self.value_network = ValueNetwork(n_states=self.n_states + self.n_skills,
                                          n_hidden_filters=self.config["n_hiddens"]).to(self.device)

        self.value_target_network = ValueNetwork(n_states=self.n_states + self.n_skills,
                                                 n_hidden_filters=self.config["n_hiddens"]).to(self.device)
        self.hard_update_target_network()

        self.discriminator = Discriminator(n_states=self.n_states, n_skills=self.n_skills,
                                           n_hidden_filters=self.config["n_hiddens"]).to(self.device)

        self.mse_loss = torch.nn.MSELoss()
        self.cross_ent_loss = torch.nn.CrossEntropyLoss()

        self.value_opt = Adam(self.value_network.parameters(), lr=self.config["lr"])
        self.q_value1_opt = Adam(self.q_value_network1.parameters(), lr=self.config["lr"])
        self.q_value2_opt = Adam(self.q_value_network2.parameters(), lr=self.config["lr"])
        self.policy_opt = Adam(self.policy_network.parameters(), lr=self.config["lr"])
        self.discriminator_opt = Adam(self.discriminator.parameters(), lr=self.config["lr"])
        self.writer = SummaryWriter()

    def choose_action(self, states):
        states = np.expand_dims(states, axis=0)
        print("choose_action states type: ",states.dtype)
        states = from_numpy(states).float().to(self.device)
        ##
        with torch.no_grad():
            logits = self.policy_network(states)
            action_probs = F.softmax(logits, dim=-1)
            action_dist = torch.distributions.Categorical(action_probs)
            action = action_dist.sample()
        
        return action.cpu().numpy()[0]

        logits, log_probs, action_probs = self.policy_network.sample_or_likelihood(states)
        print(f"Action: {action_probs}, ")
        
        return action_index.detach().cpu().numpy()[0].astype(int)

    def store(self, state, z, done, action, next_state):
        state = from_numpy(state).float().to("cpu")
        z = torch.tensor([int(z)], dtype=torch.long).to("cpu")
        done = torch.BoolTensor([done]).to("cpu")
        action = torch.LongTensor([action]).to("cpu")  # Changed to LongTensor for discrete actions
        next_state = from_numpy(next_state).float().to("cpu")
        self.memory.add(state, z, done, action, next_state)
     
    def unpack(self, batch):
        batch = Transition(*zip(*batch))
        
        states = torch.cat(batch.state).view(self.batch_size, self.n_states + self.n_skills).to(self.device)
        zs = torch.cat(batch.z).view(self.batch_size, 1).long().to(self.device)
        dones = torch.cat(batch.done).view(self.batch_size, 1).to(self.device)
          
        # Keep actions as indices (not one-hot) for discrete Q-networks
 
        actions = torch.cat(batch.action).view(self.batch_size, 1).long().to(self.device)
        next_states = torch.cat(batch.next_state).view(self.batch_size, self.n_states + self.n_skills).to(self.device)
        print(f"Unpacked zs shape: {zs.shape}, dtype: {zs.dtype}")
        print(f"Unpacked zs min/max: {zs.min().item()}/{zs.max().item()}")
        return states, zs, dones, actions, next_states

    def train(self):
        if len(self.memory) < self.batch_size:
            print(f"Memory size: {len(self.memory)}, need {self.batch_size} to start training")
            return None
        else:
            print(f"TRAINING STEP: Memory has {len(self.memory)} samples")
            batch = self.memory.sample(self.batch_size)
            states, zs, dones, actions, next_states = self.unpack(batch)

            print("=== DETAILED TENSOR DEBUG ===")
            print(f"Raw zs shape: {zs.shape}")
            print(f"Raw zs dtype: {zs.dtype}")
            print(f"Raw zs first 5 values: {zs[:5].flatten()}")
            print(f"Raw zs unique values: {zs.unique()}")
    
            max_skill = zs.max().item()
            min_skill = zs.min().item()
            unique_skills = zs.unique()
            print(f"Batch skills - Min: {min_skill}, Max: {max_skill}, Unique: {unique_skills.tolist()}")
    
            #print("zs: ",zs)
            print(f"Skill range in batch: [{min_skill}, {max_skill}], should be < {self.n_skills}")
            
             
            actions_flat = actions.squeeze(-1) if actions.dim() > 1 else actions
            zs_flat = zs.squeeze(-1)
            p_z = torch.from_numpy(self.p_z).float().to(self.device)
            
            with torch.no_grad():  # Don't track gradients for policy loss Q-values
                q1_for_policy = self.q_value_network1(states)
                q2_for_policy = self.q_value_network2(states)
                q_for_policy = torch.min(q1_for_policy, q2_for_policy)


            # Calculating the value target
            logits = self.policy_network(states)  # [batch_size, n_actions]
            action_probs = F.softmax(logits, dim=-1)  # [batch_size, n_actions]
            log_probs = F.log_softmax(logits, dim=-1)  # [batch_size, n_actions]
        

            print("action probs shape: ", action_probs.shape)
            print("qval: ", q_for_policy.shape)
            print("action_probs shape: ", action_probs.shape)
 
            print("action_probs shape: ",action_probs.shape)
            # Expected Q-value and entropy (exact computation for discrete)
            expected_q = (action_probs * q_for_policy).sum(dim=-1)
            entropy = -(action_probs * log_probs).sum(dim=-1)
            policy_loss = -(expected_q + self.config["alpha"] * entropy).mean()
            self.writer.add_scalar(f"Policy Loss, skill{zs[0]}", policy_loss)
            self.writer.add_scalar(f"Entropy, skill {zs[0]}", entropy)
     
            # Calculating the Q-Value target
            with torch.no_grad():
                target_value = expected_q + self.config["alpha"] * entropy
                #target_q = self.config["reward_scale"] * rewards.float() + \
                 #       self.config["gamma"] * self.value_target_network(next_states) * (~dones)
            value = self.value_network(states).squeeze(-1)  # [batch_size]
            value_loss = self.mse_loss(value, target_value)
            self.writer.add_scalar(f"Value Network Loss, skill {zs[0]}", value_loss)
            raw_next_states = next_states[:, :self.n_states]
            discriminator_logits = self.discriminator(raw_next_states)
            discriminator_loss = self.cross_ent_loss(discriminator_logits, zs_flat)
            self.writer.add_scalar(f"Discriminator Loss, skill {zs[0]}", discriminator_loss)

            p_z_gathered = p_z.gather(-1, zs).float()   # [batch_size, 1]
            log_q_z_ns = log_softmax(discriminator_logits, dim=-1).float()  # [batch_size, n_skills]
            log_q_z_ns_gathered = log_q_z_ns.gather(-1, zs).float()  # [batch_size, 1]
            
            rewards = (log_q_z_ns_gathered - torch.log(p_z_gathered + 1e-6)).squeeze(-1)
            rewards=rewards.float()  # [batch_size]

            # === Q-LOSS COMPUTATION ===
            # Compute fresh Q-values for Q-loss (separate from policy loss)
            q1 = self.q_value_network1(states)
            q2 = self.q_value_network2(states)

            with torch.no_grad():
                next_value = self.value_target_network(next_states).squeeze(-1)  # [batch_size]
                #reward_component = (self.config["reward_scale"] * rewards).float()
                reward_component = (self.config["reward_scale"] * rewards).float()
                gamma_component = (self.config["gamma"] * next_value * (~dones.squeeze(-1).bool())).float()
                target_q = reward_component + gamma_component # [batch_size]
            
            # Get Q-values for taken actions
            q1_taken = q1.gather(1, actions_flat.unsqueeze(-1)).squeeze(-1)  # [batch_size]
            q2_taken = q2.gather(1, actions_flat.unsqueeze(-1)).squeeze(-1)  # [batch_size]

            q1_loss = self.mse_loss(q1_taken, target_q)
            q2_loss = self.mse_loss(q2_taken, target_q)
             
            raw_states = states[:, :self.n_states]  # [batch_size, n_states]
            discriminator_logits_current = self.discriminator(raw_states)
            discriminator_loss = self.cross_ent_loss(discriminator_logits_current, zs_flat)
            # Optimization steps
            self.policy_opt.zero_grad()
            policy_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.policy_network.parameters(), max_norm=1.0)
            self.policy_opt.step()

            self.value_opt.zero_grad()
            value_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.value_network.parameters(), max_norm=1.0)
            self.value_opt.step()

            self.q_value1_opt.zero_grad()
            q1_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.q_value_network1.parameters(), max_norm=1.0)
            self.q_value1_opt.step()

            self.q_value2_opt.zero_grad()
            q2_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.q_value_network2.parameters(), max_norm=1.0)
            self.q_value2_opt.step()

            self.discriminator_opt.zero_grad()
            discriminator_loss.backward()
            self.discriminator_opt.step() 
            self.soft_update_target_network(self.value_network, self.value_target_network)

            return -discriminator_loss.item()
             

    def soft_update_target_network(self, local_network, target_network):
        for target_param, local_param in zip(target_network.parameters(), local_network.parameters()):
            target_param.data.copy_(self.config["tau"] * local_param.data +
                                    (1 - self.config["tau"]) * target_param.data)

    def hard_update_target_network(self):
        self.value_target_network.load_state_dict(self.value_network.state_dict())
        #print("state dict value target network: ", self.value_network.state_dict())
        self.value_target_network.eval()

    def get_rng_states(self):
        return torch.get_rng_state(), self.memory.get_rng_state()

    def set_rng_states(self, torch_rng_state, random_rng_state):
        torch.set_rng_state(torch_rng_state.to("cpu"))
        self.memory.set_rng_state(random_rng_state)

    def set_policy_net_to_eval_mode(self):
        self.policy_network.eval()

    def set_policy_net_to_cpu_mode(self):
        self.device = torch.device("cpu")
        self.policy_network.to(self.device)
