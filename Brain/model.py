from abc import ABC
import torch
from torch import nn
from torch.nn import functional as F
from torch.distributions import Normal
from torch.nn.functional import log_softmax
from torch.distributions import Categorical
def init_weight(layer, initializer="he normal"):
    if initializer == "xavier uniform":
        nn.init.xavier_uniform_(layer.weight)
    elif initializer == "he normal":
        nn.init.kaiming_normal_(layer.weight)


class Discriminator(nn.Module, ABC):
    def __init__(self, n_states, n_skills, n_hidden_filters=256):
        super(Discriminator, self).__init__()
        self.n_states = n_states
        self.n_skills = n_skills
        self.n_hidden_filters = n_hidden_filters

        self.hidden1 = nn.Linear(in_features=self.n_states, out_features=self.n_hidden_filters)
        init_weight(self.hidden1)
        self.hidden1.bias.data.zero_()
        self.hidden2 = nn.Linear(in_features=self.n_hidden_filters, out_features=self.n_hidden_filters)
        init_weight(self.hidden2)
        self.hidden2.bias.data.zero_()
        self.q = nn.Linear(in_features=self.n_hidden_filters, out_features=self.n_skills)
        init_weight(self.q, initializer="xavier uniform")
        self.q.bias.data.zero_()

    def forward(self, states):
        x = F.tanh(self.hidden1(states))
        x = F.tanh(self.hidden2(x))

        logits = self.q(x)
        return logits


class ValueNetwork(nn.Module, ABC):
    def __init__(self, n_states, n_hidden_filters=256):
        super(ValueNetwork, self).__init__()
        self.n_states = n_states
        self.n_hidden_filters = n_hidden_filters

        self.hidden1 = nn.Linear(in_features=self.n_states, out_features=self.n_hidden_filters)
        init_weight(self.hidden1)
        self.hidden1.bias.data.zero_()
        self.hidden2 = nn.Linear(in_features=self.n_hidden_filters, out_features=self.n_hidden_filters)
        init_weight(self.hidden2)
        self.hidden2.bias.data.zero_()
        self.value = nn.Linear(in_features=self.n_hidden_filters, out_features=1)
        init_weight(self.value, initializer="xavier uniform")
        self.value.bias.data.zero_()

    def forward(self, states):
        x = F.relu(self.hidden1(states))
        x = F.relu(self.hidden2(x))
        return self.value(x)

class QvalueNetwork(nn.Module, ABC):
    def __init__(self, n_states, n_actions, n_hidden_filters=256):
        super(QvalueNetwork, self).__init__()
        self.n_states = n_states
        self.n_hidden_filters = n_hidden_filters
        self.n_actions = n_actions

        self.hidden1 = nn.Linear(in_features=self.n_states, out_features=self.n_hidden_filters)
        init_weight(self.hidden1)
        self.hidden1.bias.data.zero_()
        self.hidden2 = nn.Linear(in_features=self.n_hidden_filters, out_features=self.n_hidden_filters)
        init_weight(self.hidden2)
        self.hidden2.bias.data.zero_()
        # Output Q-values for all actions (discrete case)
        self.q_value = nn.Linear(in_features=self.n_hidden_filters, out_features=self.n_actions)
        #self.q_value = nn.Linear(in_features=self.n_hidden_filters, out_features=1)
        init_weight(self.q_value, initializer="xavier uniform")
        self.q_value.bias.data.zero_()

    def forward(self, states ):
        print("states dims QvalueNet: ", states.shape)
    
        
        x = F.relu(self.hidden1(states))
        x = F.relu(self.hidden2(x))
        q_values = self.q_value(x)  # [batch_size, n_actions]
        return  q_values
       
class PolicyNetwork(nn.Module, ABC):
    def __init__(self, n_states, n_actions, action_bounds= None, n_hidden_filters=256):
        super(PolicyNetwork, self).__init__()
        self.n_states = n_states
        self.n_hidden_filters = n_hidden_filters
        self.n_actions = n_actions
         

        self.hidden1 = nn.Linear(in_features=self.n_states, out_features=self.n_hidden_filters)
        init_weight(self.hidden1)
        self.hidden1.bias.data.zero_()
        self.hidden2 = nn.Linear(in_features=self.n_hidden_filters, out_features=self.n_hidden_filters)
        init_weight(self.hidden2)
        self.hidden2.bias.data.zero_()
        
        # for discrete
        self.action_logits = nn.Linear(in_features=self.n_hidden_filters, out_features=self.n_actions)
        init_weight(self.action_logits, initializer="xavier uniform")
        self.action_logits.bias.data.zero_()
        self.action_logits.bias.data.uniform_(-0.1, 0.1)
    
    def forward(self, states, temperature=0.8):
        #x = F.tanh(self.hidden1(states))
        #x = F.tanh(self.hidden2(x))
        print("NN layer states shape: ", states.shape)
        x = F.tanh(self.hidden1(states))
        x = F.tanh(self.hidden2(x))
        logits = self.action_logits(x)
        #print("Policy network, logits shape: ", logits.shape)
        

        return logits  

    def sample_or_likelihood(self, states,):
       
        logits = self(states)
        action_probs = F.softmax(logits, dim=-1)
        action_dist = Categorical(action_probs)
        action = action_dist.sample()
        log_prob = action_dist.log_prob(action)
         
        #action_item = action_index.item()
        #action_item = torch.argmax(probs).item()
         
        return logits, log_prob, action_probs #(action * self.action_bounds[1]) #.clamp_(self.action_bounds[0], self.action_bounds[1]), log_prob
        
         