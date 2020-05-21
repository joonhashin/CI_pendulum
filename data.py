import gym
import torch
import os.path
import numpy as np
from collections import deque

from keras.models import Sequential
from keras.layers import Dense, BatchNormalization
from keras.callbacks import ModelCheckpoint, EarlyStopping

from ddpg_agent import Agent
from my_pendulumenv import PendulumEnv
from cqr_interval import CQR_rf_interval


def ddpg_train(n_episodes=1000, max_t=200, print_every=100):
    env = gym.make('Pendulum-v0')
    env.seed(2)
    agent = Agent(state_size=3, action_size=1, random_seed=2)
    scores_deque = deque(maxlen=print_every)
    scores = []
    for i_episode in range(1, n_episodes+1):
        state = env.reset()
        agent.reset()
        score = 0
        for t in range(max_t):
            action = agent.act(state)
            next_state, reward, done, _ = env.step(action)
            agent.step(state, action, reward, next_state, done)
            state = next_state
            score += reward
            if done:
                break 
        scores_deque.append(score)
        scores.append(score)
        print('\rEpisode {}\tAverage Score: {:.2f}'.format(i_episode, np.mean(scores_deque)), end="")
        if i_episode % print_every == 0:
            print('\rEpisode {}\tAverage Score: {:.2f}'.format(i_episode, np.mean(scores_deque)))
            torch.save(agent.actor_local.state_dict(), 'policy/actor{}.pth'.format(i_episode))
            torch.save(agent.critic_local.state_dict(), 'policy/critic{}.pth'.format(i_episode))
    return scores

def dataset(policy_path, output_path, num_traj= 1000, max_t=200, init_state=None):
    """ Takes behaviour policy from 'policy_path' and create dataset of trajectories.
    Saved in 'data/' folder + output path
    """
    env = PendulumEnv()
    env.seed(2)
    agent = Agent(state_size=3, action_size=1, random_seed=2)
    agent.actor_local.load_state_dict(torch.load(policy_path))
    dataset = []
    for i in range(num_traj):
        if init_state==None:
            state = env.reset()
        else:
            state = env.reset(init_state)
        agent.reset()
        traj = [state]
        for t in range(max_t):
            action = agent.act(state)
            next_state, reward, done, _ = env.step(action)
            traj.extend([action, reward, next_state])
            state = next_state
        dataset.append(traj)
    np.save('data/'+ output_path, dataset)
    return dataset

def dataset_mp(policy0_path, policy1_path, output_path, num_traj= 1000, max_t=200, init_state=None, alpha = 0.0):
    env = PendulumEnv()
    env.seed(2)
    agent0 = Agent(state_size=3, action_size=1, random_seed=2)
    agent1 = Agent(state_size=3, action_size=1, random_seed=2)
    agent0.actor_local.load_state_dict(torch.load(policy0_path))
    agent1.actor_local.load_state_dict(torch.load(policy1_path))
    dataset = []
    for i in range(num_traj):
        if init_state==None:
            state = env.reset()
        else:
            state = env.reset(init_state)
        agent0.reset()
        agent1.reset()
        traj = [state]
        for t in range(max_t):
            action = agent0.act(state) * (1 - alpha) + agent1.act(state) * alpha
            next_state, reward, done, _ = env.step(action)
            traj.extend([action, reward, next_state])
            state = next_state
        dataset.append(traj)
    np.save('data/'+ output_path, dataset)
    return dataset

def build_envmodel(input_size, hidden_1_size, hidden_2_size, output_size):
    model = Sequential()
    model.add(Dense(hidden_1_size, input_dim =input_size, activation='relu'))
    # model.add(BatchNormalization())
    model.add(Dense(hidden_2_size, activation='relu'))
    # model.add(BatchNormalization())
    model.add(Dense(output_size))
    return model

def train_envmodel(train_path, input_size, hidden_1_size, hidden_2_size, output_size):
    train_set = np.load('data/'+train_path, allow_pickle=True)
    x_train, y_train = preprocess_transition(train_set)
    # Callbacks
    checkpoint_path = 'model/env_best.hdf5'
    cp_callback = ModelCheckpoint(
        filepath = checkpoint_path,
        save_best_only = True,
        monitor = 'val_loss',
        mode = 'min')
    early_stop = EarlyStopping(monitor = 'val_loss', mode = 'min', patience = 10)

    EnvModel = build_envmodel(input_size, hidden_1_size, hidden_2_size, output_size)
    EnvModel.compile(loss='mean_squared_error', optimizer='adam', metrics = ['mae', 'mse'])
    EnvModel.summary()
    history = EnvModel.fit(x_train, 
                 y_train, 
                 validation_split = 0.2, 
                 batch_size=32, 
                 epochs=100,
                 callbacks = [early_stop, cp_callback])
    return EnvModel
    
def dataset_em(policy_path, output_path, env_path, num_traj= 1000, max_t=200, 
               init_state=None, hidden_1 = 64, hidden_2 = 64):
    env = PendulumEnv()
    env.seed(2)
    agent = Agent(state_size=3, action_size=1, random_seed=2)
    agent.actor_local.load_state_dict(torch.load(policy_path))
    estimated_env = build_envmodel(input_size = 4, hidden_1_size = hidden_1, hidden_2_size = hidden_2, output_size = 3)
    estimated_env.load_weights('model/'+env_path)
    dataset = []
    for i in range(num_traj):
        if init_state==None:
            state = env.reset()
        else:
            state = env.reset(init_state)
        agent.reset()
        traj = [state]
        for t in range(max_t):
            action = agent.act(state)
            sa_pair = np.append(state, action)
            next_state = estimated_env.predict(np.array([sa_pair,]))
            traj.extend([action, 0, next_state])
            state = next_state
        dataset.append(traj)
    np.save('data/'+ output_path, dataset)
    return dataset

def performance(rewards, gamma=0.99):
    performance = 0
    i = 0
    for reward in rewards:
        performance += (gamma**i)*reward
        i += 1
    return performance

def preprocess(dataset, max_t):
    X=[]
    y=[]
    for traj in dataset:
        for i in range(int(len(traj)/3)):
            if i ==max_t:
                break
            # (state, action) pair
            X.append(np.append(traj[3*i], traj[3*i+1]))           
            # reward
            y.append(traj[3*i+2])
    
    x = np.asarray(X)
    y = np.asarray(y)
    return x, y

def preprocess_traj(dataset, max_t):
    X=[]
    y=[]
    for traj in dataset:
        x_traj = []
        y_traj = []
        for i in range(int(len(traj)/3)):
            if i ==max_t:
                break
            x_traj.extend(traj[3*i])
            x_traj.extend(traj[3*i+1])
            y_traj.append(traj[3*i+2])
        X.append(x_traj)
        y.append(performance(y_traj))
    x = np.asarray(X)
    y = np.asarray(y)
    return x, y

def preprocess_transition(dataset):
    X=[]
    y=[]
    for traj in dataset:
        for i in range(int(len(traj)/3)):
            # (state, action) pair
            X.append(np.append(traj[3*i], traj[3*i+1]))           
            # next state
            y.append(traj[3*i+3])
    
    x = np.asarray(X)
    y = np.asarray(y)
    return x, y

def cqr_opve_step(train_path, test_path, num_train = 1000, max_t = 200, delta = 0.1, gamma=0.99):
    data_train = np.load('data/'+train_path, allow_pickle = True)
    data_train = data_train[:num_train] # select num_train trajectories
    data_test = np.load('data/'+test_path, allow_pickle = True)
    
    miscoverage = 1-(1-delta)**(1/float(max_t))
    x_train, y_train = preprocess(data_train, max_t)

    for i in range(len(data_test)):
        x_test, y_true = preprocess(data_test, max_t)
        y_upper, y_lower = CQR_rf_interval(x_train, y_train, x_test, y_true, miscoverage)

    perf =[performance(y_true), performance(y_lower), performance(y_upper)]
    return perf

def cqr_opve_traj(train_path, test_path, num_train = 1000, max_t = 200, traj_len = 1000, delta = 0.1, gamma=0.99):
    data_train = np.load('data/'+train_path, allow_pickle = True)
    data_train = data_train[:num_train] # select num_train trajectories
    data_test = np.load('data/'+test_path, allow_pickle = True)
    
    miscoverage = delta
    x_train, y_train = preprocess_traj(data_train, max_t)
    x_test, y_true = preprocess_traj(data_test, max_t)
    y_upper, y_lower = CQR_rf_interval(x_train, y_train, x_test, y_true, miscoverage)

    perf =[y_true[0], y_upper[0], y_lower[0]]
    return perf


if __name__ == "__main__":
    if not os.path.isfile('policy/actor100.pth'):
        ddpg_train()
    policy0 = 'policy/actor100.pth'
    train_path = 'train.npy'
    if not os.path.isfile('data/'+train_path):
        dataset(policy0, train_path)
    
    policy1 = 'policy/actor1000.pth'
    test1_path = 'test1.npy'
    if not os.path.isfile('data/'+test1_path):
        dataset(policy1, test1_path, num_traj = 1, init_state=[np.pi, 0])
        test2_path =  'test2.npy'
        dataset(policy1, test2_path, num_traj = 1, init_state=[np.pi/2, 0])
        test3_path =  'test3.npy'
        dataset(policy1, test3_path, num_traj = 1, init_state=[0, 0])
        test4_path =  'test4.npy'
        dataset(policy1, test4_path, num_traj = 1, init_state=[-np.pi/2, 0])
        
    mixed_policy_0 = 'train_mp_0.npy'
    if not os.path.isfile('data/'+mixed_policy_0):
        dataset_mp(policy0, policy1, mixed_policy_0, alpha=0.0)
        mixed_policy_2 = 'train_mp_2.npy'
        dataset_mp(policy0, policy1, mixed_policy_2, alpha=0.2)
        mixed_policy_4 = 'train_mp_4.npy'
        dataset_mp(policy0, policy1, mixed_policy_4, alpha=0.4)
        mixed_policy_6 = 'train_mp_6.npy'
        dataset_mp(policy0, policy1, mixed_policy_6, alpha=0.6)
        mixed_policy_8 = 'train_mp_8.npy'
        dataset_mp(policy0, policy1, mixed_policy_8, alpha=0.8)        
        mixed_policy_10 = 'train_mp_10.npy'
        dataset_mp(policy0, policy1, mixed_policy_10, alpha=1.0)
    
    behaviour_agnostic = 'train_ba.npy'
    if not os.path.isfile('data/'+behaviour_agnostic):
        train_mp_0 = np.load('data/train_mp_0.npy', allow_pickle=True)
        train_mp_2 = np.load('data/train_mp_2.npy', allow_pickle=True)
        train_mp_4 = np.load('data/train_mp_4.npy', allow_pickle=True)
        train_mp_6 = np.load('data/train_mp_6.npy', allow_pickle=True)
        train_mp_8 = np.load('data/train_mp_8.npy', allow_pickle=True)
        # train_mp_10 = np.load('data/train_mp_10.npy', allow_pickle=True)
        args = (train_mp_0[:100],train_mp_2[:100],train_mp_4[:100],train_mp_6[:100],train_mp_8[:100])
        new = np.concatenate(args)
        np.save('data/'+behaviour_agnostic, new)
        
    longer_train = 'long_train.npy'
    longer_test1 = 'long_test1.npy'
    longer_test2 = 'long_test2.npy'
    longer_test3 = 'long_test3.npy'
    longer_test4 = 'long_test4.npy'    
    if not os.path.isfile('data/'+longer_train):
        dataset(policy0, longer_train, max_t = 1000)
        dataset(policy1, longer_test1, num_traj = 1, max_t = 1000, init_state=[np.pi, 0])
        dataset(policy1, longer_test2, num_traj = 1, max_t = 1000, init_state=[np.pi/2, 0])
        dataset(policy1, longer_test3, num_traj = 1, max_t = 1000, init_state=[0, 0])
        dataset(policy1, longer_test4, num_traj = 1, max_t = 1000, init_state=[-np.pi/2, 0])        
    
    estmodel_path = 'em_test1.npy'
    # Model Parameters
    input_size = 4
    hidden_1_size = 64
    hidden_2_size = 64
    output_size= 3
    if not os.path.isfile('data/'+estmodel_path):
        if not os.path.isfile('model/env_best.hdf5'):
            train_envmodel(train_path, input_size, hidden_1_size, hidden_2_size, output_size)
        dataset_em(policy1, 'em_test1.npy','env_best.hdf5', num_traj = 1, init_state=[np.pi, 0])
        dataset_em(policy1, 'em_test2.npy','env_best.hdf5', num_traj = 1, init_state=[np.pi/2, 0])
        dataset_em(policy1, 'em_test3.npy','env_best.hdf5', num_traj = 1, init_state=[0, 0])
        dataset_em(policy1, 'em_test4.npy','env_best.hdf5', num_traj = 1, init_state=[-np.pi/2, 0])
        