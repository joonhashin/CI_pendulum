import torch
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, BatchNormalization
from keras.callbacks import ModelCheckpoint, EarlyStopping

from data import preprocess_transition
from ddpg_agent import Agent


def build_model(input_size, hidden_1_size, hidden_2_size, output_size):
    model = Sequential()
    model.add(Dense(hidden_1_size, input_dim =input_size, activation='relu'))
    # model.add(BatchNormalization())
    model.add(Dense(hidden_2_size, activation='relu'))
    # model.add(BatchNormalization())
    model.add(Dense(output_size))
    model.compile(loss='mean_squared_error', optimizer='adam', metrics = ['mae', 'mse'])
    return model
        
def dataset_me(policy_path, output_path, num_traj= 1000, max_t=200, init_state=None):
    """ Takes behaviour policy from 'policy_path' and create dataset of trajectories.
    Saved in 'data/' folder + output path
    """
    
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
            sa_pair = np.append(state, action)
            next_state = EnvModel.predict(sa_pair)
            traj.extend([action, 0, next_state])
            state = next_state
        dataset.append(traj)
    np.save('data/'+ output_path, dataset)
    return dataset

if __name__ =="__main__":
    # Load Data
    train_path = 'train.npy'
    train_set = np.load('data/'+train_path, allow_pickle=True)
    x_train, y_train = preprocess_transition(train_set)
    
    # Model Parameters
    input_size = 4
    hidden_1_size = 64
    hidden_2_size = 64
    output_size= 3
    
    # Callbacks
    checkpoint_path = 'model/env_best.hdf5'
    cp_callback = ModelCheckpoint(
        filepath = checkpoint_path,
        save_best_only = True,
        monitor = 'val_loss',
        mode = 'min')
    early_stop = EarlyStopping(monitor = 'val_loss', mode = 'min', patience = 10)


    EnvModel = build_model(input_size, hidden_1_size, hidden_2_size, output_size)
    EnvModel.compile(loss='mean_squared_error', optimizer='adam', metrics = ['mae', 'mse'])
    EnvModel.summary()
    history = EnvModel.fit(x_train, 
                 y_train, 
                 validation_split = 0.2, 
                 batch_size=32, 
                 epochs=100,
                 callbacks = [early_stop, cp_callback])
    
    