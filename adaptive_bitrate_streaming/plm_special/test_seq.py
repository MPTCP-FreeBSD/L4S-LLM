import numpy as np
import torch
import time
import json
import psutil
import GPUtil
from munch import Munch
from torch.utils.data import DataLoader
import pandas as pd

from plm_special.utils.utils import process_batch
from plm_special.data.dataset import ExperienceDataset
import random
import pickle


def tensor_to_list(tensor):
        # Detach the tensor and then convert it to a NumPy array and then to a list
        return tensor.detach().cpu().numpy().tolist()

def convert_exp_pool_to_dataframe(exp_pool, csv_output_path='exp_pool_data.csv', dict_output_path='exp_pool_dict.pkl'):
    """
    Converts the given experience pool into a pandas DataFrame.
    Optionally saves the DataFrame to a CSV file and the experience pool as a dictionary to a pickle file.

    Args:
        exp_pool (object): The experience pool object containing states, actions, rewards, and dones.
        csv_output_path (str): Path to save the resulting DataFrame as a CSV file (default: 'exp_pool_data.csv').

    Returns:
        pd.DataFrame: The DataFrame representation of the experience pool.
    """
    
    # Step 1: Convert the Experience Pool to a DataFrame
    
    # Create state column names based on the length of each state vector
    state_columns = [f'state_{i}' for i in range(len(exp_pool.states[0]))]  # Assuming each state is a 1D array
    
    # Flatten the states into individual columns
    expanded_states = np.array([state for state in exp_pool.states])
    
    # Create the DataFrame with expanded states
    df = pd.DataFrame(expanded_states, columns=state_columns)
    
    # Add actions, rewards, and dones as columns to the DataFrame
    df['actions'] = exp_pool.actions
    df['rewards'] = exp_pool.rewards
    df['dones'] = exp_pool.dones

    # Step 2: Save the DataFrame to a CSV file
    df.to_csv(csv_output_path, index=False)
    print(f"DataFrame saved successfully to: {csv_output_path}")
    return df



def find_nearest_length(df, user_input):
    print("||||||||||||||||"*40)
    print("df in function find_nearest_length")
    if df.empty:
        # Handle the empty DataFrame case
        print("DataFrame is empty, returning None.")
        return None  # Return None or another suitable default value

    # Calculate the absolute difference with user input
    print("user_input",user_input)
    # nearest_idx = (df['state_6'] - user_input).abs().idxmin()
    
    # df_sort = df.iloc[(df['state_6']-user_input).abs().argsort()[:1]]
    # nearest_idx = (df['state_6']-user_input).abs().argsort()[:1]
    # print("df_sort",df_sort.head(2))
    # print("11nearest_idx",nearest_idx)


    nearest_idx = (df['state_6'] - user_input).abs().idxmin()
    print("22nearest_idx",nearest_idx)

    

    if nearest_idx >= len(df):
        print("Outside df_ats limits")
    
    if nearest_idx is None:
        print("No valid index found, returning None.")
        return None  # Return None or another suitable default value
    
    print("||||||||||||||||"*40)

    return nearest_idx





def test_step(args, model, loss_fn, raw_batch):
        # Assuming raw_batch is a tuple of numpy arrays or lists
        states, actions, returns, timesteps = raw_batch

        # # Print original state shape
        # print("Original states:", states)
        # print("Original states.shape:", states[0].shape)  # Assuming states is a list of arrays

        # Convert states to tensor and ensure correct shape
        states = torch.tensor(states[0], dtype=torch.float32).to(args.device).unsqueeze(0)  # Shape [1, 8]
        # print("Tensor states:", states)
        # print("Tensor states.shape:", states.shape)  # Should be [1, 8]

        # Convert actions, returns, and timesteps to tensors
        actions = torch.tensor(actions, dtype=torch.float32).to(args.device)  # Shape [1, 1]
        returns = torch.tensor(returns, dtype=torch.float32).to(args.device)  # Shape [1, 1]
        timesteps = torch.tensor(timesteps, dtype=torch.int32).to(args.device)  # Shape [1, 1]

        # # Print shapes after conversion
        # print("Actions tensor:", actions)
        # print("Actions tensor shape:", actions.shape)  # Should be [1, 1]
        # print("Returns tensor:", returns)
        # print("Returns tensor shape:", returns.shape)  # Should be [1, 1]
        # print("Timesteps tensor:", timesteps)
        # print("Timesteps tensor shape:", timesteps.shape)  # Should be [1, 1]

        # Create a batch with the correctly formatted tensors
        # Wrap states in a list to avoid TypeError in process_batch
        batch = ([states], [actions], [returns], [timesteps])  # Ensure states is a list

        # Call process_batch
        states, actions, returns, timesteps, labels = process_batch(batch, device=args.device)

        # Predict actions using the model
        actions_pred1 = model(states, actions, returns, timesteps)

        # Permute for loss calculation
        actions_pred = actions_pred1.permute(0, 2, 1)
        loss = loss_fn(actions_pred, labels)

        return loss, states, actions, returns, timesteps, labels, actions_pred1, actions_pred



def testenvsim(args, model, exp_pool, target_return, loss_fn ,process_reward_fn=None, seed=0):
    if process_reward_fn is None:
        process_reward_fn = lambda x: x
    

    exp_dataset = ExperienceDataset(exp_pool, gamma=args.gamma, scale=args.scale, max_length=args.w, sample_step=1)

    custom_logs = {'steps': []}

    df =  convert_exp_pool_to_dataframe(exp_pool)
    # print(df.columns)
    # print(df.shape)
    print(df.describe())
    # print(df.head(5))
    # print("**"*10)
    # print(df.tail(5))
    # print("*-*-"*80)
    # df.to_csv("first_save.csv")

    max_ep_len = 1600
    llm_freq = 10

    row = df.iloc[0]
    test_start = time.time()
    cur_datapoint_idx = 0
    start_iloc = 0

    state_columns = [f'state_{i}' for i in range(len(exp_pool.states[0]))]

    for ep_index in range(max_ep_len):
        # df.to_csv("second_save.csv")
        row = df.iloc[start_iloc]
        print("row,",row)
        
        print("--" * 40)
        state = np.array(row[state_columns], dtype=np.float32)
        current_action = row['actions']
        reward=row['rewards']
        done=0
        batch = [state],[current_action],[reward],[done]
        # print("batch",batch)
        test_loss, states, actions, returns, timesteps, labels, actions_pred1, actions_pred = test_step(args, model, loss_fn, batch)

        # print("actions_pred",actions_pred)
        # print("actions_pred.shape",actions_pred.shape)

        new_action = actions_pred.detach().cpu().numpy().argmax(axis=1).flatten()
        
        # print("new_action",new_action)
        # print("type(new_action)",type(new_action))

        # print("new_action",new_action.astype(int))
        # print("type(new_action)",type(new_action.astype(int)))


        # print("new_action",new_action.item())
        # print("type(new_action)",type(new_action.item()))

        df_qt= df[df['state_0']== int(states[0][0][0])]
        
        df_ats= df_qt[df_qt['actions']== int(new_action)]
        # print("df_ats.head(3)")
        # print(df_ats.head(3))
        # print(df_ats.describe())

        if df_ats.empty:
            print("df_ats is empty, skipping this batch.")
            continue  # Skip to the next iteration of the loop
        print("current_queue_delay",states[0][0][3])
        print("length_in_bytes",states[0][0][6])
        print("packet_length",states[0][0][8])
        new_queue_length = float(states[0][0][6])
        if new_action == 0 or new_action == 2:
            new_queue_length = (float(states[0][0][6]) + float(states[0][0][8]))
        cur_datapoint_idx = find_nearest_length(df_ats, new_queue_length)
        print("datapoint",cur_datapoint_idx)
        if ep_index % llm_freq == 0:
            start_iloc = cur_datapoint_idx
        else:
            start_iloc+=1

        # Next start datapoint of episode will be the nearest datapoint,
        # we can find from the database

        print(f'Step {ep_index} - test_loss.item() {test_loss.item()}')
        
        # Log step information
        step_logs = {
            'step': ep_index,
            'test_loss': test_loss.item(),
            'actions_pred1': tensor_to_list(actions_pred1),
            'actions_pred': tensor_to_list(actions_pred),
            'states': tensor_to_list(states),
            'actions': tensor_to_list(actions),
            'returns': tensor_to_list(returns),
            'timestamps': str(time.time()),
            'timesteps': tensor_to_list(timesteps),
            'labels': tensor_to_list(labels)
        }
        custom_logs['steps'].append(step_logs)
    # Save custom logs to a JSON file for this epoch
    with open(f'./Logs/eval_logs_llm.json', 'w') as file:
        json.dump(custom_logs, file, indent=4)
    start_iloc = 0
    custom_logs = {'steps': []}
 # To Save Original Sequence
    for ep_index in range(max_ep_len):
        # df.to_csv("second_save.csv")
        print("start_iloc",start_iloc)
        row = df.iloc[start_iloc]
        print("row,",row)
        
        print("--" * 40)
        state = np.array(row[state_columns], dtype=np.float32)
        current_action = row['actions']
        reward=row['rewards']
        done=0
        batch = [state],[current_action],[reward],[done]
        # print("batch",batch)
        test_loss, states, actions, returns, timesteps, labels, actions_pred1, actions_pred = test_step(args, model, loss_fn, batch)


        print(f'Step {ep_index} - test_loss.item() {test_loss.item()}')
        
        # Log step information
        step_logs = {
            'step': ep_index,
            'test_loss': test_loss.item(),
            'actions_pred1': tensor_to_list(actions_pred1),
            'actions_pred': tensor_to_list(actions_pred),
            'states': tensor_to_list(states),
            'actions': tensor_to_list(actions),
            'returns': tensor_to_list(returns),
            'timestamps': str(time.time()),
            'timesteps': tensor_to_list(timesteps),
            'labels': tensor_to_list(labels)
        }
        start_iloc+=1
        custom_logs['steps'].append(step_logs)
    # Save custom logs to a JSON file for this epoch
    with open(f'./Logs/eval_logs_original.json', 'w') as file:
        json.dump(custom_logs, file, indent=4)


