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

column_list = [
    "queue_type",                   # q->queue_type
    "qdelay_reference",             # pprms->qdelay_ref
    "tupdate",                      # pprms->tupdate
    "max_burst",                    # pprms->max_burst
    "max_ecn_threshold",            # pprms->max_ecnth
    "alpha_coefficient",            # pprms->alpha
    "beta_coefficient",             # pprms->beta
    "flags",                        # pprms->flags
    "burst_allowance",              # pst->burst_allowance
    "drop_probability",             # pst->drop_prob
    "current_queue_delay",          # pst->current_qdelay
    "previous_queue_delay",         # pst->qdelay_old
    "accumulated_probability",      # pst->accu_prob
    "measurement_start_time",       # pst->measurement_start
    "average_dequeue_time",         # pst->avg_dq_time
    "dequeue_count",                # pst->dq_count
    "status_flags",                 # pst->sflags
    "total_packets",                # q->stats.tot_pkts
    "total_bytes",                  # q->stats.tot_bytes
    "queue_length",                 # q->stats.length
    "length_in_bytes",              # q->stats.len_bytes
    "total_drops",                  # q->stats.drops
    "dequeue_action",               # dequeue_action
]



# Define the list of columns to include
columns_to_use = [
    'queue_type', 
    'burst_allowance',
    'drop_probability',
    'current_queue_delay',
    'accumulated_probability',
    'average_dequeue_time',
    'length_in_bytes',
    'total_drops'
]

import pickle
import numpy as np
import pandas as pd

import numpy as np
import pandas as pd
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
        dict_output_path (str): Path to save the experience pool as a dictionary in a pickle file (default: 'exp_pool_dict.pkl').

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
    # df.to_csv(csv_output_path, index=False)
    # print(f"DataFrame saved successfully to: {csv_output_path}")

    # # Step 3: (Optional) Save the experience pool as a dictionary in a pickle file
    # exp_pool_dict = {
    #     'states': exp_pool.states,
    #     'actions': exp_pool.actions,
    #     'rewards': exp_pool.rewards,
    #     'dones': exp_pool.dones
    # }

    # with open(dict_output_path, 'wb') as f:
    #     pickle.dump(exp_pool_dict, f)
    # print(f"Experience pool saved as a dictionary to: {dict_output_path}")

    # Return the DataFrame
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

# def find_nearest_length(df, user_input):
#     print("()" * 40)
#     print("df in function find_nearest_length:")
    
    
#     if df.empty:
#         # Handle the empty DataFrame case
#         print("DataFrame is empty, returning None.")
#         return None  # Return None or another suitable default value

#     # Ensure 'state_6' column exists and is numeric
#     if 'state_6' not in df.columns:
#         print("'state_6' column not found in DataFrame, returning None.")
#         return None
    
#     try:
#         df['state_6'] = pd.to_numeric(df['state_6'], errors='coerce')
#     except Exception as e:
#         print(f"Error converting 'state_6' column to numeric: {e}")
#         return None

#     if df['state_6'].isnull().all():
#         print("All values in 'state_6' are NaN, returning None.")
#         return None
    
#     # Calculate the absolute difference with user_input
#     print("user_input:", user_input)
#     try:
#         nearest_idx = (df['state_6'] - user_input).abs().idxmin()
#         print("nearest_idx:", nearest_idx)
#     except Exception as e:
#         print(f"Error finding nearest index: {e}")
#         return None

#     # Ensure the index is valid
#     if nearest_idx is None or nearest_idx not in df.index:
#         print("No valid index found, returning None.")
#         return None

#     return nearest_idx





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



# Define the list of columns to include
columns_to_use = [
    'queue_type', 
    'burst_allowance',
    'drop_probability',
    'current_queue_delay',
    'accumulated_probability',
    'average_dequeue_time',
    'length_in_bytes',
    'total_drops'
]


def testenvsim(args, model, exp_pool, target_return, loss_fn ,process_reward_fn=None, seed=0):
    if process_reward_fn is None:
        process_reward_fn = lambda x: x
    

    exp_dataset = ExperienceDataset(exp_pool, gamma=args.gamma, scale=args.scale, max_length=args.w, sample_step=1)

    test_losses = []
    logs = dict()
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

    max_ep_len = 100
    start_iloc=0
    row = df.iloc[start_iloc]
    test_start = time.time()
    datapoint_idx = 0

    state_columns = [f'state_{i}' for i in range(len(exp_pool.states[0]))]

    for ep_index in range(max_ep_len):
        # df.to_csv("second_save.csv")
        start_iloc+=1
        
        # row = df.iloc[start_iloc]

        row = df.iloc[datapoint_idx]
        print("row,",row)
        
        print("--" * 40)
        state = np.array(row[state_columns], dtype=np.float32)
        current_action = row['actions']
        reward=row['rewards']
        done=0
        batch = [state],[current_action],[reward],[done]
        # print("batch",batch)
        test_loss, states, actions, returns, timesteps, labels, actions_pred1, actions_pred = test_step(args, model, loss_fn, batch)
        test_losses.append(test_loss.item())

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
            # start_iloc+=1
            # row = df.iloc[start_iloc]
            continue  # Skip to the next iteration of the loop
        print("current_queue_delay",states[0][0][3])
        print("length_in_bytes",states[0][0][6])
        datapoint_idx = find_nearest_length(df_ats, float(states[0][0][6]))
        print("datapoint",datapoint_idx)

        # Next start datapoint of episode will be the nearest datapoint,
        # we can find from the database
        start_iloc = datapoint_idx

        # CPU and RAM usage
        cpu_usage = psutil.cpu_percent()
        memory_info = psutil.virtual_memory()

        # GPU usage
        gpus = GPUtil.getGPUs()
        gpu_usage = gpus[0].load * 100 if gpus else 0
        vram_usage = gpus[0].memoryUsed if gpus else 0

        # Disk I/O stats
        current_disk_io = psutil.disk_io_counters()
        disk_read_speed = current_disk_io.read_bytes / (1024 * 1024)  # MB/s
        disk_write_speed = current_disk_io.write_bytes / (1024 * 1024)  # MB/s

        # # perform gradient accumulation update
        # test_loss = test_loss / self.grad_accum_steps
        # test_loss.backward()
        # torch.nn.utils.clip_grad_norm_(self.model.parameters(), .25)
        # if ((step + 1) % self.grad_accum_steps == 0) or (step + 1 == dataset_size):
        #     self.optimizer.step()
        #     self.optimizer.zero_grad(set_to_none=True)
        #     if self.lr_scheduler is not None:
        #         self.lr_scheduler.step()
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
            'labels': tensor_to_list(labels),
            'CPU Usage': cpu_usage,
            'RAM Usage': memory_info.percent,
            'GPU Usage': gpu_usage,
            'VRAM Usage': vram_usage,
            'Disk Read Speed (MB/s)': disk_read_speed,
            'Disk Write Speed (MB/s)': disk_write_speed,
        }
        custom_logs['steps'].append(step_logs)
    # Save custom logs to a JSON file for this epoch
    with open(f'./Logs/ custom_logs_epoch_testseq_1.json', 'w') as file:
        json.dump(custom_logs, file, indent=4)


