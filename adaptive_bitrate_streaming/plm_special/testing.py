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

def find_nearest_length(df, user_input):
    if df.empty:
        # Handle the empty DataFrame case
        print("DataFrame is empty, returning None.")
        return None  # Return None or another suitable default value

    # Calculate the absolute difference with user input
    nearest_idx = (df['length_in_bytes'] - user_input).abs().idxmin()
    
    if nearest_idx is None or nearest_idx >= len(df):
        print("No valid index found, returning None.")
        return None  # Return None or another suitable default value

    return df.iloc[nearest_idx]


# Function to convert true actions
def convert_to_classes(action):
    if action < 0.5:
        return 0
    elif action < 1.5:
        return 1
    else:
        return 2


df=pd.read_csv("./artifacts/exp_pools/lmprocesseddata.txt",names=column_list,header=None)


class Tester:
    def __init__(self, args, model, optimizer, exp_dataset, loss_fn, device, batch_size=1, grad_accum_steps=1, lr_scheduler=None):
        self.args = args
        self.model = model
        self.optimizer = optimizer
        self.exp_dataset = exp_dataset
        self.loss_fn = loss_fn
        self.device = device
        self.batch_size = batch_size
        self.grad_accum_steps = grad_accum_steps
        self.lr_scheduler = lr_scheduler
        
        self.exp_dataset_info = Munch(exp_dataset.exp_dataset_info)
        self.dataloader = DataLoader(exp_dataset, batch_size, shuffle=True, pin_memory=True)
    
    def tensor_to_list(self, tensor):
        # Detach the tensor and then convert it to a NumPy array and then to a list
        return tensor.detach().cpu().numpy().tolist()


    def test_epoch(self, epoch, report_loss_per_steps=100):
        test_losses = []
        logs = dict()
        custom_logs = {'steps': []}

        test_start = time.time()
        dataset_size = len(self.dataloader)
        start_iloc=0
        row = df.iloc[start_iloc]
        
        testing_steps = 100



        for step in range(testing_steps):
            print("-" * 40)
            state = np.array(row[columns_to_use], dtype=np.float32)
            current_action = row['dequeue_action']
            reward=row['current_queue_delay']
            done=0
            results_df = pd.DataFrame(columns=columns_to_use)

            # batch = state,current_action,reward,done
            batch = [state],[current_action],[reward],[done]
            test_loss, states, actions, returns, timesteps, labels, actions_pred1, actions_pred = self.test_step(batch,epoch,step)
            test_losses.append(test_loss.item())

            print("actions_pred",actions_pred)
            print("actions_pred.shape",actions_pred.shape)

            new_action = actions_pred.detach().cpu().numpy().argmax(axis=1).flatten()

                # First check for queue_type
            df_qt= df[df['queue_type']== int(states[0][0][0])]
            df_ats= df_qt[df_qt['dequeue_action']== int(actions[0])]
            # print("df_ats.shape",df_ats.shape)
            # print("states[0][0][6]",states[0][0][6])
                # Skip if df_ats is empty
            if df_ats.empty:
                print("df_ats is empty, skipping this batch.")
                start_iloc+=1
                row = df.iloc[start_iloc]
                continue  # Skip to the next iteration of the loop


            datapoint = find_nearest_length(df_ats, float(states[0][0][6]))

            if datapoint is None:
                    print("No valid datapoint found, skipping this batch.")
                    continue  # Skip to the next iteration if no valid datapoint
            
            print("DataPoint")
            print(datapoint[columns_to_use])
            row = datapoint

            # Convert the selected columns of the datapoint to a DataFrame
            datapoint_df = pd.DataFrame([datapoint])

            # Use pd.concat to append the new DataFrame to results_df
            results_df = pd.concat([results_df, datapoint_df], ignore_index=True)
            
            # CPU and RAM usage
            cpu_usage = psutil.cpu_percent()
            memory_info = psutil.virtual_memory()

            # GPU usage
            gpus = GPUtil.getGPUs()
            gpu_usage1 = gpus[0].load * 100 if gpus else 0
            vram_usage1 = gpus[0].memoryUsed if gpus else 0

            gpu_usage2 = gpus[1].load * 100 if gpus else 0
            vram_usage2 = gpus[1].memoryUsed if gpus else 0

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
            print(f'Step {step} - test_loss.item() {test_loss.item()}')
            
            # Log step information
            step_logs = {
                'step': step,
                'test_loss': test_loss.item(),
                'actions_pred1': self.tensor_to_list(actions_pred1),
                'actions_pred': self.tensor_to_list(actions_pred),
                'states': self.tensor_to_list(states),
                'actions': self.tensor_to_list(actions),
                'returns': self.tensor_to_list(returns),
                'timestamps': str(time.time()),
                'timesteps': self.tensor_to_list(timesteps),
                'labels': self.tensor_to_list(labels),
                'CPU Usage': cpu_usage,
                'RAM Usage': memory_info.percent,
                'GPU1 Usage': gpu_usage1,
                'VRAM1 Usage': vram_usage1,
                'GPU2 Usage': gpu_usage2,
                'VRAM2 Usage': vram_usage2,
                'Disk Read Speed (MB/s)': disk_read_speed,
                'Disk Write Speed (MB/s)': disk_write_speed,
            }
            custom_logs['steps'].append(step_logs)

            if step % report_loss_per_steps == 0:                
                mean_test_loss = np.mean(test_losses)
                print(f'Step {step} - mean test loss {mean_test_loss:>9f}')

        logs['time/testing'] = time.time() - test_start
        logs['testing/test_loss_mean'] = np.mean(test_losses)
        logs['testing/test_loss_std'] = np.std(test_losses)
        
        # Save custom logs to a JSON file for this epoch
        with open(f'custom_logs_epoch_test_{epoch}.json', 'w') as file:
            json.dump(custom_logs, file, indent=4)

        return logs, test_losses

    def test_step(self, raw_batch, epoch, step):
        # Assuming raw_batch is a tuple of numpy arrays or lists
        states, actions, returns, timesteps = raw_batch

        # # Print original state shape
        # print("Original states:", states)
        # print("Original states.shape:", states[0].shape)  # Assuming states is a list of arrays

        # Convert states to tensor and ensure correct shape
        states = torch.tensor(states[0], dtype=torch.float32).to(self.device).unsqueeze(0)  # Shape [1, 8]
        # print("Tensor states:", states)
        # print("Tensor states.shape:", states.shape)  # Should be [1, 8]

        # Convert actions, returns, and timesteps to tensors
        actions = torch.tensor(actions, dtype=torch.float32).to(self.device)  # Shape [1, 1]
        returns = torch.tensor(returns, dtype=torch.float32).to(self.device)  # Shape [1, 1]
        timesteps = torch.tensor(timesteps, dtype=torch.int32).to(self.device)  # Shape [1, 1]

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
        states, actions, returns, timesteps, labels = process_batch(batch, device=self.device)

        # Predict actions using the model
        actions_pred1 = self.model(states, actions, returns, timesteps)

        # Permute for loss calculation
        actions_pred = actions_pred1.permute(0, 2, 1)
        loss = self.loss_fn(actions_pred, labels)

        return loss, states, actions, returns, timesteps, labels, actions_pred1, actions_pred


