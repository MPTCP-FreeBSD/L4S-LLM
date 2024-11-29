import numpy as np
import torch
import time
import json
import psutil
import GPUtil
from munch import Munch
from torch.utils.data import DataLoader

from plm_special.utils.utils import process_batch
from plm_special.data.dataset import ExperienceDataset
import random

class ExperiencePool:
    """
    Experience pool for collecting trajectories.
    """
    def __init__(self):
        self.states = []
        self.actions = []
        self.rewards = []
        self.dones = []

    def add(self, state, action, reward, done):
        self.states.append(state)  # sometimes state is also called obs (observation)
        self.actions.append(action)
        self.rewards.append(reward)
        self.dones.append(done)

    def __len__(self):
        return len(self.states)


def tensor_to_list(tensor):
        # Detach the tensor and then convert it to a NumPy array and then to a list
        return tensor.detach().cpu().numpy().tolist()

# Function to reduce the experience pool by a fraction
def reduce_experience_pool(exp_pool, fraction=1/8):
    # Calculate the number of experiences to keep
    total_experiences = len(exp_pool)
    num_to_keep = int(total_experiences * fraction)
    print("num_to_keep",num_to_keep)
    
    # Randomly sample the indices to keep
    sampled_indices = random.sample(range(total_experiences), num_to_keep)
    
    # Create a new experience pool to hold the reduced data
    reduced_exp_pool = ExperiencePool()
    
    # Add the randomly sampled experiences to the reduced pool
    for i in sampled_indices:
        state = exp_pool.states[i]
        action = exp_pool.actions[i]
        reward = exp_pool.rewards[i]
        done = exp_pool.dones[i]
        
        reduced_exp_pool.add(state=state, action=action, reward=reward, done=done)
    
    return reduced_exp_pool


def ntesting(args, model, exp_pool, loss_fn, batch_size=1):
    def test_step(batch, epoch, step):
        states, actions, returns, timesteps, labels = process_batch(batch, device=args.device)
        actions_pred1 = model(states, actions, returns, timesteps)
        actions_pred = actions_pred1.permute(0, 2, 1)
        loss = loss_fn(actions_pred, labels)
        return loss, states, actions, returns, timesteps, labels, actions_pred1, actions_pred

    def tensor_to_list(tensor):
        # Detach the tensor and then convert it to a NumPy array and then to a list
        return tensor.detach().cpu().numpy().tolist()
    

    exp_pool = reduce_experience_pool(exp_pool, fraction=0.01)

    exp_dataset = ExperienceDataset(exp_pool, gamma=args.gamma, scale=args.scale, max_length=1, sample_step=1)
    print(f"Number of samples in exp_dataset: {len(exp_dataset)}")
    dataloader = DataLoader(exp_dataset, batch_size, shuffle=True, pin_memory=True)

    test_losses = []
    logs = dict()
    

    test_start = time.time()
    dataset_size = len(dataloader)
    for epoch in range(args.num_epochs):
        custom_logs = {'steps': []}
        print('='* 20, f'Testing Iteration #{epoch}', '=' * 20)
        print('>' * 10, 'Testing Information:')
        for step, batch in enumerate(dataloader):
            # Pass epoch and step explicitly to the test_step function
            test_loss, states, actions, returns, timesteps, labels, actions_pred1, actions_pred = test_step(batch, epoch, step)
            test_losses.append(test_loss.item())
            time_start_step = time.time()

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

            # Perform gradient accumulation update
            test_loss = test_loss / args.grad_accum_steps
            print(f'Step {step} - test_loss.item() {test_loss.item()}')

            # Log step information
            step_logs = {
                'step': step,
                'test_loss': test_loss.item(),
                'actions_pred1': tensor_to_list(actions_pred1),
                'actions_pred': tensor_to_list(actions_pred),
                'states': tensor_to_list(states),
                'actions': tensor_to_list(actions),
                'returns': tensor_to_list(returns),
                'timestamps': str(time.time()),
                'timestamps_each_step': str(time.time() - time_start_step),
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

            if step % 50 == 0:                
                mean_test_loss = np.mean(test_losses)
                print(f'Step {step} - mean test loss {mean_test_loss:>9f}')

        logs['time/testing'] = time.time() - test_start
        logs['testing/test_loss_mean'] = np.mean(test_losses)
        logs['testing/test_loss_std'] = np.std(test_losses)

        # Save custom logs to a JSON file for this epoch
        with open(f'custom_logs_epoch_test_{epoch}.json', 'w') as file:
            json.dump(custom_logs, file, indent=4)
        

    return logs, test_losses

