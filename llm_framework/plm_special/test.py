import numpy as np
import torch
import time
import json
import psutil
import GPUtil
from munch import Munch
from torch.utils.data import DataLoader

from plm_special.utils.utils import process_batch




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

        # self.model.train()
        for step, batch in enumerate(self.dataloader):
            test_loss, states, actions, returns, timesteps, labels, actions_pred1, actions_pred = self.test_step(batch,epoch,step)
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

            # perform gradient accumulation update
            test_loss = test_loss / self.grad_accum_steps
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
                'timestamps_each_step': str(time.time() - time_start_step),
                'timesteps': self.tensor_to_list(timesteps),
                'labels': self.tensor_to_list(labels),
                'CPU Usage': cpu_usage,
                'RAM Usage': memory_info.percent,
                'GPU Usage': gpu_usage,
                'VRAM Usage': vram_usage,
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
        with open(f'./Logs/custom_logs_epoch_test_{epoch}.json', 'w') as file:
            json.dump(custom_logs, file, indent=4)

        return logs, test_losses

    def test_step(self, batch,epoch,step):
        states, actions, returns, timesteps, labels = process_batch(batch, device=self.device)
        actions_pred1 = self.model(states, actions, returns, timesteps)
        actions_pred = actions_pred1.permute(0, 2, 1)
        loss = self.loss_fn(actions_pred, labels) 
        return loss, states, actions, returns, timesteps, labels, actions_pred1, actions_pred