# Active Queue Management (AQM) and L4S-LLM

## Overview

### Active Queue Management (AQM)
Active Queue Management (AQM) refers to a set of algorithms executed on routers and switches that aim to detect and mitigate network congestion. AQM works by monitoring either the instantaneous or average queue size to identify early signs of congestion, which typically occurs when network traffic exceeds the network's capacity. 

When network congestion arises, AQM algorithms help prevent performance degradation by adjusting the flow of data. These algorithms make decisions about packet handling—such as dropping or marking packets—based on the observed queue state, effectively clearing buffer space to avoid congestion. In some cases, AQM algorithms may set a congestion notification bit in packets, alerting clients to reroute their traffic through alternate paths to alleviate congestion in the affected router.

## Code Structure

The repository is organized into various directories and files for efficient navigation and management of the project:

### `./data/` Directory
- **exp_pool/**: Contains experience pool files, which are used for training Large Language Models (LLMs).
- **results/**: Stores the result files generated from experiments.
- **ft_plms/**: Stores the fine-tuned (adapted) LLMs checkpoints.

### `./plm_special/` Directory
- **./data/**: Contains scripts for data-related operations used in training and LLM fine-tuning and training.
  - **exp_pool.py**: Implements the experience pool to collect data trajectories.
  - **dataset.py**: Defines a dataset class that wraps the experience pool and prepares data for training.
- **./models/**: Contains the model-related code for L4S-LLM.
  - **state_encoder.py**: Implements the feature encoder for state encoding.
  - **gpt2.py, llama.py, opt.py, mistral.py, t5.py**: Custom implementations of various LLM architectures.
  - **low_rank.py**: Implements low-rank matrix factorization techniques to optimize the model.
  - **rl_policy.py**: Defines the Transformer-based reinforcement learning policy used in L4S-LLM.
- **./utils/**: Utility scripts that assist with various tasks.
  - **plm_utils.py**: Contains utilities for loading pre-trained LLMs.
  - **utils.py**: General-purpose utilities for data processing.
- **trainer.py**: Handles the fine-tuning and training process of fine-tuning the LLMs.
- **test_seq.py**: Contains scripts for evaluating sequence generation performance of adapted LLMs.
- **test.py**: Includes scripts for evaluating the overall performance of the adapted LLMs.

### `run_plm.py`
The main script for running the L4S-LLM fine-tuning and training and evaluation processes.

## Environment Setup

### Setting Up the Environment for L4S-LLM

1. Create a Conda environment for L4S-LLM:

   ```bash
   conda create -n llmenv python==3.10.15


2. Then install the following depdendencies:

   ```
   # To install pytorch if you haven't, You might want to go to official link if you want to use CPU version:
   conda install pytorch==2.1.0 torchvision==0.16.0 torchaudio==2.1.0 pytorch-cuda=11.8 -c pytorch -c nvidia
   # To install in Conda Environment:
   conda install conda-forge::transformers==4.34.1
   
   conda install conda-forge::munch
   
   pip install openprompt==1.0.1   
   conda install conda-forge::peft==0.6.2   
   conda install conda-forge::huggingface_hub==0.17.3   
   conda install conda-forge::accelerate==0.24.1   
   conda install conda-forge::scikit-learn==1.3.2   
   conda install conda-forge::huggingface_hub==0.19.1
      
   # If needed to use pip, follow the below instructions:   
   pip install torch==2.1.0
   pip install transformers==4.34.1
   (Optional)pip install numpy==1.24.4   
   pip install munch==4.0.0   
   pip install openprompt==1.0.1   
   pip install peft==0.6.2

   ```


# Running
## How to run L4S-LLM
To run L4S-LLM, first we need to download some LLMs. For example, if you want to use Llama2-7b as the foundation model, please download Llama2-7b in the directory: `../downloaded_plms/llama2/base`. The file LLM Model Info and how to download.txt contains information on whoe to downlaod these models.

**Finetune LLM**

If you want to finetune or train LLM, please run the following command:
```sh
python run_plm.py --adapt --grad-accum-steps 32 --plm-type llama --plm-size base --rank 128 --device cuda:0 --lr 0.0001 --warmup-steps 2000 --num-epochs 80 --eval-per-epoch 2 
```
This command will finetune Llama2 on the default experience pool we provided at `data/exp_pools/exp_pool.pkl`.


**Evaluate LLM**

If you want to evaluate the performance of the finetuned LLM, please run the following command:
```sh
python run_plm.py --test --grad-accum-steps 32 --plm-type llama --plm-size base --rank 128 --device cuda:0 --lr 0.0001 --warmup-steps 2000 --num-epochs 80 --eval-per-epoch 2
```
You can also specify the path to the finetuned LLM with argument `--model-dir`:
```sh
python run_plm.py --test --plm-type llama --plm-size base --rank 128 --device cuda:0 --model-dir you_finetune_llm_dir
```
