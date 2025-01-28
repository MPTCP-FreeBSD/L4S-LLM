# L4S-LLM :
---------------
This repository contains the source code for the L4S-LLM frameowrk modified to support Low Latency, Low Loss, Scalable throughput (L4S) architecture in FreeBSD Network Stack. 

The increasing network traffic complexity and rising demands on ultra-low latency communications require smarter packet management schemes. 
Given the limitations of conventional Active Queue Management (AQM) and Deep Learning-based algorithms, 
we propose leveraging Large Language Models (LLMs) to address challenges related to generalization and the complexities of model engineering.

![L4S-LLM Architecture](https://github.com/user-attachments/assets/acf40bfe-9ccf-4be0-bdc2-b684d3caf288)

We have the following folders:
1. llm_framework - Main folder which contains the main L4S-LLM code and framework
2. Results Viz - contains code to visualise results from L4S-LLM training and evaluation
3. Misc - COntains files, folders and information used while development (Not Important)

## Related Repositories

- **[FreeBSD-L4S-Experiments](https://github.com/MPTCP-FreeBSD/FreeBSD-L4S-Experiments)**:  
  This repository contains the setup and configuration scripts for our FreeBSD network environment, including the implementation of the new L4S AQM algorithm for FreeBSD.

- **[Graphs-Viz-L4S-LLM](https://github.com/MPTCP-FreeBSD/Graphs-Viz-L4S-LLM)**:  
  This repository provides the code to generate visual graphs from our experimental results, aiding in data analysis and result presentation.

- **[LLM_Gen_Exp_Pool](https://github.com/MPTCP-FreeBSD/LLM_Gen_Exp_Pool)**:  
  This repository includes code to create distinct training and evaluation experience pools, which are used to feed into our LLM (Large Language Model) for analysis and model training.




Researchers: 
- Deol Satish
- [Shiva Raj Pokhrel](https://www.deakin.edu.au/about-deakin/people/shiva-pokhrel) <shiva.pokhrel@deakin.edu.au>
- Jonathan Kua
- Anwar Walid (Amazon Science)

