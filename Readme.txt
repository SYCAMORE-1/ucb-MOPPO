
This repository contains the official implementation of our work titled "UCB-driven Utility Function Search for Multi-objective Reinforcement Learning," accepted to the ECML PKDD 2025 research track.

To execute the proposed method and baselines on Mujoco V2, please ensure you are running a Linux kernel. For Windows users, follow the `command.txt` instructions to build an independent Docker environment.

For experiments on the latest Mujoco-V4 environments, install the updated Gymnasium library:

pip install gymnasium

and refer to the latest API documentation:

https://gymnasium.farama.org/environments/mujoco/

To run the proposed methods on various tasks and random seeds, use the following command:

python ucb.py --eidx 0 --sidx 0