# Multi-UAV-MEC-Simulator
Simulation environment for AoI minimization in multi-UAV MEC systems

## Overview

This repository provides the simulation environment used for studying Age of Information (AoI) minimization in a multi-UAV mobile edge computing (MEC) system under a three-tier architecture (user devices – UAVs – cloud server).

The repository currently contains the full simulation environment used in the manuscript, including system modeling, task generation, mobility dynamics, wireless communication modeling, and energy consumption evaluation.

The federated multi-agent reinforcement learning algorithm (HetFed-MASAC) is currently being further organized and documented. The complete training framework and algorithmic implementation are planned for public release following formal publication of the paper to ensure long-term maintainability, clarity, and reproducibility.

## Repository Structure

- `mec_marl_env.py`  
  OpenAI Gym-based multi-agent environment wrapper, including:
  - Action and observation space definitions  
  - Environment step and reset logic  
  - AoI-based reward computation  
  - Energy statistics recording  

- `mec_system_model.py`  
  Core multi-UAV MEC system modeling, including:
  - Edge device (UAV) dynamics  
  - Sensor data generation process  
  - Wireless channel modeling (LoS/NLoS)  
  - Data collection and offloading mechanisms  
  - AoI evolution  
  - Energy consumption modeling  
