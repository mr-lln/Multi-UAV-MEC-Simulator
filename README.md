# Multi-UAV-MEC-Simulator
Simulation environment for AoI minimization in multi-UAV MEC systems

## Overview

This project studies Age of Information (AoI) minimization in a multi-UAV mobile edge computing (MEC) system under a three-tier architecture (user devices – UAVs – cloud server).

The repository currently contains the full simulation environment used in the manuscript, including system modeling, task generation, mobility dynamics, wireless communication modeling, and energy consumption evaluation.

The federated multi-agent reinforcement learning algorithm (HetFed-MASAC) is being further organized and documented. The complete training framework and algorithmic implementation will be released after formal publication of the paper to ensure long-term maintainability and clarity.

## Repository Structure

- `environment/`  
  Multi-UAV MEC simulation environment, including:
  - UAV mobility model  
  - User task generation process  
  - Wireless channel model (LoS/NLoS)  
  - Bandwidth allocation logic  
  - AoI computation  
  - Energy consumption model  
