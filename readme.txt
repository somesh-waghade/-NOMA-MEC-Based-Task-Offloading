NOMA-MEC Based Task Offloading Algorithm in UAV-assisted IoV Networks

This repository contains the simulation implementation for a novel task offloading strategy designed to optimize resource allocation in Internet of Vehicles (IoV) environments using Unmanned Aerial Vehicles (UAVs) acting as Mobile Edge Computing (MEC) servers. The core mechanism leverages Non-Orthogonal Multiple Access (NOMA) for efficient communication and a Genetic Algorithm (GA) for optimal task allocation.

The implementation is based on the paper: NOMA-MEC Based Task Offloading Algorithm in UAV-assisted IoV Networks

AUTHORS
-------
Paper Authors: Tingyue Xiao, Pengfei Du, Haosong Gou, Gaoyi Zhang
Implementation: Somesh Waghde (252CS030), Brijesh Saroj (252CS007)

PROJECT OVERVIEW
----------------
The project addresses the challenge of insufficient local processing power in modern vehicles for computationally intensive tasks (e.g., autonomous driving data processing) within B5G/6G telematics scenarios.

The proposed solution, the UAV-assisted IoV Task Offloading Algorithm (TOA):
* Utilizes UAVs equipped with MEC servers to provide powerful edge computing resources.
* Employs NOMA to improve spectrum utilization and allow multiple users to share resources efficiently.
* Uses a Genetic Algorithm (GA) to dynamically determine the optimal binary offloading strategy (offload or local) for each vehicle's task.
* Objective: To minimize the total task processing time under strict constraints on computational resources and energy consumption.

KEY FEATURES AND ALGORITHMS
---------------------------

1. System Model
The simulation models two main entities:
* VehicleNode: Represents the IoV vehicle, with parameters like task data size (D_j), processing density (omega_j), maximum tolerable latency (T_max), and local CPU capability (C_loc).
* UAVNode: Represents the MEC server, characterized by its position, flight speed (nu_i), and computational power (f_mec).

2. Task Offloading Algorithm (TOA)
The core mechanism compares the time taken for local processing versus MEC offloading to make the binary decision for task j:
* Local Processing Time: T_loc = (D_j * omega_j) / C_loc
* MEC Offloading Time (Transmission + MEC Processing + UAV Flight Time):
  T_mec = (D_j / R_j) + (D_j * omega_j / f_mec) + (d_i_j / v_i)

3. Optimization Strategy
A Genetic Algorithm (GA) is used to solve the minimization problem:
* Encoding: The offloading strategy for all tasks is encoded as a binary gene sequence (e.g., 010101).
* Fitness Function: The system's total task processing time, which the GA seeks to minimize through standard genetic operators (crossover and mutation).

CODE STRUCTURE
--------------
The simulation is implemented in Python and uses the 'simpy' library for discrete-event simulation.

* noma.py: Contains the implementation of the proposed GA-Optimized Task Offloading Algorithm (TOA).
* alloffload.py: Contains the implementation of the All-Offload Baseline strategy for comparison.

INSTALLATION AND SETUP
----------------------
Prerequisites:
You need Python 3.x installed.

Dependencies:
The main libraries required for the simulation are 'numpy' and 'simpy'.
Command: pip install numpy simpy

HOW TO RUN THE SIMULATION
-------------------------
The simulation is configured to compare the GA-Optimized approach against the naive "All-Offload" baseline for a network of 20 tasks and 5 UAVs.

1. Run the Proposed TOA (GA-Optimized):
   Command: python noma.py
   (This runs the run_ga_leaf_sim() function and prints the optimized strategy, minimum total time, and energy consumption.)

2. Run the Baseline (All-Offload):
   Command: python alloffload.py
   (This runs the run_all_offload_sim() function and prints the performance metrics for the baseline strategy.)

PERFORMANCE RESULTS
-------------------
Simulation results (as described in the implementation report):
* Total Task Processing Time: The TOA scheme reduces the average delay by up to 56% compared to the All-offload strategy.
* Energy Consumption: The GA-Optimized approach achieves significantly lower overall energy consumption by intelligently avoiding unnecessary transmission and UAV flight energy costs.
* Conclusion: The GA approach successfully minimizes total task processing time while satisfying computational and energy constraints, outperforming static allocation methods.