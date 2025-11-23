import numpy as np
import simpy

try:
    from leaf.infrastructure import Node
    from leaf.power import PowerModelNode
except ImportError:
    class Node:
        def __init__(self, name, cu=None, power_model=None):
            self.name = name
            self.cu = cu
    class PowerModelNode:
        def __init__(self, max_power=100.0): pass


# === VEHICLE NODE (same as GA) ===
class VehicleNode(Node):
    def __init__(self, env, name, pos, C_loc, D_j, omega_j, T_max, p_tx):
        power_model = PowerModelNode(max_power=50.0)
        super().__init__(name, cu=C_loc, power_model=power_model)
        self.env = env
        self.V_pos = np.array(pos)
        self.C_j_loc = C_loc
        self.D_j = D_j
        self.omega_j = omega_j
        self.T_j_max = T_max
        self.p_tx = p_tx
        self.is_offloading = False


# === UAV NODE (same as GA) ===
class UAVNode(Node):
    def __init__(self, env, name, pos, nu_i, p_u_i):
        power_model = PowerModelNode(max_power=100.0)
        super().__init__(name, cu=1e10, power_model=power_model)
        self.env = env
        self.U_pos = np.array(pos)
        self.nu_i = nu_i
        self.p_u_i = p_u_i


# === Baseline Optimization Class (UAV ENERGY ADDED) ===
class UavIoVTaskOffloadingAllOffload:
    def __init__(self, num_tasks, num_uavs, lambda_penalty=0.2):
        self.J = num_tasks
        self.I = num_uavs
        self.h0 = 1e-4
        self.B = 10e6
        self.N0 = 1e-20
        self.f_mec = 100e9
        self.lambda_penalty = lambda_penalty 
        # --- NEW ENERGY PARAMETERS (SAME AS GA) ---
        self.alpha = 1e-27
        self.p_mec_factor = 1.0 
        self.P_UAV_op = 150.0 # NEW: UAV Operation Power (W)
        self.rejection_penalty = 50.0

    def load_system_state(self, vehicle_nodes, uav_nodes):
        self.V_nodes = vehicle_nodes
        self.U_nodes = uav_nodes
        self.J = len(vehicle_nodes)
        self.I = len(uav_nodes)
        self.D_j = np.array([v.D_j for v in vehicle_nodes])
        self.omega_j = np.array([v.omega_j for v in vehicle_nodes])
        self.T_j_max = np.array([v.T_j_max for v in vehicle_nodes])
        self.C_j_loc = np.array([v.C_j_loc for v in vehicle_nodes])
        self.p_tx = np.array([v.p_tx for v in vehicle_nodes])
        self.V_pos = np.array([v.V_pos for v in vehicle_nodes])
        self.U_pos = np.array([u.U_pos for u in uav_nodes])
        self.nu_i = np.array([u.nu_i for u in uav_nodes])
        self.p_u_i = np.array([u.p_u_i for u in uav_nodes])

    def calculate_offload_delay(self, j, i_uav):
        d_i_j_3d = np.linalg.norm(self.U_pos[i_uav] - self.V_pos[j])
        h_i_j = self.h0 / d_i_j_3d**2
        G_i_j = np.sum(
            [self.p_u_i[x] * (self.h0 / np.linalg.norm(self.U_pos[x] - self.V_pos[j])**2)
             for x in range(self.I) if x != i_uav]
        )
        gamma_i_j = (self.p_tx[j] * h_i_j) / (G_i_j + self.B * self.N0)
        R_j = self.B * np.log2(1 + gamma_i_j)
        T_j_mec = (self.D_j[j] / R_j) + (self.D_j[j] * self.omega_j[j] / self.f_mec)
        d_i_j_horiz = np.linalg.norm(self.U_pos[i_uav, :2] - self.V_pos[j, :2])
        T_fly_j = d_i_j_horiz / self.nu_i[i_uav]
        
        T_j_off = T_j_mec + T_fly_j
        return T_j_off, R_j

    def calculate_task_energy(self, j, R_j, T_j_off):
        # E_j_loc
        E_j_loc = self.alpha * self.D_j[j] * self.omega_j[j] * (self.C_j_loc[j]**2)
        
        # E_j_mec_veh (Vehicle Trans. + MEC Proc.)
        E_j_mec_veh = (self.D_j[j] / R_j) * self.p_tx[j] + (self.D_j[j] * self.omega_j[j] / self.f_mec) * self.p_mec_factor
        
        # E_j_UAV (UAV Operation Energy) - NEW
        E_j_UAV = self.P_UAV_op * T_j_off
        
        E_j_offload = E_j_mec_veh + E_j_UAV # Total Offload Energy
        
        return E_j_loc, E_j_offload

    def run_all_offload_baseline(self):
        T_total_sum = 0
        E_total_sum = 0
        offloaded_accepted = 0
        
        i_uav = 0 
        
        for j in range(self.J):
            T_j_off, R_j = self.calculate_offload_delay(j, i_uav)
            E_j_loc, E_j_offload = self.calculate_task_energy(j, R_j, T_j_off)
            
            if T_j_off <= self.T_j_max[j]:
                T_total_sum += T_j_off
                E_total_sum += E_j_offload 
                offloaded_accepted += 1
            else:
                T_total_sum += self.rejection_penalty 
                E_total_sum += E_j_offload 
        
        offload_count = self.J
        penalty = self.lambda_penalty * (offload_count / self.J)
        
        min_total_time = (T_total_sum / self.J) + penalty
        rejected = self.J - offloaded_accepted
        
        return min_total_time, offloaded_accepted, 0, rejected, E_total_sum


# === Network Setup and Simulation (Balanced C_LOC) ===
def setup_iov_network(env, num_tasks, num_uavs):
    U_pos_init = np.array([[50, 50, 100], [150, 150, 100], [250, 250, 100],
                           [350, 350, 100], [450, 450, 100]])[:num_uavs]
    uav_nodes = [
        UAVNode(env, f"UAV-{i}", pos=U_pos_init[i],
                nu_i=np.random.uniform(700, 1000),
                p_u_i=np.random.uniform(0.5, 2.0))
        for i in range(num_uavs)
    ]
    V_pos_init = np.random.uniform(0, 500, (num_tasks, 3))
    V_pos_init[:, 2] = 0
    vehicle_nodes = [
        VehicleNode(env, f"Veh-{j}", pos=V_pos_init[j],
                    C_loc=np.random.uniform(1.5e9, 3.0e9),  
                    D_j=np.random.uniform(1e6, 3e6),
                    omega_j=np.random.uniform(1000, 5000),
                    T_max=np.random.uniform(20.0, 30.0),
                    p_tx=np.random.uniform(0.1, 1.0))
        for j in range(num_tasks)
    ]
    return vehicle_nodes, uav_nodes


def task_offloading_process(env, vehicles, uavs, interval, optimizer):
    while True:
        print(f"\n[{env.now:.2f}s] Running All-Offload Baseline...")
        optimizer.load_system_state(vehicles, uavs)
        min_total_time, offloaded_accepted, local_accepted, rejected, E_total_sum = optimizer.run_all_offload_baseline()
        
        print(f"[{env.now:.2f}s] Optimization Complete (Avg Delay: {min_total_time:.4f}s)")
        print(f"[{env.now:.2f}s] Strategy: {offloaded_accepted} offloaded, {local_accepted} local, {rejected} rejected.")
        print(f"[{env.now:.2f}s] Total Energy Consumption (E_total): {E_total_sum:.4e} J")
        yield env.timeout(interval)


def run_all_offload_sim():
    NUM_TASKS = 20
    NUM_UAVS = 5
    SIMULATION_TIME = 50
    INTERVAL = 5

    env = simpy.Environment()
    np.random.seed(42)
    vehicles, uavs = setup_iov_network(env, NUM_TASKS, NUM_UAVS)
    
    optimizer = UavIoVTaskOffloadingAllOffload(
        num_tasks=NUM_TASKS, num_uavs=NUM_UAVS, lambda_penalty=0.2
    )
    print(f"--- Starting All-Offload Baseline Simulation (UAV Energy Included) ---")
    print(f"Vehicle CPU Range (Same as GA): {1.5e9/1e9:.1f} to {3.0e9/1e9:.1f} GHz")
    
    env.process(task_offloading_process(env, vehicles, uavs, INTERVAL, optimizer))
    env.run(until=SIMULATION_TIME)
    print("\n--- Simulation Ended ---")


if __name__ == "__main__":
    run_all_offload_sim()