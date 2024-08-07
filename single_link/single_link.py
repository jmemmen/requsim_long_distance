import os, sys

# # check for correct path
sys.path.insert(0, os.path.abspath("."))
# directory = os.path.abspath(".")
# sys.path.insert(0, directory)
# print("Directory: ", directory)

import requsim
from requsim.quantum_objects import Source, SchedulingSource, Station, Pair, Qubit
from requsim.tools.protocol import TwoLinkProtocol
from requsim.world import World
from requsim.events import SourceEvent, GenericEvent, EntanglementSwappingEvent
from requsim.events import EventQueue, DiscardQubitEvent
import requsim.libs.matrix as mat
import numpy as np
from requsim.tools.noise_channels import x_noise_channel, y_noise_channel, z_noise_channel, w_noise_channel
from requsim.libs.aux_functions import apply_single_qubit_map
from requsim.tools.evaluation import standard_bipartite_evaluation
from warnings import warn
from requsim.noise import NoiseChannel
import math

C = 2 * 10**8 # speed of light in optical fibre
L_ATT = 22 * 10**3  # attenuation length


# convert dB to an efficiency
def convert_dB_to_efficiency(dB):
    efficiency = 10**(-dB/10)

    return efficiency


def construct_dephasing_noise_channel(dephasing_time):
    def lambda_dp(t):
        return (1 - np.exp(-t/dephasing_time)) / 2

    def dephasing_noise_channel(rho, t):
        return z_noise_channel(rho=rho, epsilon=lambda_dp(t))

    return NoiseChannel(n_qubits=1, channel_function=dephasing_noise_channel)


class SimpleProtocol(TwoLinkProtocol):

    def check(self, message=None):

        left_pairs = self._get_left_pairs()
        num_left_pairs = len(left_pairs)
        num_left_pairs_scheduled = len(self._left_pairs_scheduled())
        right_pairs = self._get_right_pairs()
        num_right_pairs = len(right_pairs)
        num_right_pairs_scheduled = len(self._right_pairs_scheduled())
        long_distance_pairs = self._get_long_range_pairs()

        # STEP 1: For each link, if there are no pairs established and
        #         no pairs scheduled: Schedule a pair.
        if num_left_pairs + num_left_pairs_scheduled == 0:
            self.source_A.schedule_event()
        if num_right_pairs + num_right_pairs_scheduled == 0:
            self.source_B.schedule_event()

        # STEP 2: If both links are present, do entanglement swapping.
        if num_left_pairs == 1 and num_right_pairs == 1:
            left_pair = left_pairs[0]
            right_pair = right_pairs[0]
            ent_swap_event = EntanglementSwappingEvent(
                time=self.world.event_queue.current_time,
                pairs=[left_pair, right_pair],
                station=self.station_central,
            )
            self.world.event_queue.add_event(ent_swap_event)

        # STEP 3: If a long range pair is present, save its data and delete
        #         the associated objects.
        if long_distance_pairs:
            for pair in long_distance_pairs:
                self._eval_pair(pair)
                for qubit in pair.qubits:
                    qubit.destroy()
                pair.destroy()


# L_1, L_2 effective lengths of first and second link respectively
def run(L_1, L_2, params, max_iter, detector_1, detector_2):

    detector_properties = {
        "SNSDP": {"DE": 0.6, "DC": 800e-9},
        "ID230": {"DE": 0.1, "DC": 50e-9},
        "QUBE": {"DE": 0.6, "DC": 200e-9},
    }

    if detector_1 not in detector_properties or detector_2 not in detector_properties:
        raise ValueError("Invalid detector type provided.")

    props_1 = detector_properties[detector_1]
    props_2 = detector_properties[detector_2]

    allowed_params = ["P_LINK", "T_P", "T_CUT", "F_INIT", "T_DP", "EFF_LINK_1", "EFF_LINK_2"]
    for key in params:
        if key not in allowed_params:
            warn(f"params[{key}] is not a supported parameter and will be ignored.")

    # Unpack the parameters
    P_LINK = params.get("P_LINK", 1.0)
    T_P = params.get("T_P", 1e-6)  # preparation time
    T_CUT = params.get("T_CUT", None)  # cutoff time
    F_INIT = params.get("F_INIT", 1.0)  # initial fidelity of created pairs
    EFF_LINK_1 = params.get("EFF_LINK_1", 1.0)
    EFF_LINK_2 = params.get("EFF_LINK_2", 1.0)
    ETA_1 = props_1["DE"] # detector efficiency 1
    ETA_2 = props_2["DE"] # detector efficiency 2
    P_D_1 = props_1["DC"] # dark count probability 1 (converted into probability with detection window 1e-9)
    P_D_2 = props_2["DC"] # dark count probability 2 (converted into probability with detection window 1e-9)
    try:
        T_DP = params["T_DP"]  # dephasing time
    except KeyError as e:
        raise Exception('params["T_DP"] is a mandatory argument').with_traceback(e.__traceback__)

    # combined efficiencies
    ETA_TOT_1 = EFF_LINK_1*ETA_1
    ETA_TOT_2 = EFF_LINK_2*ETA_2

    def time_distribution(source, ETA_TOT, P_D):
        comm_distance = np.max([np.abs(source.position - source.target_stations[0].position), np.abs(source.position - source.target_stations[1].position)])
        comm_time = 2 * comm_distance / C
        eta = ETA_TOT
        eta_effective = 1 - (1 - eta) * (1 - P_D)**2
        trial_time = T_P + comm_time
        random_num = np.random.geometric(eta_effective)
        return random_num * trial_time

    def state_generation(source, ETA_TOT, P_D):
        state = np.dot(mat.phiplus, mat.H(mat.phiplus))
        comm_distance = np.max([np.abs(source.position - source.target_stations[0].position), np.abs(source.position - source.target_stations[1].position)])
        storage_time = 2 * comm_distance / C
        for idx, station in enumerate(source.target_stations):
            if station.memory_noise is not None:
                state = apply_single_qubit_map(map_func=station.memory_noise, qubit_index=idx, rho=state, t=storage_time)
            if station.memory_noise is None:
                state = apply_single_qubit_map(map_func=y_noise_channel, qubit_index=idx, rho=state, epsilon=0)
                eta = ETA_TOT
                state = apply_single_qubit_map(map_func=w_noise_channel, qubit_index=idx, rho=state, alpha=alpha_of_eta(eta, P_D))
        return state


    def alpha_of_eta(eta, P_D):
        return eta * (1 - P_D) / (1 - (1 - eta) * (1 - P_D)**2)

    world = World()
    station_A = Station(world, position=0, memory_noise=None, dark_count_probability=P_D_1)
    station_central = Station(world, position=L_1, memory_noise=construct_dephasing_noise_channel(dephasing_time=T_DP), memory_cutoff_time=T_CUT)
    station_B = Station(world, position=L_1+L_2, memory_noise=None, dark_count_probability=P_D_2)
    source_A = SchedulingSource(world, position=L_1, target_stations=[station_A, station_central], time_distribution=lambda source: time_distribution(source, ETA_TOT_1, P_D_1), state_generation=lambda source: state_generation(source, ETA_TOT_1, P_D_1))
    source_B = SchedulingSource(world, position=L_1, target_stations=[station_central, station_B], time_distribution=lambda source: time_distribution(source, ETA_TOT_2, P_D_2), state_generation=lambda source: state_generation(source, ETA_TOT_2, P_D_2))
    protocol = SimpleProtocol(world, communication_speed=C)
    protocol.setup()

    # filter to discard events from time to time, use when using cutoff times
    # filter_interval = int(1e4)

    # world.event_queue.add_recurring_filter(
    # condition=lambda event: event.type == "DiscardQubitEvent" and
    #                         not event.req_objects_exist(),
    # filter_interval=filter_interval,
    # )

    # main loop
    current_message = None
    while len(protocol.time_list) < max_iter:
        protocol.check(current_message)
        current_message = world.event_queue.resolve_next_event()

    return protocol


from filelock import FileLock

if __name__ == "__main__":

    # evaluation on local computer, simple loop
    # T_P_array = np.linspace(1e-6, 1e-6, 1)
    # loss_1_db = 24.4
    # loss_2_db = 23.4
    # eff_link_1 = convert_dB_to_efficiency(loss_1_db)
    # eff_link_2 = convert_dB_to_efficiency(loss_2_db)

    # for i in T_P_array:
    #     print(f"preparation time: {i}")
    #     p = run(L_1=90e3, L_2=91.2e3, params={"T_DP": 1, "T_P":i, "T_CUT": 0.1, "ETA_TOT_1": eff_link_1*eta_1, "ETA_TOT_2": eff_link_2*eta_2}, max_iter=5)
    #     evaluation = standard_bipartite_evaluation(p.data)
    #     print(evaluation)


    # evaluation on cluster, split into jobs, 1D plot
    task_index = int(os.environ["SLURM_ARRAY_TASK_ID"])
    max_iter = 10**4
    T_CUT = 0.2 # cutoff time
    prep_time = 1e-6
    T_2_array = np.logspace(-3, 2, 80)
    link_1_db_m = [12, 46e3]
    link_2_db_m = [23, 89e3]
    station_1 = "SNSDP"
    station_2 = "ID230"
    eff_link_1 = convert_dB_to_efficiency(link_1_db_m[0])
    eff_link_2 = convert_dB_to_efficiency(link_2_db_m[0])

    output_file = "results/Erf_Wal_Eit_T2_cut0.2.txt"
    with open(output_file, "a") as file:
        # write parameters to the output file if it's the first task

        if task_index == 0:
            file.write("Parameters:\n")
            file.write(f"Station 1 type: {station_1}, Station 2 type: {station_2}\n")
            file.write(f"Link 1 db: {link_1_db_m[0]}, Link 1 m: {link_1_db_m[1]}\n")
            file.write(f"Link 2 db: {link_2_db_m[0]}, Link 2 m: {link_2_db_m[1]}\n")
            file.write(f"T_CUT = {T_CUT}, T_P = {prep_time}, max_iter = {max_iter}, mode=sim\n")

        deph_time = T_2_array[task_index]

        p = run(L_1=link_1_db_m[1], L_2=link_2_db_m[1], params={"T_DP": deph_time, "T_P": prep_time, "T_CUT": T_CUT, "EFF_LINK_1": eff_link_1, "EFF_LINK_2": eff_link_2}, max_iter=max_iter, detector_1=station_1, detector_2=station_2)
        evaluation = standard_bipartite_evaluation(p.data)
        fidelity = evaluation[1]
        key_rate = evaluation[3]

        # append the output to the file
        file.write(f"Task {task_index}: T_2 = {deph_time}, Fidelity = {fidelity}, Key rate per time = {key_rate}\n")


    # # evaluation on cluster, split into jobs, 2D plot
    # task_index = int(os.environ["SLURM_ARRAY_TASK_ID"])

    # T_P_array = np.logspace(-6, -9, 10)
    # T_2_array = np.logspace(-2, 1, 10)
    # combinations = [(T_P_val, T_2_val) for T_P_val in T_P_array for T_2_val in T_2_array]
    # T_P_val, T_2_val = combinations[task_index]



    # loss_1_db = 24.4
    # loss_2_db = 23.4
    # eff_link_1 = convert_dB_to_efficiency(loss_1_db)
    # eff_link_2 = convert_dB_to_efficiency(loss_2_db)

    # output_file = "results/2D_nocut.txt"
    # lock_file = output_file + ".lock"


    # with FileLock(lock_file):
    #     with open(output_file, "a") as file:
    #         # write parameters to the output file if it's the first task
    #         if task_index == 0:
    #             file.write("Parameters:\n")
    #             file.write(f"Efficiency Link 1: efficiency link: {eff_link_1}, efficiency detector/source: {eta_1}\n")
    #             file.write(f"Efficiency Link 2: efficiency link: {eff_link_2}, efficiency detector/source: {eta_2}\n")
    #             file.write(f"T_CUT = {T_CUT}, max_iter = {max_iter}, mode=sim\n")

    #         p = run(L_1=90e3, L_2=91.2e3, params={"T_DP": T_2_val, "T_P": T_P_val, "T_CUT": T_CUT, "ETA_TOT_1": eff_link_1*eta_1, "ETA_TOT_2": eff_link_2*eta_2}, max_iter=max_iter)
    #         evaluation = standard_bipartite_evaluation(p.data)
    #         fidelity = evaluation[1]
    #         key_rate = evaluation[3]

    #         # append the output to the file
    #         file.write(f"Task {task_index}: T_P = {T_P_val}, T_2 = {T_2_val}, Fidelity = {fidelity}, Key rate per time = {key_rate}\n")















