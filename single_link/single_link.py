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

max_iter = 10**1

ETA_P = 0.6  # preparation efficiency

T_2 = 1  # dephasing time
T_CUT = 1  # cutoff time

C = 2 * 10**8 # speed of light in optical fibre
L_ATT = 22 * 10**3  # attenuation length


P_D_1 = 800e-9  # dark count probability per detector 1
P_D_2 = 50e-9  # dark count probability per detector 2

ETA_D_1 = 0.6  # detector efficiency for station 1
ETA_D_2 = 0.1  # detector efficiency for station 2


eta_1 = ETA_P * ETA_D_1
eta_2 = ETA_P * ETA_D_2




## debugging check for AttributeError: 'DiscardQubitEvent' object has no attribute 'req_objects_exist'
# world = World()
# station_A = Station(world, position=0, memory_noise=None)
# station_B = Station(world, position=1, memory_noise=None)
# qubit = Qubit(world, station_A)
# event1 = DiscardQubitEvent(1,qubit)
# #print(dir(event1))
# print(hasattr(event1, 'req_objects_exist'))



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
def run(L_1, L_2, params, max_iter):

    allowed_params = ["P_LINK", "T_P", "T_CUT", "F_INIT", "T_DP", "ETA_TOT_1", "ETA_TOT_2", "P_D_1", "P_D_2"]
    for key in params:
        if key not in allowed_params:
            warn(f"params[{key}] is not a supported parameter and will be ignored.")
    # unpack the parameters
    P_LINK = params.get("P_LINK", 1.0)
    T_P = params.get("T_P", 1e-6)  # preparation time
    T_CUT = params.get("T_CUT", None) # cutoff time
    F_INIT = params.get("F_INIT", 1.0)  # initial fidelity of created pairs
    ETA_TOT_1 = params.get("ETA_TOT_1", 0.36) # efficiency first link
    ETA_TOT_2 = params.get("ETA_TOT_2", 0.06) # efficiency second link
    P_D_1 = params.get("P_D_1", 800e-9) # dark count probability first detector
    P_D_2 = params.get("P_D_2", 50e-9) # dark count probability second detector
    try:
        T_DP = params["T_DP"]  # dephasing time
    except KeyError as e:
        raise Exception('params["T_DP"] is a mandatory argument').with_traceback(e.__traceback__)

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
    station_central = Station(world, position=L_1, memory_noise=construct_dephasing_noise_channel(dephasing_time=T_2), memory_cutoff_time=T_CUT)
    station_B = Station(world, position=L_1+L_2, memory_noise=None, dark_count_probability=P_D_2)
    source_A = SchedulingSource(world, position=L_1, target_stations=[station_A, station_central], time_distribution=lambda source: time_distribution(source, ETA_TOT_1, P_D_1), state_generation=lambda source: state_generation(source, ETA_TOT_1, P_D_1))
    source_B = SchedulingSource(world, position=L_1, target_stations=[station_central, station_B], time_distribution=lambda source: time_distribution(source, ETA_TOT_2, P_D_2), state_generation=lambda source: state_generation(source, ETA_TOT_2, P_D_2))
    protocol = SimpleProtocol(world, communication_speed=C)
    protocol.setup()

    # filter to discard events from time to time, use when using cutoff times
    filter_interval = int(1e4)

    # world.event_queue.add_recurring_filter(
    # condition=lambda event: isinstance(event, DiscardQubitEvent) and
    #                         event.type == "DiscardQubitEvent" and
    #                         not event.req_objects_exist(),
    # filter_interval=filter_interval,
    # )

    world.event_queue.add_recurring_filter(
    condition=lambda event: event.type == "DiscardQubitEvent" and
                            not event.req_objects_exist(),
    filter_interval=filter_interval,
    )

    # main loop
    current_message = None
    while len(protocol.time_list) < max_iter:
        protocol.check(current_message)
        current_message = world.event_queue.resolve_next_event()

    return protocol


if __name__ == "__main__":

    # evaluation on local computer, simple loop
    T_P_array = np.linspace(1e-6, 1e-6, 1)
    loss_1_db = 24.4
    loss_2_db = 23.4
    eff_link_1 = convert_dB_to_efficiency(loss_1_db)
    eff_link_2 = convert_dB_to_efficiency(loss_2_db)

    for i in T_P_array:
        print(f"preparation time: {i}")
        p = run(L_1=90e3, L_2=91.2e3, params={"T_DP": 1, "T_P":i, "T_CUT": 0.1, "ETA_TOT_1": eff_link_1*eta_1, "ETA_TOT_2": eff_link_2*eta_2}, max_iter=5)
        evaluation = standard_bipartite_evaluation(p.data)
        print(evaluation)


    # # evaluation on cluster, split into jobs
    # task_index = int(os.environ["SLURM_ARRAY_TASK_ID"])
    #
    # T_P_array = np.linspace(1e-6, 1e-8, 5)
    # loss_1_db = 24.4
    # loss_2_db = 23.4
    # eff_link_1 = convert_dB_to_efficiency(loss_1_db)
    # eff_link_2 = convert_dB_to_efficiency(loss_2_db)
    #
    # output_file = "results/single_link_central.txt"
    # with open(output_file, "a") as file:
    #     # write parameters to the output file if it's the first task
    #     if task_index == 0:
    #         file.write("Parameters:\n")
    #         file.write(f"Efficiency Link 1: efficiency link: {eff_link_1}, efficiency detector/source: {eta_1}\n")
    #         file.write(f"Efficiency Link 2: efficiency link: {eff_link_2}, efficiency detector/source: {eta_2}\n")
    #         file.write(f"T_DP = {T_2}, T_CUT = {T_CUT}, max_iter = {max_iter}, mode=sim\n")
    #
    #     prep_time = T_P_array[task_index]
    #
    #     p = run(L_1=90e3, L_2=91.2e3, params={"T_DP": 1, "T_P": prep_time, "T_CUT": T_CUT, "ETA_TOT_1": eff_link_1*eta_1, "ETA_TOT_2": eff_link_2*eta_2}, max_iter=max_iter, mode="sim")
    #     evaluation = standard_bipartite_evaluation(p.data)
    #     fidelity = evaluation[1]
    #     key_rate = evaluation[3]
    #
    #     # append the output to the file
    #     file.write(f"Task {task_index}: T_P = {prep_time}, Fidelity = {fidelity}, Key rate per time = {key_rate}\n")












