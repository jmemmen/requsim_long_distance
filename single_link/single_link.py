import os, sys

# # check for correct path
sys.path.insert(0, os.path.abspath("."))
# directory = os.path.abspath(".")
# sys.path.insert(0, directory)
# print("Directory: ", directory)

import requsim
from quantum_objects import Source, SchedulingSource, Station, Pair, Qubit
from requsim.tools.protocol import TwoLinkProtocol
from requsim.world import World
from events import SourceEvent, GenericEvent, EntanglementSwappingEvent
from requsim.events import EventQueue, DiscardQubitEvent
import requsim.libs.matrix as mat
import numpy as np
from requsim.tools.noise_channels import x_noise_channel, y_noise_channel, z_noise_channel, w_noise_channel
from requsim.libs.aux_functions import apply_single_qubit_map
from requsim.tools.evaluation import standard_bipartite_evaluation
from warnings import warn
import math

max_iter = 10**1

ETA_P = 0.6  # preparation efficiency

T_2 = 1  # dephasing time
T_CUT = 0.2  # cutoff time

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

    return dephasing_noise_channel




class LuetkenhausProtocol(TwoLinkProtocol):
    """The Luetkenhaus Protocol.

    Parameters
    ----------
    world : World
        The world in which the protocol will be performed.
    mode : {"seq", "sim"}
        Selects sequential or simultaneous generation of links.

    Attributes
    ----------
    mode : str
        "seq" or "sim"

    """
    def __init__(self, world, mode="seq"):
        if mode != "seq" and mode != "sim":
            raise ValueError("LuetkenhausProtocol does not support mode %s. Use \"seq\" for sequential state generation, or \"sim\" for simultaneous state generation.")
        self.mode = mode
        super(LuetkenhausProtocol, self).__init__(world, communication_speed=C)

    def check(self, current_message=None):
        """Checks world state and schedules new events.

        Summary of the Protocol:
        Establish a left link and a right link.
        Then perform entanglement swapping.
        Record metrics about the long distance pair.
        Repeat.

        Returns
        -------
        None

        """
        # this protocol will only ever act if the event_queue is empty
        if self.world.event_queue:
            return
        try:
            pairs = self.world.world_objects["Pair"]
        except KeyError:
            pairs = []
        # if there are no pairs, begin protocol
        if not pairs:
            if self.mode == "seq":
                self.source_A.schedule_event()
            elif self.mode == "sim":
                self.source_A.schedule_event()
                self.source_B.schedule_event()
            return
        # in sequential mode, if there is only a pair on the left side, schedule creation of right pair
        left_pairs = self._get_left_pairs()
        num_left_pairs = len(left_pairs)
        right_pairs = self._get_right_pairs()
        num_right_pairs = len(right_pairs)
        if num_right_pairs == 0 and num_left_pairs == 1:
            if self.mode == "seq":
                self.source_B.schedule_event()
            return
        if num_right_pairs == 1 and num_left_pairs == 1:
            ent_swap_event = EntanglementSwappingEvent(time=self.world.event_queue.current_time, pairs=[left_pairs[0], right_pairs[0]])
            self.world.event_queue.add_event(ent_swap_event)
            return

        long_range_pairs = self._get_long_range_pairs()
        if long_range_pairs:
            long_range_pair = long_range_pairs[0]
            self._eval_pair(long_range_pair)
            # cleanup
            long_range_pair.qubits[0].destroy()
            long_range_pair.qubits[1].destroy()
            long_range_pair.destroy()
            self.check()
            return
        warn("LuetkenhausProtocol encountered unknown world state. May be trapped in an infinite loop?")

# L_1, L_2 effective lengths of first and second link respectively
def run(L_1, L_2, params, max_iter, mode="sim"):

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

    def luetkenhaus_time_distribution(source, ETA_TOT, P_D):
        comm_distance = np.max([np.abs(source.position - source.target_stations[0].position), np.abs(source.position - source.target_stations[1].position)])
        comm_time = 2 * comm_distance / C
        eta = ETA_TOT
        eta_effective = 1 - (1 - eta) * (1 - P_D)**2
        trial_time = T_P + comm_time
        random_num = np.random.geometric(eta_effective)
        return random_num * trial_time, random_num

    def luetkenhaus_state_generation(source, ETA_TOT, P_D):
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
    station_A = Station(world, position=0, memory_noise=None, memory_cutoff_time=T_CUT)
    station_central = Station(world, position=L_1, memory_noise=construct_dephasing_noise_channel(dephasing_time=T_2))
    station_B = Station(world, position=L_1+L_2, memory_noise=None, memory_cutoff_time=T_CUT)
    source_A = SchedulingSource(world, position=L_1, target_stations=[station_A, station_central], time_distribution=lambda source: luetkenhaus_time_distribution(source, ETA_TOT_1, P_D_1), state_generation=lambda source: luetkenhaus_state_generation(source, ETA_TOT_1, P_D_1))
    source_B = SchedulingSource(world, position=L_1, target_stations=[station_central, station_B], time_distribution=lambda source: luetkenhaus_time_distribution(source, ETA_TOT_2, P_D_2), state_generation=lambda source: luetkenhaus_state_generation(source, ETA_TOT_2, P_D_2))
    protocol = LuetkenhausProtocol(world, mode=mode)
    protocol.setup()

    # # filter to discard events from time to time, use when using cutoff times
    # filter_interval = int(1e4)

    # world.event_queue.add_recurring_filter(
    # condition=lambda event: isinstance(event, DiscardQubitEvent) and
    #                         event.type == "DiscardQubitEvent" and
    #                         not event.req_objects_exist(),
    # filter_interval=filter_interval,
    # )

    # main loop
    current_message = None
    while len(protocol.time_list) < max_iter:
        protocol.check(current_message)
        current_message = world.event_queue.resolve_next_event()

    return protocol


if __name__ == "__main__":

    # evaluation on local computer, simple loop
    # T_P_array = np.linspace(1e-6, 1e-6, 1)
    # loss_1_db = 24.4
    # loss_2_db = 23.4
    # eff_link_1 = convert_dB_to_efficiency(loss_1_db)
    # eff_link_2 = convert_dB_to_efficiency(loss_2_db)

    # for i in T_P_array:
    #     print(f"preparation time: {i}")
    #     p = run(L_1=90e3, L_2=91.2e3, params={"T_DP": 1, "T_P":i, "T_CUT":T_CUT, "ETA_TOT_1": eff_link_1*eta_1, "ETA_TOT_2": eff_link_2*eta_2}, max_iter=5, mode="sim")
    #     states = p.data["state"]
    #     evaluation = standard_bipartite_evaluation(p.data)
    #     print(evaluation)


    # evaluation on cluster, split into jobs
    task_index = int(os.environ["SLURM_ARRAY_TASK_ID"])

    T_P_array = np.linspace(1e-6, 1e-8, 5)
    loss_1_db = 24.4
    loss_2_db = 23.4
    eff_link_1 = convert_dB_to_efficiency(loss_1_db)
    eff_link_2 = convert_dB_to_efficiency(loss_2_db)

    output_file = "results/single_link_cut2.txt"
    with open(output_file, "a") as file:
        # write parameters to the output file if it's the first task
        if task_index == 0:
            file.write("Parameters:\n")
            file.write(f"Efficiency Link 1: efficiency link: {eff_link_1}, efficiency detector/source: {eta_1}\n")
            file.write(f"Efficiency Link 2: efficiency link: {eff_link_2}, efficiency detector/source: {eta_2}\n")
            file.write(f"T_DP = {T_2}, T_CUT = {T_CUT}, max_iter = {max_iter}, mode=sim\n")

        prep_time = T_P_array[task_index]

        p = run(L_1=90e3, L_2=91.2e3, params={"T_DP": 1, "T_P": prep_time, "T_CUT": T_CUT, "ETA_TOT_1": eff_link_1*eta_1, "ETA_TOT_2": eff_link_2*eta_2}, max_iter=max_iter, mode="sim")
        evaluation = standard_bipartite_evaluation(p.data)
        fidelity = evaluation[1]
        key_rate = evaluation[3]

        # append the output to the file
        file.write(f"Task {task_index}: T_P = {prep_time}, Fidelity = {fidelity}, Key rate per time = {key_rate}\n")












