import os, sys; sys.path.insert(0, os.path.abspath(".."))
from quantum_objects import Source, SchedulingSource, Station, Pair
from protocol import TwoLinkProtocol
from requsim.world import World
from events import SourceEvent, GenericEvent, EntanglementSwappingEvent
import requsim.libs.matrix as mat
import numpy as np
from requsim.tools.noise_channels import x_noise_channel, y_noise_channel, z_noise_channel, w_noise_channel
from requsim.libs.aux_functions import apply_single_qubit_map
import matplotlib.pyplot as plt
from warnings import warn
import math

ETA_P = 0.15  # preparation efficiency
ETA_C = 0.15  # phton-fiber coupling efficiency * wavelength conversion
T_2 = 1  # dephasing time
C = 2 * 10**8 # speed of light in optical fiber
L_ATT = 22 * 10**3  # attenuation length
E_M_A = 0  # misalignment error

P_D_1 = 800e-9  # dark count probability per detector 1
P_D_2 = 50e-9  # dark count probability per detector 2

ETA_D_1 = 0.6  # detector efficiency for station 1
ETA_D_2 = 0.1  # detector efficiency for station 2

P_BSM = 1  # BSM success probability  ## WARNING: Currently not implemented
LAMBDA_BSM = 1  # BSM ideality parameter
F = 1.16  # error correction inefficiency

ETA_TOT_1 = ETA_P * ETA_C * ETA_D_1
ETA_TOT_2 = ETA_P * ETA_C * ETA_D_2

## convert dB to a distance having the same loss (for L_ATT=22)
def convert_dB_to_eff_km(dB):
    eff_length = dB*L_ATT/(10*np.log10(math.e))

    return eff_length

def construct_dephasing_noise_channel(dephasing_time):
    def lambda_dp(t):
        return (1 - np.exp(-t/dephasing_time)) / 2

    def dephasing_noise_channel(rho, t):
        return z_noise_channel(rho=rho, epsilon=lambda_dp(t))

    return dephasing_noise_channel




def imperfect_bsm_err_func(four_qubit_state):
    return LAMBDA_BSM * four_qubit_state + (1 - LAMBDA_BSM) * mat.reorder(mat.tensor(mat.ptrace(four_qubit_state, [1, 2]), mat.I(4) / 4), [0, 2, 3, 1])




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
        super(LuetkenhausProtocol, self).__init__(world)

    def check(self):
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
            ent_swap_event = EntanglementSwappingEvent(time=self.world.event_queue.current_time, pairs=[left_pairs[0], right_pairs[0]], error_func=imperfect_bsm_err_func)
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

    allowed_params = ["P_LINK", "T_P", "E_MA", "LAMBDA_BSM", "F_INIT", "T_DP", "ETA_TOT_1", "ETA_TOT_2", "P_D_1", "P_D_2"]
    for key in params:
        if key not in allowed_params:
            warn(f"params[{key}] is not a supported parameter and will be ignored.")
    # unpack the parameters
    P_LINK = params.get("P_LINK", 1.0)
    T_P = params.get("T_P", 0)  # preparation time
    E_MA = params.get("E_MA", 0)  # misalignment error
    LAMBDA_BSM = params.get("LAMBDA_BSM", 1)  # Bell state measurement ideality parameter
    F_INIT = params.get("F_INIT", 1.0)  # initial fidelity of created pairs
    ETA_TOT_1 = params.get("ETA_TOT_1", 0.0135) # transmittance first link
    ETA_TOT_2 = params.get("ETA_TOT_2", 0.00225) # transmittance second link
    P_D_1 = params.get("P_D_1", 800e-9) # dark count probability first detector
    P_D_2 = params.get("P_D_2", 50e-9) # dark count probability second detector
    try:
        T_DP = params["T_DP"]  # dephasing time
    except KeyError as e:
        raise Exception('params["T_DP"] is a mandatory argument').with_traceback(e.__traceback__)

    def luetkenhaus_time_distribution(source, ETA_TOT, P_D):
        comm_distance = np.max([np.abs(source.position - source.target_stations[0].position), np.abs(source.position - source.target_stations[1].position)])
        comm_time = 2 * comm_distance / C
        eta = ETA_TOT * np.exp(-comm_distance / L_ATT)
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
                state = apply_single_qubit_map(map_func=y_noise_channel, qubit_index=idx, rho=state, epsilon=E_M_A)
                eta = ETA_TOT * np.exp(-comm_distance / L_ATT)
                state = apply_single_qubit_map(map_func=w_noise_channel, qubit_index=idx, rho=state, alpha=alpha_of_eta(eta, P_D))
        return state

    def imperfect_bsm_err_func(four_qubit_state):
        return LAMBDA_BSM * four_qubit_state + (1-LAMBDA_BSM) * mat.reorder(mat.tensor(mat.ptrace(four_qubit_state, [1, 2]), mat.I(4) / 4), [0, 2, 3, 1])

    def alpha_of_eta(eta, P_D):
        return eta * (1 - P_D) / (1 - (1 - eta) * (1 - P_D)**2)

    world = World()
    station_A = Station(world, position=0, memory_noise=None)
    station_central = Station(world, position=L_1, memory_noise=construct_dephasing_noise_channel(dephasing_time=T_2))
    station_B = Station(world, position=L_1+L_2, memory_noise=None)
    source_A = SchedulingSource(world, position=L_1, target_stations=[station_A, station_central], time_distribution=lambda source: luetkenhaus_time_distribution(source, ETA_TOT_1, P_D_1), state_generation=lambda source: luetkenhaus_state_generation(source, ETA_TOT_1, P_D_1))
    source_B = SchedulingSource(world, position=L_1, target_stations=[station_central, station_B], time_distribution=lambda source: luetkenhaus_time_distribution(source, ETA_TOT_2, P_D_2), state_generation=lambda source: luetkenhaus_state_generation(source, ETA_TOT_2, P_D_2))
    protocol = LuetkenhausProtocol(world, mode=mode)
    protocol.setup()

    while len(protocol.time_list) < max_iter:
        protocol.check()
        world.event_queue.resolve_next_event()

    return protocol


if __name__ == "__main__":
    T_P_array = np.linspace(10e-6, 10e-2, 1)
    eff_length_1 = convert_dB_to_eff_km(24.4)
    eff_length_2 = convert_dB_to_eff_km(23.4)
    # evaluation in one go
    # for i in T_P_array:
    #     print(f"preparation time: {i}")
    #     p = run(L_1=eff_length_1, L_2=eff_length_2, params={"T_DP": 1, "T_P":i}, max_iter=1, mode="sim")
    #     print(p.data)

    # split into tasks
    task_index = int(os.environ["SLURM_ARRAY_TASK_ID"])
    prep_time = T_P_array[task_index]
    p = run(L_1=eff_length_1, L_2=eff_length_2, params={"T_DP": 1, "T_P":prep_time}, max_iter=10**4, mode="sim")
    evaluation = standard_bipartite_evaluation(p.data)
    fidelity = evaluation[0]
    key_rate = evaluation[3]

    output_file = "results/single_link_qnetq2.txt"
    # append the output to the file
    with open(output_file, "a") as file:
        file.write(f"Task {task_index}: T_P = {prep_time}, Fidelity = {fidelity}, Key rate = {key_rate}\n")


