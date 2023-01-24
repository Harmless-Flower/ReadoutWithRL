import gym
from gym import spaces
import numpy as np
import matplotlib.pyplot as plt
import numba as nb
from numba import types
from numbalsoda import lsoda, dop853, address_as_void_pointer
from matplotlib import style

np.set_printoptions(threshold=10e5)

WIDTH = 40
T_MAX = 0.8
T_LENGTH = int(1000*T_MAX) #800
SEGMENTS = int(T_LENGTH/WIDTH)
T_EVAL = np.linspace(0., T_MAX, T_LENGTH + 1)

KAPPA = 5.35
CHI = 0.856
KERR = 0.01
P1 = KAPPA/2
P2 = CHI/2
P3 = KERR
MAX_AMP = 25
MAX_PHOTON = 7
MIN_PHOTON = 0.1

BLOB_WIDTH = 0.5
IDEAL_SIGMA = 1.96 # 2.35 for 98% fidelity readout, 1.96 for 95% fidelity readout

HIGH_AMP_WEIGHTING = 5 #Making this higher incentivizes reaching a maximum amplitude faster

U0 = np.array([0., 0.])

def rhs_g(t, u, du, a, b, c, arr_complex):
    du[0] = -a*u[0] - b*u[1] + c*((u[0])**2*u[1] + (u[1])**3) + arr_complex[int(t*T_LENGTH/T_MAX)]
    du[1] = -a*u[1] + b*u[0] - c*((u[1])**2*u[0] + (u[0])**3) + arr_complex[int(t*T_LENGTH/T_MAX + T_LENGTH + 1)]

def rhs_e(t, u, du, a, b, c, arr_complex):
    du[0] = -a*u[0] + b*u[1] + c*((u[0])**2*u[1] + (u[1])**3) + arr_complex[int(t*T_LENGTH/T_MAX)]
    du[1] = -a*u[1] - b*u[0] - c*((u[1])**2*u[0] + (u[0])**3) + arr_complex[int(t*T_LENGTH/T_MAX + T_LENGTH + 1)]

# 'P1' is the value of P1
# 'P2' is the value of P2
# 'arr_p' is the memory address of array arr_real
# 'len_arr' is the length of array arr_real

args_dtype = types.Record.make_c_struct([
    ('a', types.float32),
    ('b', types.float32),
    ('c', types.float32),
    ('arr_p', types.int64),
    ('len_arr', types.int64)])

# this function will create the numba function we pass to lsoda
def create_jit_rhs(rhs, args_dtype):
    jitted_rhs = nb.njit(rhs)
    @nb.cfunc(types.void(
        types.double,
        types.CPointer(types.double),
        types.CPointer(types.double),
        types.CPointer(args_dtype)))
    def wrapped(t, u, du, user_data_p):
        # unpack p and arr from user_data_p
        user_data = nb.carray(user_data_p, 1)
        a = user_data[0].a
        b = user_data[0].b
        c = user_data[0].c
        arr_complex = nb.carray(address_as_void_pointer(user_data[0].arr_p), (user_data[0].len_arr), dtype=np.float32)

        # then we call the jitted rhs function, passing in data
        jitted_rhs(t, u, du, a, b, c, arr_complex)
    return wrapped

# create the function to be called by lsoda
rhs_g_cfunc = create_jit_rhs(rhs_g, args_dtype)
rhs_e_cfunc = create_jit_rhs(rhs_e, args_dtype)

funcptr_g = rhs_g_cfunc.address
funcptr_e = rhs_e_cfunc.address

class PulseEnv(gym.Env):
    def __init__(self):
        super(PulseEnv, self).__init__()

        lower_bound = np.zeros(3*(SEGMENTS + 1) + 1) - 1
        upper_bound = np.zeros(3*(SEGMENTS + 1) + 1) + 1

        self.action_space = spaces.Box(lower_bound, upper_bound, dtype=np.float32)
        self.observation_space = spaces.Box(-10.e4*np.ones(4), 10.e4*np.ones(4), dtype=np.float32)
    
    def step(self, action):
        info = {}
        full_real = np.zeros(2*T_LENGTH + 1)
        full_imag = np.zeros(2*T_LENGTH + 1)
        action_real = action[:SEGMENTS + 1]
        action_imag = action[SEGMENTS + 1:2*(SEGMENTS + 1)]
        action_times = (action[2*(SEGMENTS + 1):-1] + 1)*WIDTH*0.5 + WIDTH
        counter = 0
        for i in range(SEGMENTS):
            full_real[counter: counter + int(action_times[i])] = action_real[i]
            full_imag[counter: counter + int(action_times[i])] = action_imag[i]
            counter += int(action_times[i])
        index = int((action[-1] + 1)*T_LENGTH*0.5)
        full_real[index:] = 0
        full_imag[index:] = 0

        complex_pulse = np.zeros(2*(T_LENGTH + 1))
        complex_pulse[:T_LENGTH + 1] = full_real[:T_LENGTH + 1]
        complex_pulse[T_LENGTH + 1:] = full_imag[:T_LENGTH + 1]

        arr_pulse = np.ascontiguousarray(MAX_AMP*complex_pulse, dtype=np.float32)

        args = np.array((P1, P2, P3, arr_pulse.ctypes.data, arr_pulse.shape[0]), dtype=args_dtype)

        usol_g, _ = lsoda(funcptr_g, U0, T_EVAL, data = args)
        usol_e, _ = lsoda(funcptr_e, U0, T_EVAL, data = args)

        distance_ratio = np.sqrt( np.abs(usol_g[:,1] - usol_e[:,1])**2 + np.abs(usol_g[:,0] - usol_e[:,0])**2 )/BLOB_WIDTH

        g_photon = usol_g[:,0]**2 + usol_g[:,1]**2
        e_photon = usol_e[:,0]**2 + usol_e[:,1]**2

        penalty = 0

        max_photon = 0
        final_photon = 0
        if max(g_photon) > max(e_photon):
            max_photon = max(g_photon)
        else:
            max_photon = max(e_photon)
        if g_photon[-1] > e_photon[-1]:
            final_photon = g_photon[-1]
        else:
            final_photon = e_photon[-1]

        if max_photon > MAX_PHOTON:
            penalty += 1000
        
        delta = 0
        if final_photon > MIN_PHOTON:
            delta = 1/KAPPA*np.log(final_photon/MIN_PHOTON)
        
        deviation_of_ideal = np.abs(2*IDEAL_SIGMA - max(distance_ratio))

        self.reward = -penalty - 15*deviation_of_ideal/(2*IDEAL_SIGMA) - 4*index/(T_LENGTH + 1)*T_MAX - 4*delta - HIGH_AMP_WEIGHTING*(1 - np.sum(np.abs(complex_pulse))/(2*index + 1))

        self.observation = np.array([final_photon, index*1., deviation_of_ideal, max_photon], dtype=np.float32)

        self.done = True

        return self.observation, self.reward, self.done, info
    
    def reset(self):
        self.done = False
        self.observation = np.array([0, 0, 0, 0], dtype=np.float32)
        return self.observation
    
    def grapher(self, action):
        info = {}
        full_real = np.zeros(2*T_LENGTH + 1)
        full_imag = np.zeros(2*T_LENGTH + 1)
        action_real = action[:SEGMENTS + 1]
        action_imag = action[SEGMENTS + 1:2*(SEGMENTS + 1)]
        action_times = (action[2*(SEGMENTS + 1):-1] + 1)*WIDTH*0.5 + WIDTH
        counter = 0
        for i in range(SEGMENTS):
            full_real[counter: counter + int(action_times[i])] = action_real[i]
            full_imag[counter: counter + int(action_times[i])] = action_imag[i]
            counter += int(action_times[i])
        index = int((action[-1] + 1)*T_LENGTH*0.5)
        full_real[index:] = 0
        full_imag[index:] = 0

        complex_pulse = np.zeros(2*(T_LENGTH + 1))
        complex_pulse[:T_LENGTH + 1] = full_real[:T_LENGTH + 1]
        complex_pulse[T_LENGTH + 1:] = full_imag[:T_LENGTH + 1]

        arr_pulse = np.ascontiguousarray(MAX_AMP*complex_pulse, dtype=np.float32)

        args = np.array((P1, P2, P3, arr_pulse.ctypes.data, arr_pulse.shape[0]), dtype=args_dtype)

        usol_g, _ = lsoda(funcptr_g, U0, T_EVAL, data = args)
        usol_e, _ = lsoda(funcptr_e, U0, T_EVAL, data = args)

        distance_ratio = np.sqrt( np.abs(usol_g[:,1] - usol_e[:,1])**2 + np.abs(usol_g[:,0] - usol_e[:,0])**2 )/BLOB_WIDTH

        g_photon = usol_g[:,0]**2 + usol_g[:,1]**2
        e_photon = usol_e[:,0]**2 + usol_e[:,1]**2

        penalty = 0

        max_photon = 0
        final_photon = 0
        if max(g_photon) > max(e_photon):
            max_photon = max(g_photon)
        else:
            max_photon = max(e_photon)
        if g_photon[-1] > e_photon[-1]:
            final_photon = g_photon[-1]
        else:
            final_photon = e_photon[-1]

        if max_photon > MAX_PHOTON:
            penalty += 1000
        
        delta = 0
        if final_photon > MIN_PHOTON:
            delta = 1/KAPPA*np.log(final_photon/MIN_PHOTON)
        
        deviation_of_ideal = np.abs(2*IDEAL_SIGMA - max(distance_ratio))

        self.reward = -penalty - 15*deviation_of_ideal/(2*IDEAL_SIGMA) - 4*index/(T_LENGTH + 1)*T_MAX - 4*delta - HIGH_AMP_WEIGHTING*(1 - np.sum(np.abs(complex_pulse))/(2*index + 1))

        self.observation = np.array([final_photon, index*1., deviation_of_ideal, max_photon], dtype=np.float32)

        max_index = np.where(distance_ratio == max(distance_ratio))
        graphing_index = max_index[0]

        print(f"Reward: {self.reward}")
        print(f"Maximum Sigma Number: {max(distance_ratio)}")
        print(f"Max Photon: {max_photon}")
        print(f"Final Photon: {final_photon}")
        print(f"Index Time: {index*T_MAX/(T_LENGTH + 1)}")
        print(f"Total Time till Base Photon: {index*T_MAX/(T_LENGTH + 1) + delta}")
        print(f"final photon ground: {g_photon[-1]}")
        print(f"final photon excited: {e_photon[-1]}")

        plt.rcParams.update({'font.size': 15})
        fig, ax = plt.subplots(3, 2)

        #Plot 0,0 is the real portion
        #Plot 0,1 is the imaginary portion
        #Plot 1,0 is the ground photon response
        #Plot 1,1 is the excited photon response
        #Plot 2,0 is the cavity state response with markers

        ax[0,0].plot(T_EVAL, arr_pulse[:T_LENGTH + 1], color="blue", label="real")
        ax[0,0].set_ylim([-MAX_AMP - 2.5, MAX_AMP + 2.5])
        ax[0,0].set_xlim([-0.02, T_EVAL[-1] + 0.02])
        ax[0,0].axvline(x=index/T_LENGTH*T_MAX, color='black', ls='--', lw=1.5, label='end of pulse')
        ax[0,0].axvline(x=index/T_LENGTH*T_MAX + delta, color='green', ls='--', lw=1.5, label='vacuum state')
        ax[0,0].set_xlabel("Time (us)")
        ax[0,0].set_ylabel("Amplitude (A.U.)")
        ax[0,0].legend()

        ax[0,1].plot(T_EVAL, arr_pulse[T_LENGTH + 1:], color="orange", label="imag")
        ax[0,1].set_ylim([-MAX_AMP - 2.5, MAX_AMP + 2.5])
        ax[0,1].set_xlim([-0.02, T_EVAL[-1] + 0.02])
        ax[0,0].axvline(x=index/T_LENGTH*T_MAX, color='black', ls='--', lw=1.5, label='end of pulse')
        ax[0,0].axvline(x=index/T_LENGTH*T_MAX + delta, color='green', ls='--', lw=1.5, label='vacuum state')
        ax[0,1].set_xlabel("Time (us)")
        ax[0,1].set_ylabel("Amplitude (A.U.)")
        ax[0,1].legend()

        ax[1,0].plot(T_EVAL, g_photon, color="darkviolet", label="ground")
        ax[1,0].set_xlim([-0.02, 0.02 + T_EVAL[-1]])
        ax[1,0].set_ylim([-0.01, 0.01 + max_photon])
        ax[1,0].axvline(x=index/T_LENGTH*T_MAX, color='black', ls='--', lw=1.5, label='end of pulse')
        ax[1,0].axvline(x=index/T_LENGTH*T_MAX + delta, color='green', ls='--', lw=1.5, label='vacuum state')
        ax[1,0].set_xlabel("Time (us)")
        ax[1,0].set_ylabel("Resonator Photon Population")
        ax[1,0].legend()

        ax[1,1].plot(T_EVAL, e_photon, color="deepskyblue", label="excited")
        ax[1,1].set_xlim([-0.02, 0.02 + T_EVAL[-1]])
        ax[1,1].set_ylim([-0.01, 0.01 + max_photon])
        ax[1,1].axvline(x=index/T_LENGTH*T_MAX, color='black', ls='--', lw=1.5, label='end of pulse')
        ax[1,1].axvline(x=index/T_LENGTH*T_MAX + delta, color='green', ls='--', lw=1.5, label='vacuum state')
        ax[1,1].set_xlabel("Time (us)")
        ax[1,1].set_ylabel("Resonator Photon Population")
        ax[1,1].legend()

        ax[2,0].plot(0, 0, marker="o", markersize=5, markeredgecolor="black", markerfacecolor="black")
        ax[2,0].plot(usol_g[-1, 1], usol_g[-1, 0], marker="o", markersize=5, markeredgecolor="red", markerfacecolor="red")
        ax[2,0].plot(usol_e[-1, 1], usol_e[-1, 0], marker="o", markersize=5, markeredgecolor="red", markerfacecolor="red")
        ax[2,0].plot(usol_g[graphing_index, 1], usol_g[graphing_index, 0], marker="o", markersize=5, markeredgecolor="green", markerfacecolor="green")
        ax[2,0].plot(usol_e[graphing_index, 1], usol_e[graphing_index, 0], marker="o", markersize=5, markeredgecolor="green", markerfacecolor="green")
        ax[2,0].plot(usol_g[:,1], usol_g[:,0], color="darkviolet", label='ground')
        ax[2,0].plot(usol_e[:,1], usol_e[:,0], color="deepskyblue", label='excited')
        ax[2,0].set_xlabel("imag")
        ax[2,0].set_ylabel("real")
        ax[2,0].legend()

        plt.style.use('classic')

        plt.legend(loc="upper right")
        plt.show()