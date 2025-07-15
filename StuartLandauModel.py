import os
import numpy as np
from numba import njit, prange, cuda, f8
import scipy.signal as sig

#Stuart Landau Model
@njit(nogil=True)
def simulate_SL(t_len, W_mat, sim_time, store_time, dt, dt_save, a_bif, speed):

    
    
    W = W_mat.copy()
    
    ### Defines attributes for simulation
    # Number of neuronal populations
    n = t_len.shape[0]

    # Parameters of Stuart-Landau Model
    w = 2*np.pi*40 # frequency of oscillation
    a = a_bif # bifurcation parameter
    b = 0.001 # noise

    # Connectivity matrix and delays
    if speed > 0:
        delays_raw = t_len / speed
        delays     = np.floor(delays_raw / dt)
        max_delay  = int(np.amax(delays))
        delays     = delays.astype(np.int64)
    
    # If speed is negative, interprets it as mean delay = 0
    else:
        delays    = np.zeros((t_len.shape))
        max_delay = 0
        delays    = delays.astype(np.int64)
    
    # Defines vectors to store activity. Storage starts in the points defined in store and store_time timesteps of activity are recorded
    # This is done to avoid huge arrays with mostly irrelevant data.
    store_time  = min(store_time, sim_time)
    output      = np.zeros((n, int(np.floor(store_time / dt_save))), dtype = np.complex_)
    store_count = 0

    # Fills past activity vector with random values around the target firing rate
    # This is a vector that keeps the state from the last max_delay timesteps.
    state = b*np.random.rand(n, max_delay+1) + b*1J*np.random.rand(n, max_delay+1)

    # Runs simulations
    for st in range(int(np.floor(sim_time / dt))): 

        # Gets current state (z) for all nodes
        curr_z    = np.reshape(state[:, -1].copy(), (n, 1))


        # Computes delayed input from other nodes        
        state_z   = state.copy()
        del_state = np.zeros((n, n), dtype = np.complex_)

        for i in range(n):
            for j in range(n):
                if W[i, j]>0:
                    del_state[i, j] = (state_z[j, -(delays[i, j]+1)] - state_z[i, -1])

        # Multiplies connectivity matrix with the delayed state
        input_delayed = np.reshape(np.sum(np.multiply(W, del_state), axis = 1), (n, 1))

        # Calculates state changes  
        dz = curr_z * (a + (w * 1J) - np.abs(curr_z**2)) + \
             input_delayed + b*(np.random.randn(n, 1) + 1J*np.random.randn(n, 1))

        
        # Updates firing rate and state variable arrays using Euler method            
        new_z         = curr_z + (dt*1e-3) * dz # dt has to be in seconds
        state[:, :-1] = state[:, 1:]
        state[:, -1]  = np.reshape(new_z, (n))



        # Stores data
        if st >= ((sim_time - store_time)/dt):
            if int((st % (dt_save/dt))) == 0:
                output[:, store_count]  = np.reshape(new_z, (n))
                store_count += 1
    
    return output