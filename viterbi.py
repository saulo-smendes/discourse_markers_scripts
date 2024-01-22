import numpy as np
import pandas as pd
from math import e


def max_agreement(y):
    # Search for max agreement point
    """Find frame where most algorithms converge.
    Parameters
    ----------
    y : array
        A 2D array of shape NxF where N is the number of algorithms being tested and F
        is the number of time frames.

    Returns
    -------
    sd_min : float
        The minimum standard deviation.
    state_idx_min : int
        The index of minimum standard deviation.
    
    """
    n_obs, n_states = y.shape
    print(n_obs)
    print(n_states)
    
    sd_min = np.nan
    state_idx_min = 0
    for state_idx in range(n_states):
        obs = y[:,state_idx]
        if not np.isnan(obs).any():
            if np.isnan(sd_min):
                sd_min = np.std(obs)
            
            sd = np.std(obs)
            
            if sd < sd_min:
                sd_min = sd
                state_idx_min = state_idx
        
        # What to do in case a file do not have at leas one observation space without ?
        # else:
            
    return sd_min, int(state_idx_min)


def delta_f0(f1, f2, voiced_unvoiced_proba=0.5):
    """Find delta f0 or the probability of transition between two f0 points.
    Parameters
    ----------
    f1 : float
        The first f0 point.
    f2 : float
        The second f0 point.

    Returns
    -------
    proba : float
        The transition probability from f1 to f2.
    
    """
    # See Jan Bartosek (2011) Measures: cents of semitones

    if np.isnan(f1) and np.isnan(f2):
        proba = 1.0
    elif np.isnan(f1) ^ np.isnan(f2):
        # 0.007316274717241059 is the probability of transitioning from 75Hz to 800Hz
        # TODO: find a better way to estimate this probability
        proba = voiced_unvoiced_proba
    else:
        delta = np.abs(1200*np.log2(f1/f2))
        proba = 1/(e**(0.0012*delta))
    
    # TODO: Acrescentar os casos onde ou f1 ou f2 é np.nan para levar em conta os custos de voicing
    
    return proba


def delta_candidate(f1, f2, voiced_unvoiced_proba = 0.5):
    """Find delta f0 or the probability of transition between two f0 points.
    Parameters
    ----------
    f1 : float
        The first f0 point.
    f2 : float
        The second f0 point.

    Returns
    -------
    proba : float
        The emission probability from f1 to f2.
    
    """
    # See Jan Bartosek (2011) Measures: cents of semitones

    if np.isnan(f1) and np.isnan(f2):
        proba = 1.0
    elif np.isnan(f1) ^ np.isnan(f2):
        # 0.007316274717241059 is the probability of transitioning from 75Hz to 800Hz
        # TODO: find a better way to estimate this probability
        proba = voiced_unvoiced_proba
    else:
        delta = np.abs(np.log2(f1/f2))
        proba = 1/(e**(delta))
    
    # TODO: Acrescentar os casos onde ou f1 ou f2 é np.nan para levar em conta os custos de voicing
    
    return proba


def calculate_transition_probability(f1, f2, time, forgetting_threshold):
    # TODO: Poderíamos aplicar um forgetting threshold cíclico, que se aplica aos múltiplos do FT?
    delta = np.abs(1200*np.log2(f1/f2))
    proba = (1/(e**(0.0012*delta))) + (((1-1/(e**(0.0012*delta)))*time)/forgetting_threshold)
    return proba


def viterbi_path(data, transition_matrices, emission_matrices, reinforcement=1.2):
    """Find the Viterbi path for continuous data.

    Parameters
    ----------
    data : array
        A NxF array containing the observations. The vertical axis should represent
        different PDA algorithms or best F0 candidates whereas
        the horizontal axis should represent the time frames. N is the number of
        different PDAs.
    transition_matrices : list
        An list containing F NxN arrays where F is the number of 
        frames on the observational data and N is equal to the number of 
        algorithms (candidates) on the vertical axis of data.
    emission_matrices : list
        An list containing F Nx1 arrays where F is the number of 
        frames on the observational data and N is equal to the number of 
        algorithms (candidates) on the vertical axis of data.

    Returns
    -------
    winning_path : list
        A list containing the indexes of the winning candidates.
    winning_candidates : list
        A list containing the winning candidates.
    winning_data : list
        A list of lists containing for each frame the index of the frame m,
        the index of the most probable frame m - 1, and the maximum probability for frame m.
    """
    
    # TODO: Check the size of matrices
    
    
    # Create trellis and pointers
    trellis = np.empty(shape=data.shape)
    pointers = np.empty(shape=data.shape)

    # UNUSED: Find the point where candidates agree the maximum
    # sd_min, state_idx_min = max_agreement(data)
    # trellis[:,state_idx_min] = emission_matrices[state_idx_min]
    # pointers[:,state_idx_min] = state_idx_min


    # We begin from frame 0 by assining the first emission matrix to the 0th position
    # of the trellis
    trellis[:,0] = emission_matrices[0]
    pointers[:,0] = 0

    
    # Fill the trellis
    # Based on Jan Bartosek (2011)
    # m = current time frame
    # k = candidate on frame m - 1
    # l = candidate on frame m
    for m in range(1, data.shape[1]):
        # Get the transition and the emission matrix for the m-th frame
        a = transition_matrices[m-1]
        b = emission_matrices[m]
        # Convolve the transition matrix to find the most probable path
        for l in range(0,a.shape[0]):
            prob_max = 0
            pointer = 0
            for k in range(0,a.shape[0]):
                prob = trellis[k,m-1]*a[k,l]
                # Get the max transtion prob and the pointer k of transition matrix
                if prob >= prob_max:
                    # TODO: add some reinforcement factor so as to keep the same algo l == k
                    if k == l:
                        prob_max = prob*reinforcement
                    else:
                        prob_max = prob
                    pointer = k

            # Stores the max transition probability multiplied by the current emission probability
            
            
            trellis[l,m] = prob_max*b[l]
            pointers[l,m] = pointer

    pointers = pointers.astype(int)
    back_index = np.argmax(trellis[:,-1])
    winning_data = []
    winning_path = []

    # Find the Viterbi path
    for i in reversed(range(trellis.shape[1])):
        max_prob = trellis[back_index][i]
        winning_data.append([i, back_index, max_prob])
        winning_path.append(back_index)
        # Update backindex
        back_index = pointers[back_index][i]

    # Reverse the lists
    winning_path = winning_path[::-1]
    winning_data = winning_data[::-1]
    
    # Get the values of winning cadidates
    winning_candidates = []
    for i in range(len(winning_path)):
        winning_candidates.append(data[winning_path[i]][i])
        
    return winning_path, winning_candidates, winning_data


def calculate_transition_matrices(y, voiced_unvoiced_proba = 0.35):
    """Find frame where most algorithms converge.
    Parameters
    ----------
    y : array
        A 2D array of shape NxF where N is the number of algorithms being tested and F
        is the number of time frames.

    Returns
    -------
    transition_matrices : list
        A list containing NxN arrays with transition probabilities
    
    """
    # Return a transition cost array for a given sequence of observation spaces
    n_obs, n_states = y.shape
    transition_matrices = []
    for m in range(n_states-1):
        trans_matrix = np.empty(shape=(n_obs,n_obs))
        for l in range(n_obs):
            for k in range(n_obs):
                trans_matrix[k,l] = delta_f0(y[k,m], y[l,m+1], voiced_unvoiced_proba)

        transition_matrices.append(trans_matrix)

    return transition_matrices
        
    # How to calculate and add to transition probabilities the voice/unvoiced cost


def calculate_emission_matrix(data, hnr_vector, creaky_vector, noise_award_factor=2.5, creaky_award_factor=2.5, voiced_unvoiced_proba=0.5):
    """Find frame where most algorithms converge.
    Parameters
    ----------
    y : array
        A 2D array of shape NxF where N is the number of algorithms being tested and F
        is the number of time frames.

    Returns
    -------
    transition_matrices : list
        A list containing NxN arrays with transition probabilities
    
    """
    # TODO: Check if length of arrays are compatible
    # TODO: make more difficult to change pda

    emission_matrices = []
    for m in range(data.shape[1]):

        frame = data[:,m]
        hnr = hnr_vector[m]
        creaky = creaky_vector[m]

        if sum(np.isnan(frame)) > (frame.shape[0]/2):
            # Majority decision 
            median = np.nan
        else:
            median = np.nanmedian(frame)

        frame_prob = np.empty(shape=frame.shape)
        for i in range(frame.shape[0]):
            frame_prob[i] = delta_f0(median, frame[i], voiced_unvoiced_proba)

        # Bonus for 
        if hnr < 0:
            frame_prob[0] *= noise_award_factor
            frame_prob[2] *= noise_award_factor

        # Bonus for pefac
        if creaky == 1:
            frame_prob[0] *= creaky_award_factor
            frame_prob[2] *= creaky_award_factor
        
        emission_matrices.append(frame_prob)
    
    return emission_matrices