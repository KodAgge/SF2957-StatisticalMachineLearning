#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# import blackjack as bjk
import numpy as np
import pandas as pd
from collections import defaultdict
import random


def learn_Q(env, n_sims, gamma = 1, omega = 0.77, epsilon = 0.05,
            init_val = 0.0, Q_init = None, episode_file = None,
            warmup=10000, decaying = False):
    """
    gamma: discount factor
    omega: polynomial learning rate parameter (Even-Dar & Mansour, 2003)
    epsilon: exploration probability parameter
    init_val: initiate Q-values to something other than 0?
    Q_init: pre-trained Q dict
    episode_file: save ave
    """

    c = 1

    # Can start with a previously trained Q dict
    if Q_init is None:
        Q = defaultdict(lambda: np.zeros(env.action_space.n) + init_val)
    else:
        Q = Q_init

    state_action_count = defaultdict(lambda: np.zeros(env.action_space.n,
                                                      dtype = int))
    avg_reward = 0.0
    # if we want to save the episode reward to a file,
    if episode_file:
        f = open(episode_file, "w+")
        f.write("episode,avg_reward\n")
    else:
        f = None
    for episode in range(1,n_sims + 1):
        done = False
        action_reward = 0.0
        episode_reward = 0.0
        state = env.reset()
        while not done:
            #Here you cans switch between decaying exploration and constant exploration
            if not decaying:
                explore = random.random() < epsilon 
            else:
                explore = random.random() < (epsilon / (1 + state_action_count[state].sum()))

            if state not in Q or explore:
                # Take a random action
                action = env.action_space.sample()
            else:
                # Take the best possible action
                action = np.argmax(Q[state])

            # Update the state-action count
            state_action_count[state][action] += 1

            # Draw the next state and reward of previous action
            state2, action_reward, done, info = env.step(action)
    #######################################################################
            # YOUR CODE HERE
            # Compute the learning rate and update the Q-function 

            alpha = c * state_action_count[state][action] ** (-omega)
            Q[state][action] = Q[state][action] + alpha * (action_reward + gamma * np.max(Q[state2])-Q[state][action])



    #######################################################################
            # Update state and episode reward
            state = state2
            episode_reward += action_reward

        if episode % (n_sims // 100) == 0:
            print('Mean avg reward, after {} episodes: {}'.format(
                episode, avg_reward))
            if f:
                # append to the file which we want to save to
                f.write("{},{}\n".format(episode, str(avg_reward)))

        # Game is over
        avg_reward += (episode_reward - avg_reward) / (episode + 1)
    return Q, avg_reward, state_action_count


def Q_policy(state, Q, env):
    if state in Q:
        return np.argmax(Q[state])
    return env.action_space.sample()

def filter_states(S):
    return {k: v for k, v in S.items() if type(k) == tuple and \
            k[0] > 11 and k[0] < 22 and k[1] < 11}

def fill_missing_sum_states(D, default_value = 0.0):
    S = D
    for player_sum in range(12, 22):
            for dealer_sum in range(1, 11):
                state0 = (player_sum, dealer_sum, False)
                state1 = (player_sum, dealer_sum, True)
                if state0 not in S:
                    S[state0] = default_value
                if state1 not in S:
                    S[state1] = default_value
    return S


def convert_to_sum_states(Q, env):
    """
    Function which convert the expanded state spce to a sum-based state space
    """
    S = dict()
    n = defaultdict(int)
    use_ace = lambda x: x[0] > 0 and \
                np.dot(env.deck_values, x) + 10 <= 21
    sum_p = lambda x: np.dot(env.deck_values, x) + \
                10 * use_ace(x)
    for state, action_values in Q.items():
        sum_state = (sum_p(state[0]), state[1],
                     use_ace(state[0]))
        if sum_state in S:
            n[sum_state] += 1
            S[sum_state] = (action_values + n[sum_state]  * S[sum_state])\
                    / (n[sum_state] + 1)
        else:
            S[sum_state] = action_values
    return S

def convert_to_value_function(Q):
    S = dict()
    for state, action_values in Q.items():
        S[state] = action_values.max()
    return S


def learn_MC(env, n_sims, gamma = 1, epsilon = 0.05,
            init_val = 0.0, Q_init = None, episode_file = None,
            warmup=10000, decaying = False):
    """
    gamma: discount factor
    epsilon: exploration probability parameter
    init_val: initiate Q-values to something other than 0?
    Q_init: pre-trained Q dict
    episode_file: save ave
    """

    # Can start with a previously trained Q dict
    if Q_init is None:
        Q = defaultdict(lambda: np.zeros(env.action_space.n) + init_val)
    else:
        Q = Q_init

    state_action_count = defaultdict(lambda: np.zeros(env.action_space.n,
                                                      dtype = int))
    avg_reward = 0.0
    # if we want to save the episode reward to a file,
    if episode_file:
        f = open(episode_file, "w+")
        f.write("episode,avg_reward\n")
    else:
        f = None
    for episode in range(1,n_sims + 1):
        done = False
        action_reward = 0.0
        episode_reward = 0.0
        state = env.reset()
        episode_state_action_count = defaultdict(lambda: np.zeros(env.action_space.n,
                                                      dtype = int))

        while not done:

            #Here you cans switch between decaying exploration and constant exploration
            if not decaying:
                explore = random.random() < epsilon 
            else:
                explore = random.random() < (epsilon / (1 + state_action_count[state].sum()))

            if state not in Q or explore:
                # Take a random action
                action = env.action_space.sample()
            else:
                # Take the best possible action
                action = np.argmax(Q[state])

            # Update the state-action count
            state_action_count[state][action] += 1
            episode_state_action_count[state][action] += 1

            # Draw the next state and reward of previous action
            state2, action_reward, done, info = env.step(action)

            state = state2
            episode_reward += action_reward

        if episode % (n_sims // 100) == 0:
            print('Mean avg reward, after {} episodes: {}'.format(
                episode, avg_reward))
            if f:
                # append to the file which we want to save to
                f.write("{},{}\n".format(episode, str(avg_reward)))
        # Game (episode) is over
     #################################################################### 
        # YOUR CODE HERE
            # Update avg_reward
            # Update the Q-function for each visited state/action pair. 

        avg_reward = avg_reward + (episode_reward - avg_reward) / episode
        
        for state in episode_state_action_count:
            for action in episode_state_action_count[state]:
                if episode_state_action_count[state][action] == 1:
                    Q[state][action] += (episode_reward - Q[state][action]) / state_action_count[state][action]
        
    #################################################################### 
    return Q, avg_reward, state_action_count
