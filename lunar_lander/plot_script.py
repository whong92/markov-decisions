import numpy as np
import matplotlib.pyplot as plt
import pickle

def smooth(data, k):
    num_episodes = data.shape[1]
    num_runs = data.shape[0]

    smoothed_data = np.zeros((num_runs, num_episodes))

    for i in range(num_episodes):
        if i < k:
            smoothed_data[:, i] = np.mean(data[:, :i+1], axis = 1)   
        else:
            smoothed_data[:, i] = np.mean(data[:, i-k:i+1], axis = 1)    
        

    return smoothed_data

# Function to plot result
def plot_result(plot_dicts):
    plt_agent_sweeps = []
    
    fig, ax = plt.subplots(figsize=(8,6))

    
    for plot_dict in plot_dicts:
        
        path = plot_dict['data_path']
        legend = plot_dict['data_legend']
        sum_reward_data = np.load(path)

        # smooth data
        smoothed_sum_reward = smooth(data = sum_reward_data, k = 100)
        
        mean_smoothed_sum_reward = np.mean(smoothed_sum_reward, axis = 0)

        plot_x_range = np.arange(0, mean_smoothed_sum_reward.shape[0])
        graph_current_agent_sum_reward, = ax.plot(plot_x_range, mean_smoothed_sum_reward[:], label=legend)
        plt_agent_sweeps.append(graph_current_agent_sum_reward)
    
    ax.legend(handles=plt_agent_sweeps, fontsize = 13)
    ax.set_title("Learning Curve", fontsize = 15)
    ax.set_xlabel('Episodes', fontsize = 14)
    ax.set_ylabel("Sum of\nreward\nduring\nepisode", rotation=0, labelpad=40, fontsize = 14)
    ax.set_ylim([-300, 300])

    plt.tight_layout()
    plt.show()     
