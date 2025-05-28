import pickle
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.lines as mlines
import os

# filepath: /Users/chenhang/Documents/Working/LBFL/plotting/plot_avg_stake.py
log_base_path = '/Users/chenhang/Documents/Working'
attack_type_map = {0: 'No Attack', 1: 'Poison Attack', 2: 'Label Flipping Attack', 3: 'Lazy Attack', 4: 'Poison & Lazy Attack'}

# for attack_type in [0, 4]:
#     for mal in [0, 3, 6, 10]:
#         for alpha in [1.0, 100.0]:
for attack_type in [4]:
    for mal in [3,6,9,10]:
        for alpha in [1.0, 100.0]:
            if (attack_type == 0 and mal != 0) or (attack_type != 0 and mal == 0):
                continue
            
            # across different seeds
            LBFL_log_paths = [f'{log_base_path}/LBFL/logs/{folder}' for folder in os.listdir(f'{log_base_path}/LBFL/logs') if os.path.isdir(f'{log_base_path}/LBFL/logs/{folder}') and f"mal_{mal}" in folder and f"attack_{attack_type}" in folder and f"alpha_{alpha}" in folder]

            LBFL_avg_stake_over_runs = []
            for lp in LBFL_log_paths:
                # Open and load the pickle file
                with open(f'{lp}/logger.pickle', 'rb') as file:
                    logger = pickle.load(file)
                    avg_stake_over_rounds = []
                    for comm_round, device_topos_book in logger['pos_book'].items():
                        stake_over_devices = []
                        for device_idx, pos_book in device_topos_book.items():
                            stake_over_devices.append(list(pos_book.values())) # op1: convert pos book over devices to list
                        stake_over_devices = list(zip(*stake_over_devices)) # op2: stack stake info over devices
                        avg_stake_over_rounds.append(np.mean(stake_over_devices, axis=1)) # op3: average over the device
                    avg_stake_over_rounds = list(zip(*avg_stake_over_rounds)) # op4: stack average stake of devices (by row) over rounds (by column)
                    LBFL_avg_stake_over_runs.append(avg_stake_over_rounds) 

            mean_lines = np.mean(LBFL_avg_stake_over_runs, axis=0) # op5: average over runs by same devices in the same round

            # differentiate legitimate and malicious devices
            # with open(f'{lp}/logger.pickle', 'rb') as file:
            #     logger = pickle.load(file)
            color = 'green'
            for i, ml in enumerate(mean_lines):
                if i + 1 + mal > 20:
                    color = 'red'
                    if attack_type == 3 or (attack_type == 4 and (i + 1) % 2 == 0):
                        color = 'magenta'
                plt.plot(range(1, len(ml) + 1), ml, color = color)
                plt.annotate(f'{i + 1}', xy=(len(ml), ml[-1]), xytext=(5, 0), textcoords='offset points', color=color, fontsize=8)

            green_line = mlines.Line2D([], [], color='green', label="legitimate")
            red_line = mlines.Line2D([], [], color='red', label="poison attack")
            magenta_line = mlines.Line2D([], [], color='magenta', label="laziness attack")

            if attack_type == 0:
                plt.legend(handles=[green_line], loc='best', prop={'size': 8})
            elif attack_type == 1:
                plt.legend(handles=[green_line, red_line], loc='best', prop={'size': 8})
            elif attack_type == 3:
                plt.legend(handles=[green_line, magenta_line], loc='best', prop={'size': 8})
            elif attack_type == 4:
                plt.legend(handles=[green_line, red_line, magenta_line], loc='best', prop={'size': 8})

            plt.xlabel('Communication Round', fontsize=10)
            plt.ylabel('Stake', fontsize=10)
            plt.title(f'Stake Curves - {mal} Atkers - {attack_type_map[attack_type]}, α: {alpha}', fontsize=12)

            # Add grid based on x-axis
            plt.grid(axis='x')

            # Set x-axis ticks to display every number
            plt.xticks(range(1, len(mean_lines[0]) + 1))

            plt.savefig(f'{log_base_path}/LBFL/logs/avg_stake_mal_{mal}_attack_{attack_type}_alpha_{alpha}.png', dpi=300)
            plt.clf()
            # plt.show()