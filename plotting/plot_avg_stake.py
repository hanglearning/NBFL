import pickle
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.lines as mlines

import os

log_base_path = '/Users/chenhang/Documents/Working'
attack_type_map = {0: 'No Attack', 1: 'Poison Attack', 2: 'Label Flipping Attack', 3: 'Lazy Attack'}

for attack_type in [0, 1, 2, 3]:
    for mal in [0, 3, 6, 8]:
        if (attack_type == 0 and mal != 0) or (attack_type != 0 and mal == 0):
            continue

        LBFL_log_paths = [f'{log_base_path}/LBFL/logs/{folder}' for folder in os.listdir(f'{log_base_path}/LBFL/logs') if os.path.isdir(f'{log_base_path}/LBFL/logs/{folder}') and f"mal_{mal}" in folder and f"attack_{attack_type}" in folder]

        LBFL_avg_stake_over_runs = []
        for lp in LBFL_log_paths:
            # Open and load the pickle file
            with open(f'{lp}/logger.pickle', 'rb') as file:
                logger = pickle.load(file)
                avg_stake_over_rounds = []
                for comm_round, device_to_pos_book in logger['pos_book'].items():
                    stake_over_devices = []
                    for device_idx, pos_book in device_to_pos_book.items():
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
            if i + mal >= 20:
                color = 'red'
            plt.plot(range(1, len(ml) + 1), ml, color = color)



        green_line = mlines.Line2D([], [], color='green', label="legitimate")
        red_line = mlines.Line2D([], [], color='red', label="malicious")

        plt.legend(handles=[green_line,red_line], loc='best', prop={'size': 10})

        plt.xlabel('Communication Round')
        plt.ylabel('Stake')
        plt.title(f'Stake Curves - {mal} Attackers - {attack_type_map[attack_type]}')
        
        plt.savefig(f'{log_base_path}/LBFL/logs/avg_stake_mal_{mal}_attack_{attack_type}.png', dpi=300)
        plt.clf()
        # plt.show()