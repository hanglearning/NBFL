import pickle
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.lines as mlines
from matplotlib.lines import Line2D

import os

log_base_path = '/Users/chenhang/Documents/Working'
attack_type_map = {0: 'No Attack', 1: 'Poison', 2: 'Label Flipping', 3: 'Lazy', 4: 'Poison & Lazy'}

y_offset = 0
y_axis_labels = []

# for attack_type in [0, 4]:
#     for mal in [0, 3, 6, 9, 10]:
for attack_type in [0, 4]:
    for mal in [0, 3, 6, 9, 10]:
        for alpha in [1.0]:
            if (attack_type == 0 and mal != 0) or (attack_type != 0 and mal == 0):
                continue
            
            # across different seeds
            NBFL_log_paths = sorted([f'{log_base_path}/NBFL/logs/{folder}' for folder in os.listdir(f'{log_base_path}/NBFL/logs') if os.path.isdir(f'{log_base_path}/NBFL/logs/{folder}') and f"mal_{mal}" in folder and f"attack_{attack_type}" in folder and f"alpha_{alpha}" in folder], reverse=True)

            for lp in NBFL_log_paths:
                # Open and load the pickle file
                with open(f'{lp}/logger.pickle', 'rb') as file:
                    logger = pickle.load(file)['forking_event']
                    for comm_round, nums_fork in logger.items():
                        if nums_fork > 1:
                            plt.scatter(comm_round, y_offset, marker='o', color='red')
                            plt.text(comm_round, y_offset, str(nums_fork), ha='center', va='center', color='black')
                        else:
                            plt.scatter(comm_round, y_offset, marker='o', color='white')
                y_axis_labels.append(f'M{mal} - {attack_type_map[attack_type]}, sd: {lp.split('_')[lp.split('_').index('seed') + 1]}, α: {alpha}')
                y_offset += 1
        
legend_handles = [
    Line2D([0], [0], marker='o', color='w', markerfacecolor='red', markersize=10, label=f'Forking Event')
]

plt.legend(handles=legend_handles, loc='best', prop={'size': 10})

plt.xlabel('Communication Round')
plt.ylabel('Run Name')
plt.title(f'Forking Events')
plt.yticks(range(len(y_axis_labels)), y_axis_labels)

plt.tight_layout()

plt.savefig(f'{log_base_path}/NBFL/logs/forking_events.png', dpi=300)
plt.clf()
# plt.show()