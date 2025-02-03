import os
import pickle
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

log_base_path = '/Users/chenhang/Documents/Working'
attack_type_map = {0: 'No Attack', 1: 'Poison Attack', 2: 'Label Flipping Attack', 3: 'Lazy Attack'}

y_offset = 0
y_axis_labels = []

for attack_type in [0, 1, 3]:
    for mal in [0, 3, 6, 10]:
        if (attack_type == 0 and mal != 0) or (attack_type != 0 and mal == 0):
            continue
        
        # across different seeds
        LBFL_log_paths = [f'{log_base_path}/LBFL/logs/{folder}' for folder in os.listdir(f'{log_base_path}/LBFL/logs') if os.path.isdir(f'{log_base_path}/LBFL/logs/{folder}') and f"mal_{mal}" in folder and f"attack_{attack_type}" in folder]

        for lp in LBFL_log_paths:
            # Open and load the pickle file
            with open(f'{lp}/logger.pickle', 'rb') as file:
                logger = pickle.load(file)['malicious_winning_count']
                for comm_round, malicious_winning_count in logger.items():
                    if malicious_winning_count:
                        plt.scatter(comm_round, y_offset, marker='o', color='red')
                        plt.text(comm_round, y_offset, str(malicious_winning_count), ha='center', va='center', color='black')
            y_axis_labels.append(f'{mal} Attackers - {attack_type_map[attack_type]}')
            y_offset += 1

legend_handles = [
    Line2D([0], [0], marker='o', color='w', markerfacecolor='red', markersize=10, label='Count of devices that appended a malicious block')
]

plt.legend(handles=legend_handles, loc='best', prop={'size': 10})


plt.xlabel('Communication Round')
plt.ylabel('Run Name')
plt.title('Malicious Winning Count')
plt.yticks(range(len(y_axis_labels)), y_axis_labels)

plt.tight_layout()

plt.savefig(f'{log_base_path}/LBFL/logs/malicious_winning_count.png', dpi=300)
plt.clf()
# plt.show()