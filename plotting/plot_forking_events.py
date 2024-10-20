import pickle
import matplotlib.pyplot as plt
import numpy as np

# Paths to the log files
LBFL_log_paths = [
    "logs/LBFL_seed_40_10072024_202206_rounds_3_epochs_500_val_*_mal_0_attack_0_noise_3_rewind_1_nsamples_20_nclasses_3",
    "logs/LBFL_seed_50_10072024_203118_rounds_3_epochs_500_val_*_mal_0_attack_0_noise_3_rewind_1_nsamples_20_nclasses_3",
    "logs/LBFL_seed_60_10072024_203730_rounds_3_epochs_500_val_*_mal_0_attack_0_noise_3_rewind_1_nsamples_20_nclasses_3"
]

# Set y-axis labels
y_axis_labels = ["Run 1", "Run 2", "Run 3"]
plt.yticks(range(len(y_axis_labels)), y_axis_labels)

# Iterate over each log path and corresponding y-axis position
for i, lp in enumerate(LBFL_log_paths):
    # Open and load the pickle file
    with open(f'{lp}/logger.pickle', 'rb') as file:
        logger = pickle.load(file)
        # Iterate over each communication round and forking indicator
        for comm_round, forking_indicator in logger['forking_event'].items():
            # Plot a circle at (comm_round, y_position) if forking_indicator is true
            if forking_indicator == 1:
                plt.plot(i, comm_round, 'o')

# Add labels and title to the plot
plt.xlabel('Communication Round')
plt.ylabel('Run')
plt.title('Forking Events in Different Runs')
plt.grid(True)
plt.show()