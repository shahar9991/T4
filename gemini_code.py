import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import os

# 1. Read the CSV file containing the participant task data.
# [cite_start]The data is already summarized into a single matrix where each row is a participant and each column is a task[cite: 3].
file_path = r"C:\Users\shaha\Downloads\participant_task_matrix.csv"
df = pd.read_csv(file_path, index_col=0)

# Clean the data by dropping any rows with missing values (NaN) if they exist.
df = df.dropna()

# [cite_start]The instructions recommend averaging the 'Accuracy' for each task[cite: 4].
# The provided data is already structured this way, so we proceed to the next step.

# [cite_start]2. Calculate the correlation matrix between the tasks (columns)[cite: 5].
correlation_matrix = df.corr()

# [cite_start]3. Convert the correlation matrix into a Representational Dissimilarity Matrix (RDM)[cite: 6].
# [cite_start]This is done by calculating 1 minus the correlation matrix[cite: 6].
rdm_matrix = 1 - correlation_matrix

# 4. Create a folder to save the output, if it doesn't already exist.
output_folder = 'RDM_Output'
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# [cite_start]5. Create a colorful visualization (heatmap) of the RDM matrix[cite: 7].
plt.figure(figsize=(10, 8))
sns.heatmap(rdm_matrix, annot=True, cmap='viridis', fmt=".2f", linewidths=.5)
plt.title('Representational Dissimilarity Matrix (RDM)')
plt.tight_layout()

# 6. Save the generated plot to the created folder.
output_file_path = os.path.join(output_folder, 'rdm_heatmap.png')
plt.savefig(output_file_path)

# Display the plot on the screen (optional).
plt.show()

print(f"RDM heatmap saved to: {output_file_path}")

# [cite_start]Bonus: The instructions also suggest including reaction times[cite: 10].
# This would require an additional data file with reaction time information, which was not provided.