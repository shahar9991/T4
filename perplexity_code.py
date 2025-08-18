import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import os

# Read data from the CSV file
filename = r"C:\Users\shaha\Downloads\participant_matrix.csv"
data = pd.read_csv(filename, index_col=0)

# Normalize the data for each task (z-score)
normalized_data = (data - data.mean()) / data.std()

# Compute the Representational Dissimilarity Matrix (RDM) based on 1 minus the correlation between tasks
corr = normalized_data.corr()
rdm = 1 - corr

# Create a new folder for the results
output_folder = 'RDM_results'
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# Create a colored heatmap visualization of the RDM matrix
plt.figure(figsize=(8, 6))
sns.heatmap(rdm, annot=True, cmap='coolwarm', square=True)
plt.title('Representational Dissimilarity Matrix (RDM)')
plt.tight_layout()

# Save the image in the results folder
plot_filepath = os.path.join(output_folder, 'RDM_heatmap.png')
plt.savefig(plot_filepath)
plt.close()

print(f'Results saved in folder: {output_folder}')
