
import matplotlib.pyplot as plt
import numpy as np

# Data
Tmove = ['1', '2', '3', '4']
# MAE_3 = [13.2866, 13.3738, 13.3628, 13.3456]
# MAE_6 = [18.7270, 19.0829, 18.8706, 19.0242]
MAE_12 = [24.2397, 24.8458, 24.4380, 24.6536]

# topk = ['10', '20', '30', '40', '50']
# MAE_3 = [13.6622, 13.2866, 14.1665, 13.3018, 14.1128]
# MAE_6 = [18.8693, 18.7270, 19.6338, 18.8014, 19.5609]
# MAE_12 = [23.8076, 24.2397, 24.6066, 24.3351, 24.6070]

# Plot
fig, ax = plt.subplots(figsize=(10, 6))  # Increase figure size

bars = ax.bar(Tmove, MAE_12, color='black')

# Adding text annotations
for bar in bars:
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width() / 2., height + 0.002, f'{height:.4f}', ha='center', color='black', fontsize=14, fontweight='bold', fontname='Times New Roman')

# Set y-axis to logarithmic scale
ax.set_yscale('log')

# Labels and title
ax.set_xlabel('Tm', fontsize=20, fontweight='bold', fontname='Times New Roman')
ax.set_ylabel('MAE 12', fontsize=20, fontweight='bold', fontname='Times New Roman')


# Save the plot
fig.savefig('MAE_Tmove.png')

# Display the plot
plt.show()




