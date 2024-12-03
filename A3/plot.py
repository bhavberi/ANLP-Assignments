import matplotlib.pyplot as plt
import numpy as np

# Sample data for training and validation loss
epochs = np.arange(1, 6)
# lora
# train_loss = [0.6017, 0.5584, 0.4867, 0.4391, 0.4036]
# val_loss = [19.2922, 16.8264, 14.7158, 13.4501, 13.0669]
# last
# train_loss = [0.5541, 0.5099, 0.4930, 0.4921, 0.4758]
# val_loss = [17.2898, 16.2494, 16.3394, 15.6558, 15.6085]
# prompt
train_loss = [0.6177, 0.6176, 0.6174, 0.6172, 0.6169]
val_loss = [19.9644, 19.9633, 19.9622, 19.9742, 19.9612]

train_loss = [x * 32 for x in train_loss]

# Create the plot
plt.figure(figsize=(10, 6))
plt.plot(epochs, train_loss, 'o-', color='red', label='Training Loss')
plt.plot(epochs, val_loss, 'o-', color='blue', label='Validation Loss')

# Annotate the points with loss values
for i, txt in enumerate(np.round(train_loss, 3)):
    plt.annotate(f'{txt:.3f}', (epochs[i], train_loss[i]), 
                 textcoords="offset points", 
                 xytext=(0,10), ha='center', fontsize=9)

for i, txt in enumerate(np.round(val_loss, 3)):
    plt.annotate(f'{txt:.3f}', (epochs[i], val_loss[i]), 
                 textcoords="offset points", 
                 xytext=(0,10), ha='center', fontsize=9)

# Add titles and labels
plt.title("Train and Val Loss for Soft Prompt FT")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.xticks(epochs)
plt.legend()
plt.grid(True)

# Save the plot
plt.savefig('./training_prompt.png', bbox_inches='tight')
# plt.show()
