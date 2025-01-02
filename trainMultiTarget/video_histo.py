import os
import matplotlib.pyplot as plt
import pdb
def plot_one_hot_folder(folder_path):
    # Get list of all .txt files in the folder, sorted by filename
    files = [f for f in os.listdir(folder_path) if f.endswith('.txt')]
    files.sort()
    
    y_values = []
    
    for filename in files:
        file_path = os.path.join(folder_path, filename)
        
        # Read the single line containing the one-hot vector
        with open(file_path, 'r') as f:
            line = f.readline().strip()
            one_hot = [int(x) for x in line.split()]
            
            # Find which class (1..8) is active
            active_class = one_hot.index(1) + 1
            
            # Map the class to the desired y-value: y = 5 - c
            # class 5 => 0
            # class 6 => -1
            # class 7 => -2
            # class 8 => -3
            # class 4 => 1
            # class 3 => 2
            # class 2 => 3
            # class 1 => 4
            y_val = 6 - active_class
            y_values.append(y_val)
    
    # Plotting
    plt.figure(figsize=(8,4))
    plt.bar(range(len(y_values)), y_values, color='skyblue', edgecolor='black')
    plt.title('One-Hot Classes (Transformed)')
    plt.xlabel('File index')
    plt.ylabel('Class (transformed: 5 - class)')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.show()

# Example usage:
plot_one_hot_folder('testInferred/1/')
