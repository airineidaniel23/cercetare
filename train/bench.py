import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, confusion_matrix, precision_recall_fscore_support

# Paths to directories
test_inferred_path = 'testMajority'
test_inferred_path = 'testInferred/1'
test_target_path = 'testTarget/1'

# Load all text files and convert one-hot encodings to class labels
def load_class_labels(folder_path):
    labels = []
    for filename in sorted(os.listdir(folder_path)):
        with open(os.path.join(folder_path, filename), 'r') as file:
            one_hot_encoding = file.readline().strip()
            label = np.argmax([int(i) for i in one_hot_encoding.split()])
            labels.append(label)
    return np.array(labels)

# Load predictions and ground truths
y_pred = load_class_labels(test_inferred_path)
y_true = load_class_labels(test_target_path)

# Calculate accuracy
accuracy = accuracy_score(y_true, y_pred)
print(f'Accuracy: {accuracy:.4f}')

# Calculate confusion matrix, forcing an 8x8 shape
conf_matrix = confusion_matrix(y_true, y_pred, labels=np.arange(8))
print('Confusion Matrix:')
print(conf_matrix)

# Visualize the confusion matrix using seaborn
plt.figure(figsize=(10, 8))
sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", xticklabels=np.arange(8), yticklabels=np.arange(8))
plt.title("Confusion Matrix")
plt.xlabel("Predicted Class")
plt.ylabel("True Class")
plt.show()

# Calculate precision, recall, F1 score, and specificity, ensuring all 8 classes are included
precision, recall, f1_score, _ = precision_recall_fscore_support(
    y_true, y_pred, labels=np.arange(8), zero_division=0
)

# Calculate specificity for each class
specificity = []
for i in range(8):
    tn = conf_matrix.sum() - (conf_matrix[i, :].sum() + conf_matrix[:, i].sum() - conf_matrix[i, i])
    fp = conf_matrix[:, i].sum() - conf_matrix[i, i]
    specificity.append(tn / (tn + fp) if (tn + fp) > 0 else 0)

# Create a DataFrame to organize and display metrics
metrics_df = pd.DataFrame({
    'Class': np.arange(8),
    'Precision': precision,
    'Recall': recall,
    'F1 Score': f1_score,
    'Specificity': specificity
})

# Display metrics
print(metrics_df)

# Display overall accuracy
print(f'\nOverall Accuracy: {accuracy:.4f}')
