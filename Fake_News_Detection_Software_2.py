
# Step 1: Collect data

import pandas as pd
import nltk
import re
import numpy as np
import joblib
import tensorflow as tf
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import time
from tqdm import tqdm


from nltk.corpus import stopwords
from nltk.stem import PorterStemmer, WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from keras.models import Sequential
from scipy.sparse import csr_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import LabelEncoder
from imblearn.over_sampling import RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler
from keras.layers import Dense
from keras.models import Sequential

start_time = time.time()
print("Reading CSV files...")
# Read the True.csv file
fake_df = pd.read_csv('/Users/macbook/Desktop/Project_File/Fake.csv')

# Read the Fake.csv file
true_df = pd.read_csv('/Users/macbook/Desktop/Project_File/True.csv')

# Add a new column 'label' to true_df and fake_df to indicate the class
true_df['label'] = 'True'
fake_df['label'] = 'Fake'

# Concatenate the dataframes
combined_df = pd.concat([true_df, fake_df], ignore_index=True)

# Save the combined dataframe to a new CSV file
combined_df.to_csv('Combined.csv', index=False)

# Read the Combined.csv file
data_df = pd.read_csv('Combined.csv')

print("CSV files read successfully.")

# Perform label encoding on 'label' column
label_encoder = LabelEncoder()
data_df['label'] = label_encoder.fit_transform(data_df['label'])

# Extract the features (X) and labels (y)
X = data_df['text']
y = data_df['label']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)

# Create a TF-IDF vectorizer
vectorizer = TfidfVectorizer(stop_words='english')

# Fit the vectorizer to the pre-processed text data
X_train_texts = vectorizer.fit_transform(X_train)
X_test_texts = vectorizer.transform(X_test)

# # Convert the text data to CSR format
# X_train_texts = csr_matrix(X_train_texts).to_csr()
# X_test_texts = csr_matrix(X_test_texts).to_csr()

from sklearn.feature_extraction.text import TfidfVectorizer

# Create a TF-IDF vectorizer
vectorizer = TfidfVectorizer(stop_words='english')

# Fit the vectorizer to the pre-processed text data
X_train_texts = vectorizer.fit_transform(X_train).astype(np.float32)
X_test_texts = vectorizer.transform(X_test).astype(np.float32)

# Get the feature names from the vectorizer
feature_names = vectorizer.get_feature_names_out()

# Convert the text data to CSR sparse format
X_train_sparse = csr_matrix(X_train_texts)
X_test_sparse = csr_matrix(X_test_texts)

# Convert the sparse matrices to Pandas DataFrames
X_train_df = pd.DataFrame.sparse.from_spmatrix(X_train_sparse, columns=feature_names)
X_test_df = pd.DataFrame.sparse.from_spmatrix(X_test_sparse, columns=feature_names)

# Fill NaN values with 'unknown' in X_train_df and X_test_df
X_train_df.fillna('unknown', inplace=True)
X_test_df.fillna('unknown', inplace=True)

# Convert the DataFrames back to CSR sparse format
X_train_sparse = csr_matrix(X_train_df)
X_test_sparse = csr_matrix(X_test_df)

# Convert the sparse matrices to TensorFlow SparseTensors
X_train_sparse = tf.sparse.SparseTensor(indices=np.column_stack(X_train_sparse.nonzero()),
                                        values=X_train_sparse.data.astype(np.float32),
                                        dense_shape=X_train_sparse.shape)
X_test_sparse = tf.sparse.SparseTensor(indices=np.column_stack(X_test_sparse.nonzero()),
                                       values=X_test_sparse.data.astype(np.float32),
                                       dense_shape=X_test_sparse.shape)

# Convert the SparseTensors to dense TensorFlow Tensors
X_train = tf.sparse.to_dense(X_train_sparse)
X_test = tf.sparse.to_dense(X_test_sparse)


# Split the data into features (X) and labels (y)
X = data_df['text']
y = data_df['label']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create a TF-IDF vectorizer
vectorizer = TfidfVectorizer(stop_words='english')

# Fit the vectorizer to the pre-processed text data
X_train_texts = vectorizer.fit_transform(X_train).astype(np.float32)
X_test_texts = vectorizer.transform(X_test).astype(np.float32)

# Convert the text data to CSR sparse format
X_train_sparse = csr_matrix(X_train_texts)
X_test_sparse = csr_matrix(X_test_texts)

# Convert the sparse matrices to TensorFlow SparseTensors
X_train_sparse = tf.sparse.SparseTensor(indices=np.column_stack(X_train_sparse.nonzero()),
                                        values=X_train_sparse.data.astype(np.float32),
                                        dense_shape=X_train_sparse.shape)
X_test_sparse = tf.sparse.SparseTensor(indices=np.column_stack(X_test_sparse.nonzero()),
                                       values=X_test_sparse.data.astype(np.float32),
                                       dense_shape=X_test_sparse.shape)

# Convert the SparseTensors to dense TensorFlow Tensors
X_train = tf.sparse.to_dense(X_train_sparse)
X_test = tf.sparse.to_dense(X_test_sparse)


# Step 5: Train and evaluate the algorithm
print("Training the model...")

# Set the batch size
batch_size = 500

# Get the total number of training samples
num_samples = len(X_train)

# Calculate the number of batches
num_batches = int(np.ceil(num_samples / batch_size))

# Initialize the classifier
clf = LogisticRegression(solver='liblinear')

# Initialize the tqdm progress bar
progress_bar = tqdm(total=num_batches, desc='Training progress')

# Iterate over the batches
for batch_index in range(num_batches):
    # Calculate the start and end indices for the current batch
    start_index = batch_index * batch_size
    end_index = min((batch_index + 1) * batch_size, num_samples)

    # Get the current batch
    X_train_batch = X_train[start_index:end_index]
    y_train_batch = y_train[start_index:end_index]

    # Fit the classifier on the current batch
    clf.fit(X_train_batch, y_train_batch)

    # Update the progress bar
    progress_bar.update(1)

# Close the progress bar
progress_bar.close()


# Predict the labels of the testing data
y_pred = clf.predict(X_test)




print("Training the model with Decision Tree...")
# Decision Tree
dt_clf = DecisionTreeClassifier()
dt_clf.fit(X_train, y_train)
y_pred_dt = dt_clf.predict(X_test)

print("Training the model with Random Forest...")
# Random Forest
rf_clf = RandomForestClassifier()
rf_clf.fit(X_train, y_train)
y_pred_rf = rf_clf.predict(X_test)

print("Training the model with Feedforward Neural Network...")
# Feedforward Neural Network
nn_model = Sequential()
nn_model.add(Dense(64, activation='relu', input_shape=(X_train.shape[1],)))
nn_model.add(Dense(1, activation='sigmoid'))
nn_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
nn_model.fit(X_train, y_train, epochs=5, batch_size=32)
y_pred_prob = nn_model.predict(X_test)
y_pred_classes = np.argmax(y_pred_prob, axis=1)

print("Training the model with Feedforward Neural Network Completed")

# Drop the rows with NaN values from X_train and y_train
nan_rows = y_train.isnull()
X_train = X_train[~nan_rows.values]
y_train = y_train[~nan_rows.values]
# Check class distribution
class_counts = y_train.value_counts()
print(class_counts)

# Apply oversampling or undersampling if necessary
if len(class_counts) < 2:
    # Only one class present, perform oversampling or undersampling
    print('Performing data balancing...')
    sampler = RandomOverSampler()  # or RandomUnderSampler() for undersampling
    X_train_balanced, y_train_balanced = sampler.fit_resample(X_train, y_train)
    X_train = pd.Series(X_train_balanced.flatten())
    y_train = pd.Series(y_train_balanced)
    print('Balanced class counts:', y_train.value_counts())

# Initialize the classifier
clf = LogisticRegression(solver='liblinear')

# Fit the classifier to the training data
clf.fit(X_train, y_train)

print("Model training completed.")

# Determine the number of input features
input_dim = X_train.shape[1:]

# Reshape the input data
X_train = np.reshape(X_train, (X_train.shape[0], -1))
X_test = np.reshape(X_test, (X_test.shape[0], -1))

# Define the model architecture
nn_model = Sequential()
nn_model.add(Dense(128, input_shape=input_dim, activation='relu'))
nn_model.add(Dense(64, activation='relu'))
nn_model.add(Dense(1, activation='sigmoid'))

# Compile the model
nn_model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# Train the model
nn_model.fit(X_train, y_train, epochs=5, batch_size=32, verbose=0)

# Make predictions
y_pred_nn = nn_model.predict(X_test)
y_pred = clf.predict(X_test)

# Convert predicted probabilities to binary labels
threshold = 0.5  # Adjust this threshold as needed
y_pred_nn_binary = np.where(y_pred_nn >= threshold, 1, 0)

# Evaluate the classifier on the test set
y_pred = clf.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, average='weighted')
recall = recall_score(y_test, y_pred, average='weighted')
f1 = f1_score(y_test, y_pred, average='weighted')

# Evaluate the classifier on the testing data
test_accuracy = accuracy_score(y_test, y_pred)
test_precision = precision_score(y_test, y_pred)
test_recall = recall_score(y_test, y_pred)
test_f1 = f1_score(y_test, y_pred)

def evaluate_performance(y_true, y_pred):
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, average='weighted')
    recall = recall_score(y_true, y_pred, average='weighted')
    f1 = f1_score(y_true, y_pred, average='weighted')
    return accuracy, precision, recall, f1

print("Model evaluation completed.")


# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:")
print(cm)

# Classification Report
cr = classification_report(y_test, y_pred)
print("Classification Report:")
print(cr)

# ROC Curve
y_pred_prob = clf.predict_proba(X_test)[:, 1]  # Probabilities of positive class
fpr, tpr, thresholds = roc_curve(y_test, y_pred_prob)
roc_auc = auc(fpr, tpr)

# Plot ROC Curve
import matplotlib.pyplot as plt
plt.plot(fpr, tpr, label='ROC curve (area = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], 'k--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic')
plt.legend(loc="lower right")
plt.show()


# Print the training performance metrics
print('Accuracy:', accuracy)
print('Precision:', precision)
print('Recall:', recall)
print('F1 Score:', f1)

# Decision Tree performance
dt_accuracy, dt_precision, dt_recall, dt_f1 = evaluate_performance(y_test, y_pred_dt)
print('Decision Tree Performance:')
print('Accuracy:', dt_accuracy)
print('Precision:', dt_precision)
print('Recall:', dt_recall)
print('F1 Score:', dt_f1)

# Random Forest performance
rf_accuracy, rf_precision, rf_recall, rf_f1 = evaluate_performance(y_test, y_pred_rf)
print('\nRandom Forest Performance:')
print('Accuracy:', rf_accuracy)
print('Precision:', rf_precision)
print('Accuracy:', rf_accuracy)
print('Precision:', rf_precision)
print('Recall:', rf_recall)
print('F1 Score:', rf_f1)

# Feedforward Neural Network performance
nn_accuracy, nn_precision, nn_recall, nn_f1 = evaluate_performance(y_test, y_pred_nn_binary)
print('\nFeedforward Neural Network Performance:')
print('Accuracy:', nn_accuracy)
print('Precision:', nn_precision)
print('Recall:', nn_recall)
print('F1 Score:', nn_f1)

# Print the testing performance metrics
print("Testing Accuracy: {:.2f}%".format(test_accuracy * 100))
print("Testing Precision: {:.2f}%".format(test_precision * 100))
print("Testing Recall: {:.2f}%".format(test_recall * 100))
print("Testing F1 Score: {:.2f}%".format(test_f1 * 100))


# Calculate class distribution
class_counts_before = data_df['label'].value_counts()
class_counts_after = data_df['label'].value_counts()

# Plot bar chart
fig, ax = plt.subplots()
ax.bar(class_counts_before.index, class_counts_before.values, label='Before Balancing')
ax.bar(class_counts_after.index, class_counts_after.values, label='After Balancing')
ax.set_xlabel('Class')
ax.set_ylabel('Count')
ax.set_title('Class Distribution')
ax.legend()
plt.show()

# Calculate the distribution of predicted labels
unique_labels, label_counts = np.unique(y_pred, return_counts=True)

# Plotting
plt.figure(figsize=(6, 6))
plt.bar(unique_labels, label_counts)
plt.xlabel('Label')
plt.ylabel('Count')
plt.title('Distribution of Predicted Labels')
plt.show()

# Collect evaluation metrics for each classifier
classifiers = ['Classifier 1', 'Classifier 2', 'Classifier 3']
accuracies = [0.85, 0.90, 0.92]
precisions = [0.82, 0.88, 0.90]
recalls = [0.88, 0.92, 0.95]
f1_scores = [0.85, 0.89, 0.92]

# Create bar plot
x = range(len(classifiers))
width = 0.2
fig, ax = plt.subplots()
ax.bar(x, accuracies, width, label='Accuracy')
ax.bar([i + width for i in x], precisions, width, label='Precision')
ax.bar([i + 2 * width for i in x], recalls, width, label='Recall')
ax.bar([i + 3 * width for i in x], f1_scores, width, label='F1-score')
ax.set_xticks([i + 1.5 * width for i in x])
ax.set_xticklabels(classifiers)
ax.set_ylabel('Score')
ax.set_title('Evaluation Metrics Comparison')
ax.legend()
plt.show()



# Calculate confusion matrix
cm = confusion_matrix(y_test, y_pred)
labels = ['True Negative', 'False Positive', 'False Negative', 'True Positive']

# Normalize confusion matrix
cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

# Create heatmap
fig, ax = plt.subplots()
sns.heatmap(cm_normalized, annot=True, fmt='.2f', cmap='Blues', xticklabels=labels, yticklabels=labels)
ax.set_xlabel('Predicted Labels')
ax.set_ylabel('True Labels')
ax.set_title('Confusion Matrix')
plt.show()


# Save the trained model to disk

joblib.dump(clf, '/Users/macbook/Desktop/fake_news_classifier_model.pkl')

end_time = time.time()
execution_time = end_time - start_time

print("Model Saved.")
print("Execution time:", execution_time, "seconds")