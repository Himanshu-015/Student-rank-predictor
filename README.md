# Student-rank-predictor
1. Import Necessary Libraries
We'll start by importing all the required libraries.

python
Copy
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error
import matplotlib.pyplot as plt

2. Load Data
We’ll load the three datasets:

current_quiz_data.csv: Contains student quiz performance data.
historical_quiz_data.csv: Contains historical quiz performance and actual NEET ranks.
college_cutoffs.csv: Contains college names and their NEET cutoff ranks.
python
Copy
# Load the datasets
quiz_data = pd.read_csv('current_quiz_data.csv')  # Current quiz performance data
historical_data = pd.read_csv('historical_quiz_data.csv')  # Historical quiz scores and actual NEET rank
colleges = pd.read_csv('college_cutoffs.csv')  # College name and NEET rank cutoff

3. Data Preprocessing and Feature Engineering
Here, we’ll preprocess the quiz_data to extract features like accuracy per topic and the performance score over time.

python
Copy
# Step 1: Feature Engineering

# Calculate if the student's answer was correct for each question
quiz_data['is_correct'] = quiz_data['selected_option_id'] == quiz_data['correct_answer']

# Calculate accuracy per topic (Physics, Chemistry, Biology)
topic_accuracy = quiz_data.groupby('topic')['is_correct'].mean()

# Calculate the average score per student based on historical data (how well each student performed in past quizzes)
historical_scores = historical_data.groupby('student_id')['score'].mean()

# Merge the features into a DataFrame for easier modeling
features = pd.DataFrame({
    'avg_score': historical_scores,
    'physics_accuracy': topic_accuracy.get('Physics', 0),
    'chemistry_accuracy': topic_accuracy.get('Chemistry', 0),
    'biology_accuracy': topic_accuracy.get('Biology', 0),
})

# Make sure to match student IDs to the target labels (NEET rank)
labels = historical_data.groupby('student_id')['neet_rank'].first()  # Assuming the first entry corresponds to their rank
4. Train-Test Split
We’ll split the data into training and testing sets. This allows us to test how well the model generalizes to unseen data.

python
Copy
# Step 2: Train-Test Split

X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=42)

# Normalize the features (important for regression models)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
5. Train the Model
Now, we’ll train a Linear Regression model to predict the NEET rank based on the students' quiz performance.

python
Copy
# Step 3: Model Training - Linear Regression

model = LinearRegression()
model.fit(X_train, y_train)
6. Model Evaluation
After training the model, we will evaluate its performance using the Mean Absolute Error (MAE) metric. This tells us the average difference between the predicted and actual NEET ranks.

python
Copy
# Step 4: Model Evaluation

predicted_ranks = model.predict(X_test)

# Calculate Mean Absolute Error
mae = mean_absolute_error(y_test, predicted_ranks)
print(f"Mean Absolute Error: {mae}")

# Visualize the predicted vs. actual ranks
plt.figure(figsize=(10, 6))
plt.scatter(y_test, predicted_ranks)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], color='red', lw=2)
plt.xlabel('Actual Ranks')
plt.ylabel('Predicted Ranks')
plt.title('Actual vs Predicted NEET Ranks')
plt.show()
7. Predict Likely Colleges
Given a student's predicted NEET rank, we can match them to likely colleges. Colleges have cutoffs, and if the predicted rank is below the cutoff, the student is likely to be admitted.

python
Copy
# Step 5: Predicting Likely Colleges Based on Predicted Rank

# Sort the colleges by their cutoff rank (ascending order)
colleges_sorted = colleges.sort_values(by='cutoff_rank')

# Function to predict likely colleges based on predicted rank
def predict_college(predicted_rank, colleges_sorted):
    # Filter colleges that have a cutoff rank greater than or equal to the predicted rank
    possible_colleges = colleges_sorted[colleges_sorted['cutoff_rank'] >= predicted_rank]
    return possible_colleges.head(5)  # Return the top 5 colleges

# Example: Predict the likely colleges for a student with a predicted rank of 2000
predicted_rank = 2000
likely_colleges = predict_college(predicted_rank, colleges_sorted)
print("\nLikely Colleges Based on Predicted Rank:")
print(likely_colleges)
8. Predict for an Individual Student
We can also predict the NEET rank and likely colleges for an individual student based on their quiz performance.

python
Copy
# Step 6: Predict for an Individual Student

# Example features for a new student
individual_student_features = pd.DataFrame({
    'avg_score': [75],  # Example score from quizzes
    'physics_accuracy': [0.8],  # Example topic accuracy for Physics
    'chemistry_accuracy': [0.7],  # Example topic accuracy for Chemistry
    'biology_accuracy': [0.9],  # Example topic accuracy for Biology
})

# Normalize the features for the individual student
individual_student_features = scaler.transform(individual_student_features)

# Predict the student's rank
predicted_individual_rank = model.predict(individual_student_features)
print(f"\nPredicted NEET Rank for the Individual Student: {predicted_individual_rank[0]}")

# Predict the most likely colleges for the individual student
likely_colleges_individual = predict_college(predicted_individual_rank[0], colleges_sorted)
print("\nLikely Colleges Based on Individual Student's Predicted Rank:")
print(likely_colleges_individual)
