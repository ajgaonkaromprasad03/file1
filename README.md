/*
# file1
this one for ref


#slip01
Q.1 

# Create two vectors of integer type
vector1 <- c(2, 4, 6, 8)
vector2 <- c(1, 3, 5, 7)

# Add two vectors
add_result <- vector1 + vector2
cat("Vector Addition Result: ", add_result, "\n")

# Multiply two vectors
multiply_result <- vector1 * vector2
cat("Vector Multiplication Result: ", multiply_result, "\n")

# Divide two vectors
# Note: Make sure the divisor vector (vector2) does not contain zeros to avoid division by zero errors
divide_result <- vector1 / vector2
cat("Vector Division Result: ", divide_result, "\n")

Q.2

C:\\Users\\siddhii\\.spyder-py3\\student_scores.csv

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error,mean_squared_error,r2_score

data=pd.read_csv('C:\\Users\\siddhii\\.spyder-py3\\student_marks.csv')
x=data[['Physics']]
y=data[['Chemistry']]

model=LinearRegression()
model.fit(x,y)

plt.scatter(x,y)
plt.plot(x,model.predict(x))


predictions=model.predict(x)

mae=mean_absolute_error(y,predictions)
mse=mean_squared_error(y, predictions)
rmse = np.sqrt(mse)

print("mean absolute error",mae);
print("mean squared error",mse);

print("root mean sq error",rmse);


#slip02

Q.1 

# R Program to find the multiplicationtable (from 1 to 10)

# take input from the user

num = as.integer(readline(prompt = "Enter a number: "))

# use for loop to iterate 10 times
for(i in 1:10) {
  print(paste(num,'x', i, '=', num*i))
}

Q.2

# Step 1: Import necessary libraries
#import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans

# Step 2: Generate a synthetic dataset
data, true_labels = make_blobs(n_samples=300, centers=4, random_state=42)

# Step 3: Visualize the synthetic dataset
plt.scatter(data[:, 0], data[:, 1], s=30, cmap='viridis')
plt.title("Synthetic Dataset")
plt.show()

# Step 4: Choose the number of clusters (k)
k = 4

# Step 5: Create and fit the KMeans model
kmeans = KMeans(n_clusters=k, random_state=42)
kmeans.fit(data)

# Step 6: Get cluster centers and labels
centroids = kmeans.cluster_centers_
labels = kmeans.labels_

# Step 7: Visualize the clusters and centroids
plt.scatter(data[:, 0], data[:, 1], c=labels, s=30, cmap='viridis')
plt.scatter(centroids[:, 0], centroids[:, 1], marker='X', s=200, linewidths=3, color='red')
plt.title("K-means Clustering")
plt.show()



#slip03

Q.1 


n=567
Reverse=function(n)
{ 
  sum=0
  rev=0
  while(n>0)
  { 
    r=n%%10
    sum=sum+r 
    rev=rev*10+r
    n=n%/%10
  } 
  print(rev) 
  print(sum) 
} 
Reverse(n)


Q.2

import numpy as np

# Given data
x = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 11, 13])
y = np.array([1, 3, 2, 5, 7, 8, 8, 9, 10, 12, 16, 18])

# Calculate the mean of x and y
mean_x = np.mean(x)
mean_y = np.mean(y)

# Calculate the slope (b1) and intercept (b0) using the formula
b1 = np.sum((x - mean_x) * (y - mean_y)) / np.sum((x - mean_x) ** 2)
b0 = mean_y - b1 * mean_x

# Print the estimated coefficients
print("Estimated Coefficient b0 (intercept):", b0)
print("Estimated Coefficient b1 (slope):", b1)


#slip04

Q.1


m1 = matrix(c(1, 2, 3, 4, 5, 6), nrow = 2)
print("Matrix-1:")
print(m1)
m2 = matrix(c(0, 1, 2, 3, 0, 2), nrow = 2)
print("Matrix-2:")
print(m2)

result = m1 + m2
print("Result of addition")
print(result)


Q.2

from sklearn.naive_bayes import MultinomialNB
from sklearn.preprocessing import LabelEncoder

# Given data
weather = ['Sunny','Sunny','Overcast','Rainy','Rainy','Rainy','Overcast','Sunny','Sunny','Rainy','Sunny','Overcast','Overcast','Rainy']
temp = ['Hot','Hot','Hot','Mild','Cool','Cool','Cool','Mild','Cool','Mild','Mild','Mild','Hot','Mild']
play = ['No','No','Yes','Yes','Yes','No','Yes','No','Yes','Yes','Yes','Yes','Yes','No']

# Encode categorical variables
le_weather = LabelEncoder()
le_temp = LabelEncoder()
le_play = LabelEncoder()

weather_encoded = le_weather.fit_transform(weather)
temp_encoded = le_temp.fit_transform(temp)
play_encoded = le_play.fit_transform(play)

# Combine the encoded features into a feature matrix
X = list(zip(weather_encoded, temp_encoded))

# Create and train the Naïve Bayes model
model = MultinomialNB()
model.fit(X, play_encoded)

# New tuple to predict [0: Overcast, 2: Mild]
new_data = [(le_weather.transform(['Overcast'])[0], le_temp.transform(['Mild'])[0])]

# Predict using the model
prediction = model.predict(new_data)

# Decode the prediction
predicted_class = le_play.inverse_transform(prediction)

# Print the prediction
print("Prediction: Whether to play sports or not -", predicted_class[0])


#slip05

Q.1


# Create two factors
factor1 <- factor(c("A", "B", "C"))
factor2 <- factor(c("X", "Y", "Z"))

# Concatenate the two factors
concatenated_factor <- c(factor1, factor2)

# Print the original factors
cat("Factor 1: ", levels(factor1), "\n")
cat("Factor 2: ", levels(factor2), "\n")

# Print the concatenated factor
cat("Concatenated Factor: ", levels(concatenated_factor), "\n")


Q.2


import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Load the dataset (downloaded from Kaggle)
# Make sure to adjust the path to the actual location where you have saved the dataset
data = pd.read_csv('C:\\Users\\siddhii\\.spyder-py3\\diabetes.csv')

# Display the first few rows of the dataset
print(data.head())

# Extract features (X) and target variable (y)
X = data.drop('Outcome', axis=1)
y = data['Outcome']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Build the Decision Tree Classifier
clf = DecisionTreeClassifier(random_state=42)
clf.fit(X_train, y_train)

# Make predictions on the test set
y_pred = clf.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)
class_report = classification_report(y_test, y_pred)

# Print the evaluation metrics
print(f'Accuracy: {accuracy:.4f}\n')
print('Confusion Matrix:\n', conf_matrix)
print('\nClassification Report:\n', class_report)


#slip06

Q.1

a = c(10,20,10,10,40,50,20,30)
b = c(10,30,10,20,0,50,30,30)
print("Original data frame:")
ab = data.frame(a,b)
print(ab)
print("Duplicate elements of the said data frame:")
print(duplicated(ab))
print("Unique rows of the said data frame:")
print(unique(ab))

Q.2

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
dataset = pd.read_csv('C:\\Users\\siddhii\\.spyder-py3\\Mall_Customers.csv')
X = dataset.iloc[:, [3, 4]].values
"""from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)""" 
# Feature Scaling
"""from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)
sc_y = StandardScaler()
y_train = sc_y.fit_transform(y_train)"""
import scipy.cluster.hierarchy as sch
dendrogram = sch.dendrogram(sch.linkage(X, method = 'ward'))
plt.title('Dendrogram')
plt.xlabel('Customers')
plt.ylabel('Euclidean distances')
plt.show()
from sklearn.cluster import AgglomerativeClustering
hc = AgglomerativeClustering(n_clusters = 5, affinity = 'euclidean', linkage = 'ward')
y_hc = hc.fit_predict(X)
plt.scatter(X[y_hc == 0, 0], X[y_hc == 0, 1], s = 100, c = 'red', label = 'Cluster 1')
plt.scatter(X[y_hc == 1, 0], X[y_hc == 1, 1], s = 100, c = 'blue', label = 'Cluster 2') 
plt.scatter(X[y_hc == 2, 0], X[y_hc == 2, 1], s = 100, c = 'green', label = 'Cluster 3')
plt.scatter(X[y_hc == 3, 0], X[y_hc == 3, 1], s = 100, c = 'cyan', label = 'Cluster 4')
plt.scatter(X[y_hc == 4, 0], X[y_hc == 4, 1], s = 100, c = 'magenta', label = 'Cluster 5')
plt.title('Clusters of customers')
plt.xlabel('Annual_Income_(k$)')
plt.ylabel('Spending_Score')
plt.legend()
plt.show()

#slip07

Q.1

print("Sequence of numbers from 20 to 50:")
print(seq(20,50))
print("Mean of numbers from 20 to 60:")
print(mean(20:60))
print("Sum of numbers from 51 to 91:")
print(sum(51:91))

Q.2

import matplotlib.pyplot as plt
import numpy as np
from scipy import stats
x = np.array([1,2,3,4,5,6,7,8])
y = np.array([7,14,15,18,19,21,26,23])
slope, intercept, r, p, std_err = stats.linregress(x, y)
def myfunc(x):
 return slope * x + intercept
mymodel = list(map(myfunc, x))
plt.scatter(x, y)
plt.plot(x, mymodel)
plt.show()


#slip08


Q.1


Fibonacci <- numeric(10)
Fibonacci[1] <- Fibonacci[2] <- 1 
for (i in 3:10) Fibonacci[i] <- Fibonacci[i - 2] + Fibonacci[i - 1]
print("First 10 Fibonacci numbers:")
print(Fibonacci)

Q.2

import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

# Load the dataset
data = pd.read_csv('C:\\Users\\siddhii\\.spyder-py3\\CC GENERAL.csv')

# Fill missing values with mean of the respective columns
data.fillna(data.mean(), inplace=True)

# Select relevant features for clustering
X = data.iloc[:, 1:].values  # Exclude the 'CUST_ID' column for clustering

# Standardize the data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Apply K-means algorithm
kmeans = KMeans(n_clusters=5, init='k-means++', random_state=42)
kmeans.fit(X_scaled)

# Add cluster labels to the original dataset
data['Cluster'] = kmeans.labels_

# Print the count of customers in each cluster
print(data['Cluster'].value_counts())

# Visualization (considering only 2 dimensions for plotting)
plt.scatter(X_scaled[:, 0], X_scaled[:, 1], c=kmeans.labels_, cmap='viridis', marker='.')
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], s=300, c='red', label='Centroids')
plt.title('K-means Clustering')
plt.legend()
plt.show()


#slip09

Q.1

# Creating a data frame with details of 5 employees
employee_data <- data.frame(
  Employee_ID = c(1, 2, 3, 4, 5),
  Name = c("John", "Emma", "Michael", "Sophia", "William"),
  Age = c(30, 28, 35, 26, 32),
  Department = c("HR", "Finance", "IT", "Marketing", "Operations"),
  Salary = c(50000, 60000, 70000, 55000, 65000)
)

# Displaying the summary of the data frame
summary(employee_data)


Q.2

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score

# Load the breast cancer dataset
data = load_breast_cancer()
X = data.data
y = data.target

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Build an SVM model
svm_model = SVC(kernel='linear', random_state=42)
svm_model.fit(X_train, y_train)

# Make predictions on the test set
y_pred = svm_model.predict(X_test)

# Calculate accuracy, precision, and recall
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)

# Display evaluation metrics
print(f'Accuracy: {accuracy:.4f}')
print(f'Precision: {precision:.4f}')
print(f'Recall: {recall:.4f}')

                            OR

#Import scikit-learn dataset library
from sklearn import datasets
#Load dataset
cancer = datasets.load_breast_cancer()
# print the names of the 13 features
print("Features: ", cancer.feature_names)
# print the label type of cancer('malignant' 'benign')
print("Labels: ", cancer.target_names)
# print data(feature)shape
cancer.data.shape 
# print the cancer data features (top 5 records)
print(cancer.data[0:5])
# print the cancer labels (0:malignant, 1:benign)
print(cancer.target) 
# Import train_test_split function
from sklearn.model_selection import train_test_split
# Split dataset into training set and test set
X_train, X_test, y_train, y_test = train_test_split(cancer.data, cancer.target,
test_size=0.3,random_state=109) # 70% training and 30% test
#Import svm model
from sklearn import svm

#Create a svm Classifier
clf = svm.SVC(kernel='linear') # Linear Kernel
#Train the model using the training sets
clf.fit(X_train, y_train)
#Predict the response for test dataset
y_pred = clf.predict(X_test)
#Import scikit-learn metrics module for accuracy calculation
from sklearn import metrics
# Model Accuracy: how often is the classifier correct?
print("Accuracy:",metrics.accuracy_score(y_test, y_pred))



#slip10

Q.1

# Given vector
vector <- c(10, 5, 8, 15, 3, 20)

# Find maximum value
max_value <- max(vector)

# Find minimum value
min_value <- min(vector)

# Print the maximum and minimum values
cat("Maximum value:", max_value, "\n")
cat("Minimum value:", min_value, "\n")


Q.2

import pandas as pd
from itertools import combinations

# Read the Iris dataset
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data"
names = ['sepal-length', 'sepal-width', 'petal-length', 'petal-width', 'class']
data = pd.read_csv(url, names=names)

# For the sake of this example, we'll drop the 'class' column as Apriori works with itemsets
data.drop('class', axis=1, inplace=True)

# Convert numerical values to categorical (for the sake of this demonstration)
for col in data.columns:
    data[col] = pd.cut(data[col], bins=3, labels=['small', 'medium', 'large'])

# Define a function to generate frequent itemsets
def find_frequent_itemsets(data, min_support):
    itemsets = {}
    for column in data.columns:
        itemsets[column] = data[column].unique()

    all_frequent_itemsets = {}
    for i in range(1, len(itemsets.keys()) + 1):
        frequent_itemsets = {}
        combinations_set = list(combinations(itemsets.keys(), i))
        for comb in combinations_set:
            comb_data = data[list(comb)]
            support = (comb_data.apply(lambda x: all(x == comb), axis=1)).sum() / len(data)
            if support >= min_support:
                frequent_itemsets[comb] = support
        all_frequent_itemsets.update(frequent_itemsets)
    return all_frequent_itemsets

# Apply Apriori algorithm
min_support_threshold = 0.1
frequent_itemsets = find_frequent_itemsets(data, min_support_threshold)

# Display frequent itemsets
print("Frequent Itemsets:")
for itemset, support in frequent_itemsets.items():
    print(f"{itemset} - Support: {support:.4f}")

#slip 11

Q.1

l1 = list("x", "y", "z")
l2 = list("X", "Y", "Z", "x", "y", "z")
print("Original lists:")
print(l1) 
print(l2) 
print("All elements of l2 that are not in l1:")
setdiff(l2, l1)


Q.2

import matplotlib.pyplot as mtp
import pandas as pd
dataset = pd.read_csv('C:\\Users\\siddhii\\.spyder-py3\\Wholesale customers data.csv')
dataset
x = dataset.iloc[:, [3, 4]].values
print(x)
import scipy.cluster.hierarchy as shc
dendro = shc.dendrogram(shc.linkage(x, method="ward"))
mtp.title("Dendrogrma Plot")
mtp.ylabel("Euclidean Distances")
mtp.xlabel("Customers")
mtp.show()
from sklearn.cluster import AgglomerativeClustering
hc= AgglomerativeClustering(n_clusters=5, affinity='euclidean', linkage='ward')
y_pred= hc.fit_predict(x)
mtp.scatter(x[y_pred == 0, 0], x[y_pred == 0, 1], s = 100, c = 'blue', label = 'Cluster 1')
mtp.scatter(x[y_pred == 1, 0], x[y_pred == 1, 1], s = 100, c = 'green', label = 'Cluster 2')
mtp.scatter(x[y_pred== 2, 0], x[y_pred == 2, 1], s = 100, c = 'red', label = 'Cluster 3')
mtp.scatter(x[y_pred == 3, 0], x[y_pred == 3, 1], s = 100, c = 'cyan', label = 'Cluster 4')
mtp.scatter(x[y_pred == 4, 0], x[y_pred == 4, 1], s = 100, c = 'magenta', label = 'Cluster 5')
mtp.title('Clusters of customers')
mtp.xlabel('Milk')
mtp.ylabel('Grocery')
mtp.legend()
mtp.show()


#slip 12

Q.1

# Create a data frame for 5 employees
employees <- data.frame(
  empno = c(1, 2, 3, 4, 5),
  empname = c("John Doe", "Emma Smith", "Michael Johnson", "Sophia Williams", "William Brown"),
  gender = c("M", "F", "M", "F", "M"),
  age = c(30, 28, 35, 26, 32),
  designation = c("Manager", "Analyst", "Engineer", "Marketing Specialist", "Supervisor")
)

# Display details of the employees data frame
print(employees)


Q.2

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt

# Load the dataset
data = pd.read_csv('C:\\Users\\siddhii\\.spyder-py3\\Cars.csv')

# Display the first few rows of the dataset
print(data.head())

# Selecting features and target variable
X = data[['Weight', 'Volume']]
y = data['CO2']

# Splitting the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Creating a Linear Regression model
model = LinearRegression()

# Training the model
model.fit(X_train, y_train)

# Making predictions
y_pred = model.predict(X_test)

# Model evaluation
print('Coefficients:', model.coef_)
print('Intercept:', model.intercept_)
print('Mean Squared Error (MSE): %.2f' % mean_squared_error(y_test, y_pred))
print('Coefficient of Determination (R^2): %.2f' % r2_score(y_test, y_pred))

# Plotting predicted vs actual values
plt.scatter(y_test, y_pred)
plt.xlabel('Actual CO2 Emissions')
plt.ylabel('Predicted CO2 Emissions')
plt.title('Actual vs Predicted CO2 Emissions')
plt.show()


#slip13

Q.1

digits <- c(7,2,6,3,4,8)
Frequency <- c(1,2,3,4,5,6)
# Plot the chart.
pie(digits, Frequency)

Q.2

import pandas as pd

# Read the dataset
file_path = "C:\\Users\\siddhii\\.spyder-py3\\StudentsPerformance.csv"  # Update with your file path
data = pd.read_csv(file_path)

# Display the shape of the dataset
print("Shape of the dataset:", data.shape)

# Display the top rows of the dataset with their columns
print("\nTop rows of the dataset:")
print(data.head())

#slip 14

Q.1


# Create a list of employee names
employee_list <- list("John", "Emma", "Michael", "Sophia", "William")

# a. Display names of employees in the list
cat("Names of employees in the list:\n")
print(employee_list)

# b. Add an employee at the end of the list
new_employee <- "Olivia"
employee_list <- c(employee_list, new_employee)

cat("\nAfter adding an employee at the end of the list:\n")
print(employee_list)

# c. Remove the third element of the list
employee_list <- employee_list[-3]

cat("\nAfter removing the third element from the list:\n")
print(employee_list)


Q.2

import pandas as pd
from apyori import apriori

# Load the Groceries dataset
store_data = pd.read_csv('C:\\Users\\siddhii\\.spyder-py3\\Groceries_dataset.csv')

# Display the first few rows of the dataset
print(store_data.head())

# Convert the dataset into a list of lists
records = []
for i in range(len(store_data)):
    records.append([str(store_data.values[i, j]) for j in range(len(store_data.columns))])

# Apply Apriori algorithm
association_rules = apriori(records, min_support=0.0045, min_confidence=0.2, min_lift=3, max_length=None)
association_results = list(association_rules)

# Extract and print association rules along with support and confidence
for item in association_results:
    pair = item[0]
    items = [x for x in pair]
    
    print("Rule: " + ', '.join(items))
    print("Support: " + str(item[1]))
    print("Confidence: " + str(item[2][0][2]))
    print("=====================================")


#slip15

Q.1

# Create two vectors of integer type (minimum length 4)
vector1 <- c(4, 7, 2, 9)
vector2 <- c(3, 5, 8, 2)

# Addition of two vectors
addition_result <- vector1 + vector2
cat("Addition Result:", addition_result, "\n")

# Multiplication of two vectors
multiplication_result <- vector1 * vector2
cat("Multiplication Result:", multiplication_result, "\n")

# Division of two vectors
division_result <- vector1 / vector2
cat("Division Result:", division_result, "\n")

Q.2

import pandas 
#from sklearn import tree
#from sklearn.tree import DecisionTreeClassifier
#import matplotlib.pyplot as plt
df = pandas.read_csv('C:\\Users\\siddhii\\.spyder-py3\\Shows.csv')
d = {'UK': 0, 'USA': 1, 'N': 2}
df['Nationality'] = df['Nationality'].map(d)
d = {'YES': 1, 'NO': 0}
df['Go'] = df['Go'].map(d)
features = ['Age', 'Experience', 'Rank', 'Nationality']
X = df[features]
y = df['Go']
print(X) 
print(y)

#slip16

Q.1


# Create a data frame with the given data
data <- data.frame(
  Year = c(2001, 2002, 2003),
  Export = c(26, 32, 35),
  Import = c(35, 40, 50)
)

# Plotting a bar plot
barplot(
  height = t(data[, -1]),  # Exclude 'Year' column and transpose for barplot
  beside = TRUE,           # Place bars beside each other
  names.arg = data$Year,  # Year on x-axis
  col = c("blue", "red"),  # Colors for Export and Import bars
  xlab = "Year",           # X-axis label
  ylab = "Value",          # Y-axis label
  main = "Export and Import Data"  # Title of the plot
)
legend("topright", legend = c("Export", "Import")


Q.2

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report

# Load the dataset
data = pd.read_csv('C:\\Users\\siddhii\\.spyder-py3\\diabetes.csv')

# Display the first few rows of the dataset
print("First few rows of the dataset:")
print(data.head())

# Features and target variable
X = data.drop('Outcome', axis=1)
y = data['Outcome']

# Splitting the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Build Decision Tree Classifier
clf = DecisionTreeClassifier(random_state=42)
clf.fit(X_train, y_train)

# Make predictions on the test set
y_pred = clf.predict(X_test)

# Evaluate model
print("\nAccuracy Score:", accuracy_score(y_test, y_pred))
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

#slip17

Q.1


# Function to generate the first 'n' Fibonacci numbers
generate_fibonacci <- function(n) {
  if (n <= 0) {
    return(NULL)
  } else if (n == 1) {
    return(c(0))
  } else if (n == 2) {
    return(c(0, 1))
  } else {
    fibonacci <- numeric(n)
    fibonacci[1] <- 0
    fibonacci[2] <- 1
    for (i in 3:n) {
      fibonacci[i] <- fibonacci[i - 1] + fibonacci[i - 2]
    }
    return(fibonacci)
  }
}

# Get the first 20 Fibonacci numbers
first_20_fibonacci <- generate_fibonacci(20)

# Print the first 20 Fibonacci numbers
cat("The first 20 Fibonacci numbers are:", first_20_fibonacci, "\n")



Q.2



import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

# Create the Stock_Market dictionary
Stock_Market = {
    'Year': [2017]*12 + [2016]*12,
    'Month': list(range(12, 0, -1))*2,
    'Interest_Rate': [2.75,2.5,2.5,2.5,2.5,2.5,2.5,2.25,2.25,2.25,2,2,2,1.75,1.75,1.75,1.75,1.75,1.75,1.75,1.75,1.75,1.75,1.75],
    'Unemployment_Rate': [5.3,5.3,5.3,5.3,5.4,5.6,5.5,5.5,5.5,5.6,5.7,5.9,6,5.9,5.8,6.1,6.2,6.1,6.1,6.1,5.9,6.2,6.2,6.1],
    'Stock_Index_Price': [1464,1394,1357,1293,1256,1254,1234,1195,1159,1167,1130,1075,1047,965,943,958,971,949,884,866,876,822,704,719]
}

# Create a DataFrame from the Stock_Market dictionary
df = pd.DataFrame(Stock_Market)

# Selecting features and target variable
X = df[['Interest_Rate', 'Unemployment_Rate']]
y = df['Stock_Index_Price']

# Create and fit the regression model
model = LinearRegression()
model.fit(X, y)

# Plotting Stock_Index_Price vs. Interest_Rate
plt.scatter(df['Interest_Rate'], df['Stock_Index_Price'], color='blue')
plt.plot(df['Interest_Rate'], model.predict(X), color='red', linewidth=2)
plt.title('Stock Market Price vs Interest Rate')
plt.xlabel('Interest Rate')
plt.ylabel('Stock Market Price')
plt.show()



#slip18

Q.1

# Given vector
given_vector <- c(3, 8, 12, 5, 9, 15, 2, 10)

# Find maximum value
max_value <- max(given_vector)
cat("Maximum value:", max_value, "\n")

# Find minimum value
min_value <- min(given_vector)
cat("Minimum value:", min_value, "\n")


Q.2

import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Given data
x = np.array([1, 2, 3, 4, 5, 6, 7, 8]).reshape(-1, 1)  # Reshape to 2D array
y = np.array([7, 14, 15, 18, 19, 21, 26, 23])

# Initialize and fit the linear regression model
model = LinearRegression()
model.fit(x, y)

# Estimated coefficients
b0 = model.intercept_
b1 = model.coef_[0]

print("Estimated coefficient b0 (intercept):", b0)
print("Estimated coefficient b1 (slope):", b1)

# Make predictions
y_pred = model.predict(x)

# Model evaluation
mse = mean_squared_error(y, y_pred)
r2 = r2_score(y, y_pred)

print("\nMean Squared Error (MSE):", mse)
print("Coefficient of Determination (R^2 Score):", r2)


#slip19

Q.1


# Creating a data frame with details of 5 students
students_df <- data.frame(
  Rollno = c(101, 102, 103, 104, 105),
  Studname = c("Alice", "Bob", "Charlie", "David", "Eva"),
  Address = c("Address1", "Address2", "Address3", "Address4", "Address5"),
  Marks = c(85, 76, 92, 81, 89)
)

# Displaying the details of the students
print(students_df)


Q.2

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt

# Load the dataset
data = pd.read_csv('C:\\Users\\siddhii\\.spyder-py3\\Cars.csv')

# Display the first few rows of the dataset
print(data.head())

# Selecting features and target variable
X = data[['Weight', 'Volume']]
y = data['CO2']

# Splitting the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Creating a Linear Regression model
model = LinearRegression()

# Training the model
model.fit(X_train, y_train)

# Making predictions
y_pred = model.predict(X_test)

# Model evaluation
print('Coefficients:', model.coef_)
print('Intercept:', model.intercept_)
print('Mean Squared Error (MSE): %.2f' % mean_squared_error(y_test, y_pred))
print('Coefficient of Determination (R^2): %.2f' % r2_score(y_test, y_pred))

# Plotting predicted vs actual values
plt.scatter(y_test, y_pred)
plt.xlabel('Actual CO2 Emissions')
plt.ylabel('Predicted CO2 Emissions')
plt.title('Actual vs Predicted CO2 Emissions')
plt.show()


#slip20
Q.1


# Given vectors
vector1 <- c(1, 2, 3, 4, 5)
vector2 <- c("A", "B", "C", "D", "E")
vector3 <- c(10.5, 20.2, 15.8, 18.6, 12.3)
vector4 <- c(TRUE, FALSE, TRUE, FALSE, TRUE)

# Create a data frame from the given vectors
data_frame <- data.frame(Column1 = vector1, Column2 = vector2, Column3 = vector3, Column4 = vector4)

# Display the created data frame
print(data_frame)


Q.2

#import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
# Importing the dataset
dataset = pd.read_csv('C:\\Users\\siddhii\\.spyder-py3\\Ecommerce Customers.csv')
X = dataset.iloc[:, [3, 4]].values
# y = dataset.iloc[:, 3].values
# Splitting the dataset into the Training set and Test set

"""from sklearn.cross_validation import train_test_split 
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)"""
# Feature Scaling
"""from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test) 
sc_y = StandardScaler()
y_train = sc_y.fit_transform(y_train)"""
# Using the dendrogram to find the optimal number of clusters
import scipy.cluster.hierarchy as sch
dendrogram = sch.dendrogram(sch.linkage(X, method = 'ward'))
plt.title('Dendrogram')
plt.xlabel('Customers')
plt.ylabel('Euclidean distances')
plt.show()
# Fitting Hierarchical Clustering to the dataset
from sklearn.cluster import AgglomerativeClustering
hc = AgglomerativeClustering(n_clusters = 5, affinity = 'euclidean', linkage = 'ward')
y_hc = hc.fit_predict(X)
# Visualising the clusters
plt.scatter(X[y_hc == 0, 0], X[y_hc == 0, 1], s = 100, c = 'red', label = 'Cluster 1')
plt.scatter(X[y_hc == 1, 0], X[y_hc == 1, 1], s = 100, c = 'blue', label = 'Cluster 2')
plt.scatter(X[y_hc == 2, 0], X[y_hc == 2, 1], s = 100, c = 'green', label = 'Cluster 3')
plt.scatter(X[y_hc == 3, 0], X[y_hc == 3, 1], s = 100, c = 'cyan', label = 'Cluster 4')

plt.scatter(X[y_hc == 4, 0], X[y_hc == 4, 1], s = 100, c = 'magenta', label = 'Cluster 5')
plt.title('Clusters of customers')
plt.xlabel('Annual Income (k$)')
plt.ylabel('Spending Score (1-100)')
plt.legend()
plt.show()

*/







