# Assignment_fw
FW01

# Assignment

#### DataSet: 
    The Data-set considered is for creditcard fraud. A use case for classification, A binary classifier would predict the a transaction being fraudulent. Dataset Link: [Credit Card Fraud](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)
    From: Kaggle: 
    ``The dataset contains transactions made by credit cards in September 2013 by European cardholders.
This dataset presents transactions that occurred in two days, where we have 492 frauds out of 284,807 transactions. The dataset is highly unbalanced, the positive class (frauds) account for 0.172% of all transactions.

It contains only numerical input variables which are the result of a PCA transformation. Unfortunately, due to confidentiality issues, we cannot provide the original features and more background information about the data. Features V1, V2, … V28 are the principal components obtained with PCA, the only features which have not been transformed with PCA are 'Time' and 'Amount'. Feature 'Time' contains the seconds elapsed between each transaction and the first transaction in the dataset. The feature 'Amount' is the transaction Amount, this feature can be used for example-dependant cost-sensitive learning. Feature 'Class' is the response variable and it takes value 1 in case of fraud and 0 otherwise.``

##### Rationale for choosing the dataset: 
    The Data set denotes a use case where fraud detection is of utmost importance. Financial fraud detection is the need of the hour when more and more technology is being used and a lot of Technologically challenged consumers are starting to use these for the ease. This would in turn increase the likelihood of consumers being targeted for fraud. an Early detection of fraud would ensure that the crime can be stopped at the source. 

#### Machine Learning: 
    Spark ML's Logistic Regression is used for detecting cases with fraud. 
    Label/Class 0 - Denotes No Fraud and 
    Label/Class 1 - Denotes Fraud. 
    The Dataset is extremely skewed with most of the cases having class 0 as is the case in reality. 
    A random split of 70:30 is used for Test and Train Data creation. 

#### Streaming: 
    This is an excellent case for streaming analytics. a Credit card transaction data is streaming data in the real world. The ability to predict cases fo fraud near real time is essential. 
    the Partitioned data is stored in testData folder. 
    Streaming is achieved through Spark structured Streaming capability. The Test Data is partitioned into 10 partitions to achieve streaming data input and the output is streamed to the console. 


## How to Setup: 
    The setup requires Spark and Python to be installed and setup and is not in the purview of the asssignment. 
    1. use the requirement.txt to install the libraries required. 
    2. place the source data creditcard.csv in the working directory.
    3. Delete the testData folder if present in the working directory. 
