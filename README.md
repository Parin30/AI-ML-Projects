# AI & ML Projects

This repository contains a collection of my AI and ML projects that I have completed during my studies. Each project is represented by an IPython Notebook file (.ipynb), showcasing different techniques and algorithms.

## Project 1 : Supervised Learning

The "Supervised Learning Project" demonstrates my skills in applying supervised learning algorithms to solve a specific problem. The notebook includes data preprocessing, feature engineering, model training, evaluation, and result analysis.

<h2 align="center"><strong>Part - A</strong></h2>

**DOMAIN**: Medical

**CONTEXT**: Develop an AI/ML model for medical research to predict patient conditions based on biomechanics features without revealing patient details due to confidentiality.

**DATA DESCRIPTION**: Biomechanics features of patients are represented by six attributes, including pelvic tilt, lumbar lordosis angle, sacral slope, pelvic incidence, pelvic radius, and spondylolisthesis degree.

**PROJECT OBJECTIVE**: Train Supervised Learning algorithms to predict patient conditions using the provided biomechanics data.

<h2 align="center"><strong>Part - B</strong></h2>

**DOMAIN**: Banking, Marketing

**CONTEXT**: Implement Machine Learning to improve marketing campaigns in a growing bank by predicting potential customers who will convert based on historical data.

**DATA DESCRIPTION**: The dataset includes customer attributes like age, customer since, highest spend, zip code, hidden score, monthly average spend, level, mortgage, security, fixed deposit account, internet banking, credit card, and loan on a card.

**PROJECT OBJECTIVE**: Build an ML model for targeted marketing to increase conversion ratio and expand the borrower base with the same budget.

**Skills & Tools Used:**
- Python
- pandas
- numpy
- matplotlib
- seaborn
- Jupyter Notebook
- scikit-learn
- Hyperparameter-tuning
- Grid search
- Randomized Search CV
- KNN
- SVM
- SMOTE

## Project 2: Ensemble Techniques

This Project showcases my skills on Ensemble Techniques, where I explored various methods like Random Forest, AdaBoost, Gradient Boosting, Grid Search. The project uses real-world data to predict customer churn behavior in a telecom company, demonstrating the effectiveness of ensemble methods in improving machine learning model performance.

**DOMAIN:** Telecommunications.

**CONTEXT:** A telecom company wants to use their historical customer data and leverage machine learning to predict behavior in an attempt to retain customers. The end goal is to develop focused customer retention programs.

**DATA DESCRIPTION:** The dataset is relevant for understanding customer churn behavior and predicting customer churn. Analyzing customer churn is important for businesses to identify factors that contribute to customer attrition and develop strategies to retain customers.

**PROJECT OBJECTIVE**: The objective, as a data scientist hired by the telecom company, is to build a model that will help to identify the potential customers who have a higher probability to churn. This will help the company to understand the pain points and patterns of customer churn and will increase the focus on strategising customer retention.

**Skills & Tools Used:**
- Python
- pandas
- numpy
- matplotlib
- seaborn
- Jupyter Notebook
- scikit-learn
- Hyperparameter-tuning
- Grid search
- RandomForestClassifier
- AdaBoostClassifier
- GradientBoostingClassifier
- DecisionTreeClassifier

## Project 3: Unsupervised Learning

Welcome to my Unsupervised Learning Project, where I explore and analyze a dataset containing information about various vehicle attributes. In this project, I perform data preprocessing, exploration, clustering, and dimensionality reduction techniques to gain insights from the data.

**DOMAIN:** Automobile.

**CONTEXT:** The data concerns city-cycle fuel consumption in miles per gallon to be predicted in terms of 3 multivalued discrete and 5 continuous attributes.

**DATA DESCRIPTION:** This automobile dataset is designed to predict city-cycle fuel consumption (mpg) based on eight attributes, including multivalued discrete variables like "cylinders," "model year," and "origin," as well as continuous variables such as "acceleration," "displacement," "horsepower," "weight," and the target variable "mpg." Each data point is uniquely identified by a "car name." It serves as a valuable resource for understanding the relationship between these attributes and city-cycle fuel efficiency in automobiles.

**PROJECT OBJECTIVE**: To understand K-means Clustering by applying on the Car Dataset to segment the cars into various categories.

### **Project Overview**

This project focuses on the following key aspects:

- **Data Exploration**: I start by loading and examining the dataset, understanding its structure, and performing basic exploratory data analysis (EDA) to get insights into the data's features and characteristics.
- **Data Cleaning and Preprocessing**: I handle missing values, identify outliers, and convert data types as necessary. I also perform data scaling to ensure uniformity across the dataset's numerical features.
- **Clustering Analysis**: Using K-means clustering, I identify patterns and groupings within the dataset based on the attributes of the vehicles. I determine the optimal number of clusters using techniques such as the elbow method and silhouette analysis.
- **Dimensionality Reduction**: I apply Principal Component Analysis (PCA) to reduce the dimensionality of the dataset while retaining relevant information. This helps visualize the data in a lower-dimensional space.
- **Model Training and Evaluation**: I train Support Vector Machine (SVM) models on both the original data and the reduced PCA components. I evaluate the models using classification metrics and analyze their performance.

**Skills & Tools Used:**
- Python
- pandas
- numpy
- Scipy
- matplotlib
- seaborn
- Jupyter Notebook
- scikit-learn
- Hyperparameter-tuning
- Grid search
- Support Vector Machines 
- KMeans Clustering
- Principle Component Analysis(PCA)

## Project 4: Featurization, Model Selection and Tuning

This Project showcases my skills on featurization, model selection, and tuning in the semiconductor manufacturing process domain. The project aims to predict the Pass/Fail yield of a specific production entity by analyzing a dataset containing 1567 data points and 591 features.

**DOMAIN:**  Semiconductor manufacturing process.

**DATA DESCRIPTION:** The data consists of 1567 datapoints each with 591 features. The dataset presented in this case represents a selection of such features where each example represents a single production entity with associated measured features and the labels represent a simple pass/fail yield for in house line testing. Target column “ –1” corresponds to a pass and “1” corresponds to a fail and the data time stamp is for that specific test point.

**PROJECT OBJECTIVE**: We will build a classifier to predict the Pass/Fail yield of a particular process entity and analyse whether all the features are required to build the model or not.

### **A. Data Exploration**

- Explored dataset structure and characteristics.
- Identified key data observations.
- Applied data cleansing, handling missing values, and dropping irrelevant features.

### **B. Data Analysis & Visualization**

- Conducted univariate analysis.
- Performed bivariate and multivariate analysis.
- Uncovered relationships and correlations between variables.

### **C. Data Pre-processing**

- Segregated predictors vs. target attributes.
- Addressed target class imbalance.
- Performed train-test split and standardization.

### **D. Model Training & Evaluation**

- Trained models using supervised learning.
- Employed cross-validation techniques.
- Applied hyperparameter tuning.
- Enhanced model performance with various techniques.

### **E. Post Training & Conclusion**

- Compared model performance and selected the best model.
- Pickled the selected model for future use.
- Summarized project results and conclusions.

## Project 5 : Neural Networks

Unveiling Neural Networks' Power: Predicting Signal Quality & Street View Housing Numbers. Harnessing the potential of Neural Networks and Deep Learning, this project dives into signal quality prediction for communication equipment while tackling the challenge of recognizing multi-digit street numbers in real-world images. Embrace the journey through feature analysis, model complexity trade-offs, and ethical implications in deploying cutting-edge technologies.

<h2 align="center"><strong>Part - A</strong></h2>

**DOMAIN**: Electronics and Telecommunication

**CONTEXT**: A communications equipment manufacturing company has a product which is responsible for emitting informative signals. Company wants to build a machine learning model which can help the company to predict the equipment’s signal quality using various parameters.

**DATA DESCRIPTION**: The data set contains information on various signal tests performed:

Parameters: Various measurable signal parameters.
Signal_Quality: Final signal strength or quality

**PROJECT OBJECTIVE**: To build a classifier which can use the given parameters to determine the signal strength or quality.

<h2 align="center"><strong>Part - B</strong></h2>

**DOMAIN**: Autonomous Vehicles

**CONTEXT**: A Recognising multi-digit numbers in photographs captured at street level is an important component of modern-day map making. A classic example of a corpus of such street-level photographs is Google’s Street View imagery composed of hundreds of millions of geo-located 360-degree panoramic images. The ability to automatically transcribe an address number from a geo-located patch of pixels and associate the transcribed number with a known street address helps pinpoint, with a high degree of accuracy, the location of the building it represents. More broadly, recognising numbers in photographs is a problem of interest to the optical character recognition community. While OCR on constrained domains like document processing is well studied, arbitrary multi-character text recognition in photographs is still highly challenging. This difficulty arises due to the wide variability in the visual appearance of text in the wild on account of a large range of fonts, colours, styles, orientations, and character arrangements. The recognition problem is further complicated by environmental factors such as lighting, shadows, specularity, and occlusions as well as by image acquisition factors such as resolution, motion, and focus blurs. In this project, we will use the dataset with images centred around a single digit (many of the images do contain some distractors at the sides). Although we are taking a sample of the data which is simpler, it is more complex than MNIST because of the distractors.

**DATA DESCRIPTION**: The SVHN is a real-world image dataset for developing machine learning and object recognition algorithms with the minimal requirement on data formatting but comes from a significantly harder, unsolved, real-world problem (recognising digits and numbers in natural scene images). SVHN is obtained from house numbers in Google Street View images. Where the labels for each of this image are the prominent number in that image i.e. 2,6,7 and 4 respectively. The dataset has been provided in the form of h5py files.

Acknowledgement: Yuval Netzer, Tao Wang, Adam Coates, Alessandro Bissacco, Bo Wu, Andrew Y. Ng Reading Digits in Natural Images with Unsupervised Feature Learning NIPS Workshop on Deep Learning and Unsupervised Feature Learning 2011. PDF http://ufldl.stanford.edu/housenumbers as the URL for this site.

**PROJECT OBJECTIVE**: To build a digit classifier on the SVHN (Street View Housing Number) dataset.

**Skills & Tools Used:**
- Python
- pandas
- numpy
- matplotlib
- seaborn
- Google Colab
- scikit-learn
- Tensorflow
- Keras
- h5py

## Contact

Feel free to connect with me on LinkedIn if you have any questions or would like to discuss my projects further:

https://www.linkedin.com/in/parin-parmar-0b9700200/

### PS: I will be working on various projects ahead and will keep updating the repository accordingly.
