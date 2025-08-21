# Human Activity Recognition (HAR) Using Smartphone Data


## 1. Project Overview

This project focuses on building a machine learning model to classify human activities based on sensor data collected from a smartphone. The goal is to accurately identify whether a person is **Walking, Walking Upstairs, Walking Downstairs, Sitting, Standing, or Laying Down**.

This is a classic multi-class classification problem using time-series data. The project demonstrates a complete machine learning workflow from data acquisition to model evaluation.

---

## 2. Dataset

The dataset used is the **UCI HAR Dataset**, a public dataset collected from 30 volunteers performing the six activities mentioned above. The data was captured using a smartphone's embedded accelerometer and gyroscope.

*   **Source:** [UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/datasets/human+activity+recognition+using+smartphones)
*   **Characteristics:** The dataset is pre-processed, with features already engineered from the raw sensor signals (e.g., mean, standard deviation). This makes it ideal for focusing on the modeling aspect.

---

## 3. Methodology

The project followed these key steps:

1.  **Data Acquisition:** The dataset was downloaded and unzipped directly within the notebook environment.
2.  **Data Loading:** The training and test datasets were loaded into pandas DataFrames.
3.  **Exploratory Data Analysis (EDA):** The distribution of the different activities was visualized to ensure the dataset was balanced.
4.  **Model Training:** A `RandomForestClassifier` from Scikit-learn was chosen for its high performance and interpretability. The model was trained on the provided training data.
5.  **Model Evaluation:** The trained model was evaluated on the unseen test data to assess its real-world performance.

---

## 4. Results and Key Findings

The model achieved an overall **accuracy of 93%** on the test set.

The performance across different activities was excellent, as shown in the classification report:
```
                    precision    recall  f1-score   support

           WALKING       0.89      0.97      0.93       496
  WALKING_UPSTAIRS       0.88      0.89      0.89       471
WALKING_DOWNSTAIRS       0.97      0.86      0.91       420
           SITTING       0.91      0.90      0.90       491
          STANDING       0.91      0.92      0.91       532
            LAYING       1.00      1.00      1.00       537
          accuracy                           0.93      2947
         macro avg       0.93      0.92      0.92      2947
      weighted avg       0.93      0.93      0.93      2947
```

**Key Insights:**
*   The model is exceptionally good at identifying `LAYING` with 100% precision and recall.
*   It performs very well in distinguishing between static (`SITTING`, `STANDING`, `LAYING`) and dynamic activities (`WALKING` types).
*   The main area of confusion is between the different types of walking, especially `WALKING_DOWNSTAIRS`, which had the lowest recall (86%). This is an expected challenge due to the similarity in sensor patterns.

The confusion matrix below provides a visual representation of the model's predictions versus the actual labels:

<img width="940" height="852" alt="image" src="https://github.com/user-attachments/assets/144de643-d32e-4e41-8d92-76ae5196f075" />

---

## 5. Technologies Used

*   **Python 3**
*   **Pandas:** For data manipulation and loading.
*   **NumPy:** For numerical operations.
*   **Matplotlib & Seaborn:** For data visualization.
*   **Scikit-learn:** For machine learning model training and evaluation.
*   **Jupyter Notebook / Google Colab:** For the development environment.

---

## 6. How to Run This Project

1.  Clone this repository or download the source code.
2.  Open the `HAR_Analysis.ipynb` notebook in Google Colab or a local Jupyter environment.
3.  The notebook includes the code to download the dataset automatically.
4.  Run the cells in the notebook sequentially.
