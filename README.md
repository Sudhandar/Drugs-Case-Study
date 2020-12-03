# Case study : Persistency of drugs

One of the challenge for all Pharmaceutical companies is to understand the persistency of drug as per the physician prescription. 
With an objective to gather insights on the factors that are impacting the persistency, I have build a classification model predicting the persisitency of drugs for the given dataset. 

Link to view the notebook: [Github View](https://github.com/Sudhandar/Drugs-Case-Study/blob/master/notebooks/drugs_persistency_case_study.ipynb)

## Dataset Desription:

| Bucket                   | Variable                            | Variable Description                                                                                                                                                                     |
| ------------------------ | ----------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Unique Row Id            | Patient ID                          | Unique ID of each patient                                                                                                                                                                |
| Target Variable          | Persistency  Flag                   | Flag indicating if a patient was persistent or not                                                                                                                                       |
| Demographics             | Age                                 | Age of the patient during their therapy                                                                                                                                                  |
|                          | Race                                | Race of the patient from the patient table                                                                                                                                               |
|                          | Region                              | Region of the patient from the patient table                                                                                                                                             |
|                          | Ethnicity                           | Ethnicity of the patient from the patient table                                                                                                                                          |
|                          | Gender                              | Gender of the patient from the patient table                                                                                                                                             |
|                          | IDN Indicator                       | Flag indicating patients mapped to IDN                                                                                                                                                   |
| Provider Attributes      | NTM - Physician Specialty           | Specialty of the HCP that prescribed the NTM Rx                                                                                                                                          |
| Clinical Factors         | NTM - T-Score                       | T Score of the patient at the time of the NTM Rx (within 2 years prior from rxdate)                                                                                                      |
|                          | Change in T Score                   | Change in Tscore before starting with any therapy and after receiving therapy (Worsened, Remained Same, Improved, Unknown)                                                               |
|                          | NTM - Risk Segment                  | Risk Segment of the patient at the time of the NTM Rx (within 2 years days prior from rxdate)                                                                                            |
|                          | Change in Risk Segment              | Change in Risk Segment before starting with any therapy and after receiving therapy (Worsened, Remained Same, Improved, Unknown)                                                         |
|                          | NTM - Multiple Risk Factors         | Flag indicating if patient falls under multiple risk category (having more than 1 risk) at the time of the NTM Rx (within 365 days prior from rxdate)                                    |
|                          | NTM - Dexa Scan Frequency           | Number of DEXA scans taken prior to the first NTM Rx date (within 365 days prior from rxdate)                                                                                            |
|                          | NTM - Dexa Scan Recency             | Flag indicating the presence of Dexa Scan before the NTM Rx (within 2 years prior from rxdate or between their first Rx and Switched Rx; whichever is smaller and applicable)            |
|                          | Dexa During Therapy                 | Flag indicating if the patient had a Dexa Scan during their first continuous therapy                                                                                                     |
|                          | NTM - Fragility Fracture Recency    | Flag indicating if the patient had a recent fragility fracture (within 365 days prior from rxdate)                                                                                       |
|                          | Fragility Fracture During Therapy   | Flag indicating if the patient had fragility fracture during their first continuous therapy                                                                                              |
|                          | NTM - Glucocorticoid Recency        | Flag indicating usage of Glucocorticoids (>=7.5mg strength) in the one year look-back from the first NTM Rx                                                                              |
|                          | Glucocorticoid Usage During Therapy | Flag indicating if the patient had a Glucocorticoid usage during the first continuous therapy                                                                                            |
| Disease/Treatment Factor | NTM - Injectable Experience         | Flag indicating any injectable drug usage in the recent 12 months before the NTM OP Rx                                                                                                   |
|                          | NTM - Risk Factors                  | Risk Factors that the patient is falling into. For chronic Risk Factors complete lookback to be applied and for non-chronic Risk Factors, one year lookback from the date of first OP Rx |
|                          | NTM - Comorbidity                   | Comorbidities are divided into two main categories - Acute and chronic, based on the ICD codes.                                                                                          |
|                          | NTM - Concomitancy                  | Concomitant drugs recorded prior to starting with a therapy(within 365 days prior from first rxdate)                                                                                     |
|                          | Adherence                           | Adherence for the therapies                                                                                                                                                              |

## Data Preprocessing

The dataset was imbalanced with a greater number of non-persistency flags (65 % approx.). The 3 disease/risk factor buckets were present as individual binary columns (yes or no). There were 4 columns having large number of unknown values.
The following are the changes done to the dataset,

### Grouping of disease/risk factor columns:

The individual columns were grouped into their respective buckets, converted to 0’s and 1’s using encoding and found the count of values across each bucket (concomitancy count, risk factors count, comorbidity count).

### Dropping columns with unknown values:

The four columns (change risk segment, risk segment during rx, tscore bucket during rx, change tscore) with many unknown values were removed.

![alt text](https://github.com/Sudhandar/Drugs-Case-Study/blob/master/images/null_values.png)

### Feature Selection:

Mutual Information Classifier was used to rank features based on their relevancy scores and the top 7 features were considered.

| fetaure                    | score    |
| -------------------------- | -------- |
| concomitancy\_count        | 0.152989 |
| dexa\_freq\_during\_rx     | 0.150484 |
| dexa\_during\_rx           | 0.124688 |
| comorbidity\_count         | 0.067733 |
| gluco\_record\_during\_rx  | 0.034607 |
| ntm\_speciality            | 0.032738 |
| tscore\_bucket\_prior\_ntm | 0.018785 |

### Fixing Imbalanced dataset:

During initial iterations, the imbalanced nature of the dataset gave rise to large number of false negatives and poor f1 score. In order to fix that issue, SMOTE oversampling method was used to generate equal number of 0’s and 1’s in the target variable.

#### Before balancing

| Target | Count |
| ------ | ----- |
| 0      | 2135  |
| 1      | 1289  |

#### After Balancing

| Target | Count |
| ------ | ----- |
| 0      | 2135  |
| 1      | 2135  |

### Splitting the data

A 70-30 split for training and testing sets were used.

## Model Selection and Fine Tuning:

The data was trained separately on five different models and each model was evaluated using k-fold cross validation score and the Random Forest model was chosen since it has the highest mean ROC AUC value.

| Model                    | Train AUC |
| ------------------------ | --------- |
| Random Forest            | 0.885584  |
| SVC                      | 0.860227  |
| Logistic Regression      | 0.8585741 |
| K Neighbours Classifiers | 0.8548163 |
| Decision Tree            | 0.777741  |


### Finding the best parameters using Grid Search CV:

The best parameter values were selected using the Grid Search CV method.

| Parameters          | Value |
| ------------------- | ----- |
| n\_estimators       | 1600  |
| min\_samples\_split | 10    |
| min\_samples\_leaf  | 1     |
| max\_features       | sqrt  |
| max\_depth          | 20    |
| bootstrap           | TRUE  |

## Results:

After training the model using the above mentioned parameters, the following are the results obtained,

### ROC AUC Curve

[!alt text](https://github.com/Sudhandar/Drugs-Case-Study/blob/master/images/roc_auc.png)

The Random Forest model has an AUC score of **0.90** and f1 score of **0.83** on the test set.

