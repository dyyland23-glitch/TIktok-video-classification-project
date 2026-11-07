
# TIktok-video-classification-project: Binary Logistic Regression & Content Moderation

This repository contains hands-on data science labs, exploratory analysis, and an end-to-end binary logistic regression workflow to predict whether TikTok users made a claim or gave an opinion during a video, utilizing the PACE (Plan, Analyze, Construct, Execute) framework.

---

## 📂 Repository Contents & File Breakdown

* **[`Video classification project.md`](https://github.com/dyyland23-glitch/TIktok-video-classification-project/blob/main/Video%20classification%20project.md)**: Comprehensive documentation detailing the structured data science lifecycle, model performance metrics, and business insights.
* **[`tiktok_dataset (1).csv`](https://github.com/dyyland23-glitch/TIktok-video-classification-project/blob/main/tiktok_dataset%20(1).csv)**: The primary dataset containing user video engagement characteristics, transcription text lengths, author ban statuses, and target claim/opinion labels.
* **[`OLS Summary table.png`](https://github.com/dyyland23-glitch/TIktok-video-classification-project/blob/main/OLS%20Summary%20table.png)**: Visual summary table displaying regression coefficients, standard errors, $z$-scores, and $p$-values.
* **[`the regression model.png`](https://github.com/dyyland23-glitch/TIktok-video-classification-project/blob/main/the%20regression%20model.png)**: Visual representation of the fitted regression decision boundary and model architecture.

---

## 🎯 Project Overview & Key Methods

* **Project Objective**: Build and evaluate a binary logistic regression model to classify video content as either a **claim** or an **opinion** (`claim_status`), helping prioritize user reports and streamline content moderation pipelines.
* **Key Tasks & Methods**:
  * **Exploratory Data Analysis (EDA)**: Investigated distributions of engagement metrics (such as `video_view_count`, `video_like_count`, `video_comment_count`, and `video_duration_sec`) and analyzed transcription lengths across account types.
  * **Class Imbalance Handling**: Addressed distributional skews in target classes using resampling and data balancing techniques.
  * **Feature Engineering & Multicollinearity**: Generated text-length features (`video_transcription_text` length) and evaluated feature relationships using correlation matrices to avoid violating logistic regression assumptions.
  * **Categorical Encoding**: Prepared categorical variables (such as `claim_status` and `author_ban_status`) using one-hot encoding.
  * **Model Training & Evaluation**: Fitted a `LogisticRegression` classifier, validated performance on held-out test data, and assessed accuracy using confusion matrices, classification reports (precision, recall, F1-score), and log-odds coefficient interpretations.

---

## 🛠️ Tech Stack & Key Libraries & Functions

* **Language**: Python 3.10+
* **Data Manipulation & Analysis**: `pandas`, `numpy`, `scipy`
  * **Key Functions**: `pd.read_csv()`, `df.isna()`, `df.drop_duplicates()`, `df.describe()`, `np.array()`
* **Statistical Modeling & Machine Learning**: `scikit-learn`, `statsmodels`
  * **Key Functions**: `LogisticRegression()`, `train_test_split()`, `StandardScaler()`, `OneHotEncoder()`
* **Model Evaluation & Metrics**: `scikit-learn` metrics
  * **Key Functions**: `confusion_matrix()`, `ConfusionMatrixDisplay()`, `classification_report()`, `r2_score()`
* **Data Visualization**: `matplotlib`, `seaborn`
  * **Key Functions**: `sns.boxplot()`, `sns.histplot()`, `sns.heatmap()`

---

## 🚀 Getting Started

### 1. Clone the Repository
```bash
git clone [https://github.com/dyyland23-glitch/TIktok-video-classification-project](https://github.com/dyyland23-glitch/TIktok-video-classification-project)
cd TIktok-video-classification-project
