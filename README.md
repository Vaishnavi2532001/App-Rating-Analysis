📱 App Rating Prediction

 🧠 Objective

Build a model to predict an app's rating based on various characteristics provided in the dataset

## 📝 Problem Statement

Google Play Store is launching a new feature to boost visibility of promising apps in areas like:
- Recommendations (“Similar apps”, “You might also like”)
- Search result rankings

The goal is to predict which apps are likely to receive **high user ratings**, helping Google promote the right ones.

---

## 📁 Dataset Details

- Source: Google Play Store
- File: `googleplaystore.csv`

### 🔑 Features in the Dataset:
- `App`: App name
- `Category`: App category (e.g., Tools, Games)
- `Rating`: User rating (target variable)
- `Reviews`: Number of user reviews
- `Size`: App size (in KB/MB)
- `Installs`: Number of installs
- `Type`: Free or Paid
- `Price`: Price of the app
- `Content Rating`: Age group target
- `Genres`: App genres (e.g., Action, Puzzle)
- `Last Updated`, `Current Ver`, `Android Ver`: App version details

---

## 🔧 Steps to Perform

### 1. Load and Clean the Data
- Handle null values
- Convert formats:
  - `Size`: Convert MB to KB
  - `Reviews`: Convert to numeric
  - `Installs`: Remove '+' and ',' → convert to int
  - `Price`: Remove `$` and convert to float

### 2. Sanity Checks
- Ratings should be between **1 and 5**
- `Reviews ≤ Installs`
- Free apps should have price = 0

### 3. Univariate Analysis
- 📦 Boxplot for `Price` and `Reviews`
- 📊 Histogram for `Rating` and `Size`

### 4. Outlier Treatment
- Drop apps with price > $200
- Remove apps with reviews > 2 million
- Drop apps with extremely high installs (based on percentiles)

### 5. Bivariate Analysis
- 📈 `Rating vs Price` (scatterplot)
- 📈 `Rating vs Reviews`
- 📈 `Rating vs Size`
- 📦 `Rating vs Content Rating`
- 📦 `Rating vs Category`

### 6. Data Preprocessing
- Apply `log` transformation to `Reviews` and `Installs`
- Drop non-useful columns: `App`, `Last Updated`, `Current Ver`, `Android Ver`
- One-hot encode: `Category`, `Genres`, `Content Rating`

### 7. Train-Test Split
- Split dataset 70/30 into `df_train` and `df_test`

### 8. Create Modeling Sets
- `X_train`, `y_train`, `X_test`, `y_test`

### 9. Model Building
- Use **Linear Regression**
- Evaluate model with **R² on train set**

### 10. Make Predictions
- Predict app ratings on the test set

---

## ✅ Deliverables
- Cleaned dataset
- EDA visuals and outlier handling
- Trained model and performance metrics
- Summary of insights

---

## 🧑‍💻 Author

Vaishnavi Vispute  
📧 [your-email@example.com]  
🔗 [LinkedIn](https://linkedin.com/in/your-profile)

                                         
