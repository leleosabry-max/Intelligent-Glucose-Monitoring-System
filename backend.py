from xml.parsers.expat import model
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import DBSCAN 
import skfuzzy as fuzz
from sklearn.svm import SVC
import shap
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, accuracy_score
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.metrics import silhouette_score
import sqlite3
from sklearn.metrics.pairwise import cosine_similarity

conn = sqlite3.connect("glucose.db")
cursor = conn.cursor()

cursor.execute("""
CREATE TABLE IF NOT EXISTS results (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    score REAL,
    condition TEXT,
    decision TEXT
)
""") 
conn.commit()
MEDICAL_DB = {
    "LOW GLUCOSE": {
        "range": "< 70 mg/dL",
        "meaning": "Hypoglycemia risk",
        "action": "Eat fast-acting carbohydrates immediately",
        "severity": "HIGH"
    },
    "NORMAL": {
        "range": "70–180 mg/dL",
        "meaning": "Stable glucose level",
        "action": "Maintain current routine",
        "severity": "LOW"
    },
    "HIGH GLUCOSE": {
        "range": "> 180 mg/dL",
        "meaning": "Hyperglycemia risk",
        "action": "Seek medical advice / adjust insulin",
        "severity": "HIGH"
    }
}

history = []


def store_case(score, condition, decision):
    history.append({
        "score": score,
        "condition": condition,
        "decision": decision
    })

def retrieve_similar(current_score, k=3):
    if not history:
        return []

    sorted_cases = sorted(
        history,
        key=lambda x: abs(x["score"] - current_score)
    )

    return sorted_cases[:k]

#read the datasets

df1=pd.read_csv("diabetes.csv")
df2=pd.read_csv("GlucoBench_benchmark_dataset.csv")
#cleaning dataset1
duplicatesIn1=df1[df1.duplicated(keep=False)]
print(len(duplicatesIn1))   # no duplicates in df1

missingValues1=df1[df1.isnull().any(axis=1)]  
print   (len(missingValues1))   # no missing values in df1


cols=df1.columns
for col in cols:
    df1[col]=df1[col].replace(0,np.nan)  # replace 0 with NaN for certain columns
df1=df1.fillna(df1.mean())  # fill NaN values with the mean of each column
X1=df1.drop(columns=["Outcome"])
scaler=StandardScaler()
X1_scaled=scaler.fit_transform(X1)  #scale the features in df1

print(df1.describe())
print(df1.shape)

print (df2.describe())
df2=df2[["timestamp","glucose"]]  

df2["timestamp"]=pd.to_datetime(df2["timestamp"])  # convert timestamp to datetime format
df2=df2.sort_values("timestamp")  # sort the dataframe by timestamp
df2=df2.set_index("timestamp")  # set timestamp as the index
df2 = df2.resample("5min").mean()
df2["glucose"] = df2["glucose"].interpolate()
print(df2.isnull().sum())  # check for missing values in df2
print(df2.head())

def run_DBSCAN(X1_scaled):

    model=DBSCAN(eps=0.5,min_samples=5)       # fit the DBSCAN model to the data
    labels=model.fit_predict(X1_scaled)

    return labels

def run_fcm(X1_scaled, n_clusters=3):

    cntr, u, _, _, _, _, _=fuzz.cluster.cmeans(  # fit the FCM model to the data
        X1_scaled.T,
        c=n_clusters,
        m=2,
        error=0.005,
        maxiter=1000
   )

    labels=np.argmax(u, axis=0)
                                    # assign each data point to the cluster with the highest membership value
    return labels


def create_labels(X1):
    dbscan_labels=run_DBSCAN(X1)
    fcm_labels=run_fcm(X1)           # create labels using both DBSCAN and FCM and return them as a combined label set
    
    dbscan_labels=np.where(dbscan_labels==-1,
                           np.max(dbscan_labels)+1, dbscan_labels)  # replace DBSCAN noise points with a new cluster label
    combined_pairs = [f"{f}_{d}" for f, d in zip(fcm_labels, dbscan_labels)]
    
    le=LabelEncoder()
    return le.fit_transform(combined_pairs) # combine the DBSCAN and FCM labels by adding them together   
    

def train_svm(X1_scaled, Y):
    svm_model=SVC(kernel="rbf", C=1,probability=True)  # create an SVM model with RBF kernel and fit it to the data using the combined labels as the target variable
    svm_model.fit(X1_scaled, Y)
    return svm_model

def predict(model,X1_scaled):
    predictions=model.predict(X1_scaled)     # make predictions using the trained SVM model and return both the predicted labels and the probabilities for each class
    probabilities=model.predict_proba(X1_scaled)
    return predictions, probabilities

def explain(model, X_scaled):
  background = shap.kmeans(X_scaled, 10) 
  explainer = shap.KernelExplainer(model.predict_proba, background)
    
    # Explain a small subset
  shap_values = explainer.shap_values(X_scaled[:10]) 
    
    # Calculate global mean importance as a single float
  abs_shap = np.abs(np.array(shap_values)) 
  importance_value = np.mean(abs_shap) 
  return importance_value

def detect_drift(forecast_values, threshold=5):
    volatility=np.std(forecast_values)  # calculate the standard deviation of the forecasted values to measure volatility
    trend=np.abs(forecast_values[-1] - forecast_values[0])  # calculate the absolute difference between the last and first forecasted values to measure trend
    return volatility > threshold or trend > threshold  # return True if either the volatility or trend exceeds the specified threshold, indicating potential drift in glucose levels

def forecast(series):
    model=SARIMAX (series,order=(1,1,1))
    results=model.fit()

    return results.forecast(steps=5).to_numpy() # fit a SARIMA model to the glucose time series data and forecast the next 5 values, returning them as a numpy array   
series=df2["glucose"]
forecast_values=forecast(series)  
drift_flag = detect_drift(forecast_values)
print("Drift flag:", drift_flag)
print("Forecasted glucose values for the next 5 time steps:", forecast_values)

volatility = np.std(forecast_values)
trend = abs(forecast_values[-1] - forecast_values[0])

forecast_risk = volatility + 0.5 * trend  # calculate a continuous risk score based on the volatility and trend of the forecasted glucose values, giving more weight to the trend by multiplying it by 0.5

labels=create_labels(X1_scaled)  # create combined labels using DBSCAN and FCM clustering on the scaled features
svm_model=train_svm(X1_scaled, labels)  # train the SVM model using the scaled features and the combined labels
predictions, probabilities=predict(svm_model, X1_scaled)

if drift_flag:
    print("⚠️ Drift detected — retraining model")
    labels=create_labels(X1_scaled)  # create combined labels using DBSCAN and FCM clustering on the scaled features
    svm_model=train_svm(X1_scaled, labels)  # retrain the SVM

importance_value = explain(svm_model, X1_scaled)  


svm_confidence=np.max(probabilities, axis=1)  # calculate the confidence of the SVM predictions by taking the maximum probability for each prediction
total_score=svm_confidence * importance_value * (1+forecast_risk)  # combine the SVM confidence score and the forecast risk score to calculate a total risk score for each data point

print("svm_confidence shape:", np.shape(svm_confidence))
print("importance shape:", np.shape(importance_value))
print("forecast_risk type:", type(forecast_risk))
print("forecast_risk value:", forecast_risk)
print("svm:", svm_confidence.shape)
print("importance:", np.shape(importance_value))
print("risk:", np.shape(forecast_risk))
print("Min:", np.min(total_score))
print("Max:", np.max(total_score))
print("Mean:", np.mean(total_score))
print("Std:", np.std(total_score))

mean_score=np.mean(total_score)
std_score=np.std(total_score)
min_threshold=mean_score
max_threshold=mean_score + std_score

def make_decision(individual_score, mean_ref, std_ref):
    if individual_score > (mean_ref + std_ref):
        return "HIGH RISK"
    elif individual_score > mean_ref:
        return "MEDIUM RISK"
    else:
        return "LOW RISK"

decision=make_decision(total_score[-1], mean_score, std_score)  # make a decision based on the total risk score of the last data point, using the mean and standard deviation of the scores as reference points for determining risk levels

print("average score:",mean_score)
print("decision:", decision)
def detect_glucose_condition(forecast_values):
    if np.min(forecast_values) < 70:
        return "LOW GLUCOSE"
    elif np.max(forecast_values) > 180:
        return "HIGH GLUCOSE"
    else:
        return "NORMAL"

def save_to_db(score, condition, decision):
    cursor.execute(
        "INSERT INTO results (score, condition, decision) VALUES (?, ?, ?)",
        (float(score), condition, decision)
    )
    conn.commit()

def get_recent_cases(limit=5):
    cursor.execute(
        "SELECT score, condition, decision FROM results ORDER BY id DESC LIMIT ?",
        (limit,)
    )
    return cursor.fetchall()

def retrieve_rag(condition):
    return MEDICAL_DB.get(condition, {
        "range": "Unknown",
        "meaning": "No medical info available",
        "action": "Consult a healthcare professional",
        "severity": "UNKNOWN"
    })

def find_similar_cases(X_scaled, current_index, top_k=3):
    current_vector = X_scaled[current_index].reshape(1, -1)

    similarities = cosine_similarity(current_vector, X_scaled)[0]

    # Get indices of most similar points (excluding itself)
    similar_indices = np.argsort(similarities)[::-1][1:top_k+1]

    return similar_indices, similarities[similar_indices]
condition = detect_glucose_condition(forecast_values)
current_index = len(X1_scaled) - 1  # last data point
similar_idx, sim_scores = find_similar_cases(X1_scaled, current_index)


def action_router(condition, decision, score, drift_flag, similar_cases):

    
    if drift_flag:
        return {
            "action": "RETRAIN",
            "message": "Model updated due to detected drift",
            "priority": "HIGH"
        }

    
    if decision == "HIGH RISK":
        return {
            "action": "EMERGENCY",
            "message": "Immediate medical attention required",
            "priority": "HIGH"
        }

    
    if similar_cases:
        avg_sim_score = np.mean([c["score"] for c in similar_cases])

        if avg_sim_score > score:
            return {
                "action": "ESCALATE",
                "message": "Similar past cases were worse — be cautious",
                "priority": "MEDIUM"
            }

   
    if condition == "HIGH GLUCOSE":
        return {
            "action": "CONTROL",
            "message": "Adjust insulin / consult doctor",
            "priority": "MEDIUM"
        }

    if condition == "LOW GLUCOSE":
        return {
            "action": "INTAKE",
            "message": "Consume fast sugar immediately",
            "priority": "HIGH"
        }

    return {
        "action": "MONITOR",
        "message": "Continue monitoring",
        "priority": "LOW"
    }

current_score = float(total_score[-1])
store_case(current_score, condition, decision)
save_to_db(current_score, condition, decision)
similar_cases = retrieve_similar(current_score)
recent_cases = get_recent_cases()

action = action_router(
    condition,
    decision,
    total_score[-1],
    drift_flag,
    similar_cases
)

rag_output = retrieve_rag(condition)
rag_output["similar_cases"] = similar_cases
rag_output["recent_cases"] = recent_cases

current_score = float(total_score[-1])

action = action_router(
    condition,
    decision,
    current_score,
    drift_flag,
    similar_cases
)

similar_cases = retrieve_similar(current_score)
recent_cases = get_recent_cases()


X_train, X_test, y_train, y_test = train_test_split(
    X1_scaled, labels, test_size=0.2, random_state=42
)

model = train_svm(X_train, y_train)

predictions = model.predict(X_test)
accuracy = accuracy_score(y_test, predictions)
report = classification_report(y_test, predictions)

def evaluate_forecast(actual, predicted):
    mae = mean_absolute_error(actual, predicted)
    rmse = np.sqrt(mean_squared_error(actual, predicted))
    return mae, rmse

actual_values = series[-5:].to_numpy() 
mae, rmse = evaluate_forecast(actual_values, forecast_values)

s_score = silhouette_score(X1_scaled, labels)

print("\n--- COSINE SIMILAR CASES ---")
for i, idx in enumerate(similar_idx):
    print(f"Case {idx} | Similarity: {sim_scores[i]:.3f}")

