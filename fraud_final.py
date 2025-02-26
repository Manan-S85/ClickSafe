import pandas as pd
import numpy as np
import re
import nltk
import pickle
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import Input, Embedding, LSTM, Dense, Dropout, SpatialDropout1D, Concatenate, Flatten
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, classification_report

# Download stopwords
nltk.download('stopwords')
from nltk.corpus import stopwords

# Load dataset
df = pd.read_csv("fraud_predictions.csv")
df.dropna(inplace=True)

# Convert target column to binary
df['Fraud_Prediction'] = df['Fraud_Prediction'].astype(int)

# Text cleaning function
def clean_text(text):
    text = str(text).lower()  # Convert to lowercase
    text = re.sub(r'\W', ' ', text)  # Remove special characters
    text = re.sub(r'\s+', ' ', text).strip()  # Remove extra spaces
    text = ' '.join([word for word in text.split() if word not in stopwords.words('english')])  # Remove stopwords
    return text

# Apply text cleaning
df['Translated_Review'] = df['Translated_Review'].apply(clean_text)

# Select text feature (LSTM Input)
X_text = df['Translated_Review']

# Select numerical features (DNN Input)
numerical_features = ["Rating", "Reviews", "Installs", "Price", "Sentiment_Polarity", "Sentiment_Subjectivity", "Reviews_to_Installs_Ratio", "Paid_App_High_Installs"]
df[numerical_features] = df[numerical_features].apply(pd.to_numeric, errors='coerce')

# Encode categorical features
categorical_features = ["Category", "Type", "Content Rating", "Genres", "Android Ver"]
for col in categorical_features:
    df[col] = LabelEncoder().fit_transform(df[col])

# Define Target
y = df["Fraud_Prediction"]

# Split dataset
X_text_train, X_text_test, X_num_train, X_num_test, y_train, y_test = train_test_split(
    X_text, df[numerical_features + categorical_features], y, test_size=0.2, random_state=42)

# Tokenization & Padding for Text Data
MAX_VOCAB_SIZE = 10000  
MAX_SEQUENCE_LENGTH = 100  

tokenizer = Tokenizer(num_words=MAX_VOCAB_SIZE, oov_token="<OOV>")
tokenizer.fit_on_texts(X_text_train)

X_text_train_seq = tokenizer.texts_to_sequences(X_text_train)
X_text_test_seq = tokenizer.texts_to_sequences(X_text_test)

X_text_train_pad = pad_sequences(X_text_train_seq, maxlen=MAX_SEQUENCE_LENGTH, padding='post')
X_text_test_pad = pad_sequences(X_text_test_seq, maxlen=MAX_SEQUENCE_LENGTH, padding='post')

# Scale numerical features
scaler = StandardScaler()
X_num_train_scaled = scaler.fit_transform(X_num_train)
X_num_test_scaled = scaler.transform(X_num_test)

# Save Tokenizer & Scaler
with open("fraud_tokenizer.pkl", "wb") as f:
    pickle.dump(tokenizer, f)

with open("fraud_scaler.pkl", "wb") as f:
    pickle.dump(scaler, f)

# Build Hybrid LSTM + DNN Model
EMBEDDING_DIM = 100

# LSTM Branch (Text)
text_input = Input(shape=(MAX_SEQUENCE_LENGTH,), name="Text_Input")
embedding = Embedding(input_dim=MAX_VOCAB_SIZE, output_dim=EMBEDDING_DIM, input_length=MAX_SEQUENCE_LENGTH)(text_input)
x = SpatialDropout1D(0.3)(embedding)
x = LSTM(128, return_sequences=True)(x)
x = Dropout(0.4)(x)
x = LSTM(64, return_sequences=True)(x)
x = Dropout(0.4)(x)
x = LSTM(32)(x)
x = Dropout(0.3)(x)
text_output = Dense(16, activation="relu")(x)

# DNN Branch (Numerical)
num_input = Input(shape=(X_num_train_scaled.shape[1],), name="Num_Input")
y = Dense(128, activation="relu")(num_input)
y = Dropout(0.3)(y)
y = Dense(64, activation="relu")(y)
y = Dropout(0.3)(y)

# Merge Both Branches
merged = Concatenate()([text_output, y])
final_output = Dense(1, activation="sigmoid")(merged)

# Define Model
model = Model(inputs=[text_input, num_input], outputs=final_output)

# Compile Model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Add EarlyStopping
early_stopping = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)

# Train Model
history = model.fit(
    [X_text_train_pad, X_num_train_scaled], y_train,
    epochs=20,  
    batch_size=32,
    validation_data=([X_text_test_pad, X_num_test_scaled], y_test),
    callbacks=[early_stopping]
)

# Save Model
model.save("fraud_app_hybrid_model.h5")

# Evaluate Model
y_pred = (model.predict([X_text_test_pad, X_num_test_scaled]) > 0.5).astype("int32")

accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.4f}")
print(classification_report(y_test, y_pred))

# Plot Training Accuracy
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.show()

# Load Model & Tokenizer for Later Use
model = load_model("fraud_app_hybrid_model.h5")

with open("fraud_tokenizer.pkl", "rb") as f:
    tokenizer = pickle.load(f)

with open("fraud_scaler.pkl", "rb") as f:
    scaler = pickle.load(f)

# Function to Predict Fraudulent App
def predict_fraud(review, num_features):
    cleaned_text = clean_text(review)
    seq = tokenizer.texts_to_sequences([cleaned_text])
    padded = pad_sequences(seq, maxlen=MAX_SEQUENCE_LENGTH, padding='post')
    
    num_features_scaled = scaler.transform([num_features])  
    prediction = model.predict([padded, num_features_scaled])[0][0]
    
    return "Fraudulent App" if prediction > 0.5 else "Genuine App"

# Example Prediction
test_review = "This app is a scam, it stole my money!"
test_numerical_features = [4.5, 100000, 5000000, 0.0, 0.2, 0.5, 0.05, 1]  # Example numerical values

print(f"Review: {test_review}")
print(f"Prediction: {predict_fraud(test_review, test_numerical_features)}")
