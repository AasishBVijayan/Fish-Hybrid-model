import tensorflow as tf
import joblib
import numpy as np
import pandas as pd  # <-- Make sure pandas is imported
from tensorflow.keras.preprocessing import image as keras_image
from tensorflow.keras.applications.resnet50 import preprocess_input as resnet50_preprocess

# --- 1. Define Image Class Labels (CRITICAL) ---
# List of 7 diseases from your FishImage.ipynb
CLASS_LABELS_IMAGE = [
    'Bacterial Red disease',
    'Bacterial diseases - Aeromoniasis',
    'Bacterial gill disease',
    'Fungal diseases Saprolegniasis',
    'Healthy Fish',
    'Parasitic diseases',
    'Viral diseases White tail disease'
]
# CLASS_LABELS_WATER is no longer needed, your model class handles it.

# --- 2. PASTE YOUR FULL 'class FishHealthModel:' DEFINITION HERE ---
# (I have pasted the code you just provided)

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
import joblib

class FishHealthModel:
    """
    A hybrid model that uses a trained RandomForestClassifier to predict overall
    health status and a rule-based engine to diagnose specific disease risks
    based solely on water quality parameters.
    """
    def __init__(self):
        self.model = RandomForestClassifier(n_estimators=150, random_state=42, class_weight='balanced')
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
        self.trained_features = []
        self.scaler_features = [] # To store names of scaled columns

    def train(self, excel_filepath):
        """
        Loads data from an Excel file, preprocesses it, balances the dataset,
        and trains the Random Forest model to predict the general 'Health Status'.
        """
        print("Starting model training...")
        try:
            df = pd.read_excel(excel_filepath)
        except FileNotFoundError:
            print(f"Error: The file '{excel_filepath}' was not found.")
            return
        except ImportError:
            print("Error: The 'openpyxl' package is required to read Excel files.")
            print("Please install it by running: pip install openpyxl")
            return

        # --- Columns to Drop (cleaned and leakage-free) ---
        DROP_COLUMNS = [
            'Datetime', 'Month', 'Month_Num', 'month_x', 'month_y',
            'Average Fish Weight (g)', 'Survival Rate (%)', 'Disease Occurrence (Cases)',
            'day', 'hour', 'oxigeno_scaled', 'ph', 'turbidez'
        ]

        df_clean = df.drop(columns=DROP_COLUMNS, errors='ignore')

        TARGET_COLUMN = 'Health Status'
        if TARGET_COLUMN not in df_clean.columns:
            print(f"Error: Target column '{TARGET_COLUMN}' not found.")
            print("Available columns:", list(df_clean.columns))
            return

        # Encode target variable
        self.label_encoder.fit(df_clean[TARGET_COLUMN])
        df_clean[TARGET_COLUMN] = self.label_encoder.transform(df_clean[TARGET_COLUMN])

        X = df_clean.drop(columns=[TARGET_COLUMN])
        y = df_clean[TARGET_COLUMN]

        # One-hot encode categorical variables
        X = pd.get_dummies(X, drop_first=True)
        self.trained_features = X.columns.tolist()

        # Identify numerical features for scaling
        self.scaler_features = X.select_dtypes(include=np.number).columns.tolist()

        # 🔹 Step 1: Balance the dataset (Upsampling minority class)
        from sklearn.utils import resample
        train_df = pd.concat([X, y], axis=1)
        target_col = 'Health Status'

        # Separate majority and minority
        majority_class = train_df[train_df[target_col] == train_df[target_col].value_counts().idxmax()]
        minority_class = train_df[train_df[target_col] == train_df[target_col].value_counts().idxmin()]

        print(f"Before balancing → Majority: {len(majority_class)}, Minority: {len(minority_class)}")

        # Upsample minority
        minority_upsampled = resample(
            minority_class,
            replace=True,
            n_samples=len(majority_class),
            random_state=42
        )

        # Combine both classes
        balanced_df = pd.concat([majority_class, minority_upsampled])
        X = balanced_df.drop(columns=[target_col])
        y = balanced_df[target_col]

        print(f"After balancing → Each class has {len(minority_upsampled)} samples")

        # 🔹 Step 2: Split and scale
        X_train, _, y_train, _ = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
        X_train[self.scaler_features] = self.scaler.fit_transform(X_train[self.scaler_features])

        # 🔹 Step 3: Train Random Forest
        self.model.fit(X_train, y_train)
        print("✅ Model training complete with balanced data.")

    def save_model(self, filepath='trained_fish_health_model.joblib'):
        """
        Saves the entire trained model instance to a file.
        """
        print(f"Saving model to {filepath}...")
        joblib.dump(self, filepath)
        print("Model saved successfully.")

    @classmethod
    def load_model(cls, filepath='trained_fish_health_model.joblib'):
        """
        Loads a trained model instance from a file.
        """
        print(f"Loading model from {filepath}...")
        model = joblib.load(filepath)
        print("Model loaded successfully.")
        return model

    def _get_specific_disease_risk(self, params):
        """
        Rule-based engine to diagnose specific diseases based on water parameters.
        """
        risks = {
            'Aeromoniasis': 0.0,
            'Bacterial Gill Disease': 0.0,
            'Bacterial Red Disease (Nitrite Poisoning)': 0.0,
            'Saprolegniasis': 0.0,
            'White Tail Disease': 0.0
        }
        ammonia = params.get('Ammonia (mg/L)')
        nitrite = params.get('Nitrite (mg/L)')
        nitrate = params.get('Nitrate (mg/L)')
        do = params.get('Dissolved Oxygen (mg/L)', 7)
        if ammonia is not None and ammonia > 0.1:
            risks['Bacterial Gill Disease'] = max(risks['Bacterial Gill Disease'], min(1.0, (ammonia - 0.1) / 0.9))
        if nitrite is not None and nitrite > 0.2:
            risk_score = min(1.0, (nitrite - 0.2) / 1.0)
            risks['Bacterial Red Disease (Nitrite Poisoning)'] = max(risks['Bacterial Red Disease (Nitrite Poisoning)'], risk_score)
            risks['Aeromoniasis'] += risk_score * 0.5
        if nitrate is not None and nitrate > 50:
            risks['Aeromoniasis'] = max(risks['Aeromoniasis'], min(1.0, (nitrate - 50) / 100.0))
        if do < 4.5:
            risk_score = (4.5 - do) / 2.5
            risks['Aeromoniasis'] = max(risks['Aeromoniasis'], risk_score * 0.6)
            risks['Bacterial Gill Disease'] = max(risks['Bacterial Gill Disease'], risk_score * 0.3)
        temp = params.get('Temperature (°C)', 28)
        turbidity = params.get('Turbidity (NTU)', 5)
        if temp < 24 and turbidity > 15:
            temp_factor = (24 - temp) / 4.0
            turbidity_factor = (turbidity - 15) / 10.0
            risks['Saprolegniasis'] = min(1.0, (temp_factor + turbidity_factor) / 2)
        temp_high = params.get('High Temperature (°C)', 32)
        temp_low = params.get('Low Temperature (°C)', 25)
        if (temp_high - temp_low) > 8:
            risks['White Tail Disease'] = min(1.0, ((temp_high - temp_low) - 8) / 5.0)
        for disease in risks:
            risks[disease] = min(1.0, risks[disease])
        return risks

    def predict(self, scenario_data):
        
        if not self.trained_features:
            print("Model is not trained. Please train the model first.")
            return None

        # Prepare input dataframe
        input_df = pd.DataFrame([scenario_data])
        input_df_encoded = pd.get_dummies(input_df)
        input_df_reindexed = input_df_encoded.reindex(columns=self.trained_features, fill_value=0)
        input_df_reindexed[self.scaler_features] = self.scaler.transform(input_df_reindexed[self.scaler_features])

        # --- ML-based health prediction ---
        status_prediction_encoded = self.model.predict(input_df_reindexed)[0]
        status_prediction = self.label_encoder.inverse_transform([status_prediction_encoded])[0]

        # --- Rule-based risk assessment ---
        specific_risks = self._get_specific_disease_risk(scenario_data)

        # --- Combine intelligently (Rule Override) ---
        avg_risk = np.mean(list(specific_risks.values()))
        max_risk = max(specific_risks.values())

        # If disease risk is high (>40%), override "Stable" to "At Risk"
        if status_prediction == "Stable" and (avg_risk > 0.4 or max_risk > 0.6):
            print(f"⚠️ Rule-based override triggered (avg risk: {avg_risk:.2f}, max risk: {max_risk:.2f})")
            status_prediction = "At Risk (Rule Override)"

        return {
            'Predicted Health Status': status_prediction,
            'Specific Disease Risks': specific_risks
        }


# --- 3. Define Image Preprocessing (Unchanged) ---
def preprocess_image(image_path):
    """
    Loads and preprocesses an image for the ResNet50 model.
    """
    try:
        IMG_SIZE = (224, 224)

        img = keras_image.load_img(image_path, target_size=IMG_SIZE)
        img_array = keras_image.img_to_array(img)
        img_array_expanded = np.expand_dims(img_array, axis=0)
        return resnet50_preprocess(img_array_expanded)
    except FileNotFoundError:
        print(f"Error: Image file not found at {image_path}")
        return None
    except Exception as e:
        print(f"Error preprocessing image: {e}")
        return None

# --- 4. Load All Model Artifacts ---
IMAGE_MODEL_PATH = '/content/drive/MyDrive/project/Fish_img_model_fixed.h5'
WATER_MODEL_PATH = '/content/drive/MyDrive/Project/models/trained_fish_health_model.joblib'

print("Loading models...")
try:
    image_model = tf.keras.models.load_model(IMAGE_MODEL_PATH)
    # Use the load_model method from the class (or joblib.load, both work)
    water_model = FishHealthModel.load_model(WATER_MODEL_PATH)
    print("All models loaded successfully.")
    print(f"Loaded water model type: {type(water_model)}")
    print(f"Water model expects {len(water_model.trained_features)} processed features.")
except Exception as e:
    print(f"--- FATAL ERROR LOADING MODELS ---")
    print(f"Error: {e}")
    image_model = None

# --- 5. Create The "Supportive" Prediction Function (MODIFIED) ---

def predict_supportive(image_path, raw_water_dict):
    """
    Combines predictions using the "Supportive" method.
    Takes an image path and a DICTIONARY of raw water inputs.
    """

    if image_model is None or water_model is None:
        print("Models are not loaded. Cannot predict.")
        return

    # --- 1. Get Image Prediction (Unchanged) ---
    processed_image = preprocess_image(image_path)
    if processed_image is None:
        return

    image_probs = image_model.predict(processed_image)[0]
    image_class_index = np.argmax(image_probs)
    image_prediction = CLASS_LABELS_IMAGE[image_class_index]
    image_confidence = image_probs[image_class_index]

    # --- 2. Get Water Prediction (SIMPLIFIED) ---

    try:
        # Use the class's own predict method
        water_result = water_model.predict(raw_water_dict)
        water_prediction = water_result['Predicted Health Status']
        specific_risks = water_result['Specific Disease Risks']

    except Exception as e:
        print(f"Error during water model prediction: {e}")
        return

    # --- 3. Combine Logically ---
    final_diagnosis = {
        "image_prediction": image_prediction,
        "image_confidence": f"{image_confidence*100:.2f}%",
        "water_status": water_prediction,
        "final_assessment": "",
        "water_based_risks": specific_risks
    }

    is_healthy = "Healthy Fish" in image_prediction

    if is_healthy and water_prediction == 'Stable':
        final_diagnosis["final_assessment"] = "High Confidence: Fish is healthy and water is stable. Conditions are optimal."

    elif is_healthy and water_prediction == 'At Risk':
        final_diagnosis["final_assessment"] = "Warning: Fish appears healthy, but water quality is 'At Risk'. Fish is in danger of future illness. Check rule-based risks."

    elif not is_healthy and water_prediction == 'At Risk':
        final_diagnosis["final_assessment"] = f"High Confidence: Image shows '{image_prediction}', and the 'At Risk' water quality supports this diagnosis."

    elif not is_healthy and water_prediction == 'Stable':
        final_diagnosis["final_assessment"] = f"Low Confidence: Image shows '{image_prediction}', but water quality is 'Stable'. This may be an isolated case, an early infection, or a misidentification. Please double-check."

    return final_diagnosis

# --- 6. Example Usage (MODIFIED) ---

EXAMPLE_IMAGE_PATH = "/content/drive/MyDrive/Project/Freshwater Fish Disease Aquaculture in south asia/Test/Fungal diseases Saprolegniasis/Fungal diseases Saprolegniasis (1).jpeg"

# --- THIS IS THE FIX ---
# Provide the RAW inputs as a DICTIONARY, just like in your notebook
# This uses the 'scenario_healthy' from your code
RAW_WATER_INPUT_DICT = {
    'Average Fish Weight (g)': 310, 'Survival Rate (%)': 88, 'Disease Occurrence (Cases)': 4,
    'Temperature (°C)': 28, 'Dissolved Oxygen (mg/L)': 6.5, 'Turbidity (NTU)': 4,
    'Ammonia (mg/L)': 0.05, 'Nitrite (mg/L)': 1.2, 'Nitrate (mg/L)': 40,
    'Oxygenation Interventions': 0, 'Corrective Interventions': 1,
    'Average Temperature (°C)': 28.5, 'High Temperature (°C)': 32, 'Low Temperature (°C)': 26,
    'Precipitation (inches)': 0.4, 'Month_Num': 9, 'day': 5, 'hour': 13,
    'Oxygenation Automatic': 'Yes', 'Corrective Measures': 'Yes',
    'Thermal Risk Index': 'Moderate', 'Low Oxygen Alert': 'Safe'
}


print(f"\n--- Making 'Supportive' Prediction ---")

final_result = predict_supportive(
    EXAMPLE_IMAGE_PATH,
    RAW_WATER_INPUT_DICT
)

if final_result:
    print("\n--- Dual-Modal Diagnosis ---")
    print(f"Image Prediction: {final_result['image_prediction']}")
    print(f"Image Confidence: {final_result['image_confidence']}")
    print(f"Water Status:     {final_result['water_status']}")
    print("-------------------------------------------------")
    print(f"Assessment:       {final_result['final_assessment']}")
    print("\nSpecific Water-Based Risks:")
    for disease, risk in final_result['water_based_risks'].items():
        print(f"  - {disease}: {risk:.2%}")
else:
    print("Prediction failed. Please check errors above.")