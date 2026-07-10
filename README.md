🐟 Dual-Modal Fish Health Diagnosis System

A hybrid AI-powered fish disease diagnosis system that combines Computer
Vision and Water Quality Analysis to provide more reliable and
context-aware health assessments for aquaculture.

Instead of relying only on fish images or only on environmental data,
this project analyzes both visual symptoms and water quality parameters,
allowing the system to validate predictions and produce more confident
diagnoses.

🎯 Overview

Fish diseases are rarely caused by a single factor. Poor water quality
often weakens fish, making them more vulnerable to bacterial, fungal,
viral, and parasitic infections.

This project integrates: - Image Classification (CNN) - Water Quality
Assessment (Random Forest + Rule-Based Engine)

to generate a reliable final diagnosis.

✨ Features

-   Hybrid AI approach
-   Image-based disease detection
-   Water quality analysis
-   Supportive diagnosis
-   Streamlit dashboard
-   Interactive visualizations
-   JSON report export

📁 Project Structure

    ├── .devcontainer/
    ├── app.py
    ├── water_model.py
    ├── requirements.txt
    ├── Fish_img_model_fixed.h5
    ├── trained_fish_health_model1.joblib
    └── README.md

🧠 Models

Image Classification

Classes: - Healthy Fish - Bacterial Red Disease - Bacterial Diseases –
Aeromoniasis - Bacterial Gill Disease - Fungal Disease –
Saprolegniasis - Parasitic Diseases - Viral Disease – White Tail Disease

Water Quality Assessment

Uses a Random Forest classifier together with an expert rule engine to
classify water as Stable or At Risk based on environmental parameters.

🚀 Running the Project

Clone the repository:

    git clone https://github.com/your-username/Fish-Hybrid-model.git
    cd Fish-Hybrid-model

Install dependencies:

    pip install -r requirements.txt

Run the application:

    streamlit run app.py

Open: http://localhost:8501

📊 Combined Diagnosis

-   Healthy + Stable → Optimal
-   Healthy + At Risk → Warning
-   Diseased + Stable → Warning
-   Diseased + At Risk → High Confidence Alert

🎯 Goal

Provide a practical hybrid diagnostic system that combines computer
vision and water-quality analysis to support early disease detection and
better decision-making in aquaculture.
