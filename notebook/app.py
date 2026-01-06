import pandas as pd
import numpy as np
import joblib
import gradio as gr
import os

MODEL_PATH = "notebook/trainedmodel/melbourne_price_model.joblib"

# Load the trained pipeline (preprocess + model)
model = joblib.load(MODEL_PATH)

FEATURE_COLS_PATH = "notebook/trainedmodel/feature_columns.csv"
feature_cols = pd.read_csv(FEATURE_COLS_PATH, header=None)[0].tolist()


UI_REF_PATH = "notebook/trainedmodel/ui_reference.csv"
ui_ref = pd.read_csv(UI_REF_PATH)

# Identify numeric/categorical from the UI reference
numeric_cols = ui_ref.select_dtypes(include=[np.number]).columns.tolist()
categorical_cols = [c for c in ui_ref.columns if c not in numeric_cols]

# Defaults: numeric -> median, categorical -> mode
defaults = {}
for c in ui_ref.columns:
    if c in numeric_cols:
        defaults[c] = float(ui_ref[c].median()) if ui_ref[c].notna().any() else 0.0
    else:
        defaults[c] = str(ui_ref[c].mode().iloc[0]) if ui_ref[c].notna().any() else "Unknown"

# For dropdown choices: unique values
choices = {}
for c in categorical_cols:
    choices[c] = sorted(ui_ref[c].dropna().astype(str).unique().tolist())
    if defaults[c] not in choices[c] and len(choices[c]) > 0:
        defaults[c] = choices[c][0]

ui_numeric = [c for c in ["Rooms","Bedroom2","Bathroom","Car","Distance","Landsize","BuildingArea","YearBuilt","Postcode"] if c in ui_ref.columns]
ui_categorical = [c for c in ["Suburb","Type","Regionname","CouncilArea","Method","SellerG"] if c in ui_ref.columns]

def predict_price(*vals):
    # vals are in the order of ui_numeric + ui_categorical
    data = defaults.copy()
    keys = ui_numeric + ui_categorical

    for k, v in zip(keys, vals):
        data[k] = v

    # Ensure numeric cast
    for c in numeric_cols:
        if c in data:
            try:
                data[c] = float(data[c])
            except Exception:
                data[c] = defaults[c]

    # Build 1-row dataframe with ALL training columns
    row_df = pd.DataFrame([data])

    for col in feature_cols:
        if col not in row_df.columns:
            row_df[col] = defaults.get(col, 0)

    row_df = row_df[feature_cols]

    pred = model.predict(row_df)[0]
    return float(pred)

# Build Gradio inputs
inputs = []
for c in ui_numeric:
    inputs.append(gr.Number(label=c, value=defaults[c]))

for c in ui_categorical:
    inputs.append(gr.Dropdown(label=c, choices=choices.get(c, []), value=defaults[c]))

demo = gr.Interface(
    fn=predict_price,
    inputs=inputs,
    outputs=gr.Number(label="Predicted Price (AUD)"),
    title="Melbourne House Price Predictor",
    description="Enter property features and get a predicted sale price using the trained regression model."
)

if __name__ == "__main__":
    port = int(os.environ.get("PORT", os.environ.get("GRADIO_SERVER_PORT", 7860)))
    demo.launch(server_name="0.0.0.0", server_port=port)
