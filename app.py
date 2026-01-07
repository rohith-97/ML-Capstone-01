import pickle
import gradio as gr

model_file = 'rf_model_40_trees_depth_10_min_samples_leaf_1.bin'

with open(model_file, 'rb') as f_in:
    One_Hot_encoder, rf = pickle.load(f_in)

def predict_gradio(gender, age, hypertension, heart_disease, smoking_history,
                   bmi, HbA1c_level, blood_glucose_level):
    patient = {
        "gender": gender,
        "age": float(age),
        "hypertension": int(hypertension),
        "heart_disease": int(heart_disease),
        "smoking_history": smoking_history,
        "bmi": float(bmi),
        "HbA1c_level": float(HbA1c_level),
        "blood_glucose_level": float(blood_glucose_level),
    }

    X = One_Hot_encoder.transform([patient])
    y_prob = float(round(rf.predict_proba(X)[0, 1], 3))
    label = "Diabetes" if y_prob >= 0.5 else "No diabetes"
    return y_prob, label

title = "Diabetes Risk Predictor"
description = "Enter patient data to get a diabetes probability and simple label."

inputs = [
    gr.inputs.Dropdown(['Male', 'Female', 'Other'], label='gender'),
    gr.inputs.Number(default=50, label='age'),
    gr.inputs.Checkbox(label='hypertension (yes=checked)'),
    gr.inputs.Checkbox(label='heart_disease (yes=checked)'),
    gr.inputs.Textbox(lines=1, placeholder='e.g. never, current, former', label='smoking_history'),
    gr.inputs.Number(default=25.0, label='bmi'),
    gr.inputs.Number(default=5.5, label='HbA1c_level'),
    gr.inputs.Number(default=100.0, label='blood_glucose_level'),
]

outputs = [gr.outputs.Number(label='diabetes_probability'), gr.outputs.Label(label='label')]

iface = gr.Interface(fn=predict_gradio, inputs=inputs, outputs=outputs,
                     title=title, description=description, allow_flagging='never')

if __name__ == '__main__':
    iface.launch(server_name='0.0.0.0', server_port=7860)
