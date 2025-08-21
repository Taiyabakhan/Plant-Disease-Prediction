import os
import json
import numpy as np
from PIL import Image
import tensorflow as tf
from flask import Flask, render_template, request, redirect, url_for
from werkzeug.utils import secure_filename


# Flask App
app = Flask(__name__)

# Define upload folder 
UPLOAD_FOLDER = os.path.join("static", "uploads")
os.makedirs(UPLOAD_FOLDER, exist_ok=True)  # create folder if not exists
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

# Working directory
working_dir = os.path.dirname(os.path.abspath(__file__))
model_path = f"{working_dir}/plant_disease_prediction_model.keras"  # correct path
# Load the pre-trained model
model = tf.keras.models.load_model(model_path)

# Load class indices
class_indices = json.load(open(f"{working_dir}/class_indices.json"))
# Load recommendations
recommendations_path = f"{working_dir}/recommendations.json"
with open(recommendations_path, encoding="utf-8") as f:
    recommendation_dict = json.load(f)

# Threshold for valid prediction (in percentage)
THRESHOLD = 70

# Function to preprocess image
def load_and_preprocess_image(image, target_size=(224, 224)):
    img = Image.open(image).convert("RGB")   # ensure 3 channels
    img = img.resize(target_size)
    img_array = np.array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array.astype('float32') / 255.0
    return img_array


# Prediction function
def predict_image_class(model, image, class_indices, top_k=3):
    preprocessed_img = load_and_preprocess_image(image)
    predictions = model.predict(preprocessed_img)[0]  # shape: (num_classes,)

    # Sort by probability
    top_indices = predictions.argsort()[::-1][:top_k]
    results = []
    for i in top_indices:
        results.append({
            "class": class_indices[str(i)],
            "confidence": float(predictions[i] * 100)  # convert to %
        })

    return results  # list of dicts



# Routes
# @app.route("/", methods=["GET", "POST"])
# def index():
#     prediction = None
#     if request.method == "POST":
#         if "file" not in request.files:
#             return redirect(request.url)
#         file = request.files["file"]
#         if file.filename == "":
#             return redirect(request.url)
#         if file:
#             prediction = predict_image_class(model, file, class_indices)
#     return render_template("index.html", prediction=prediction)

@app.route("/", methods=["GET", "POST"])
def index():
    plant_name = None
    disease_name = None
    status = None
    confidence = None
    alternatives = None
    recommendation = None
    uploaded_image = None

    if request.method == "POST":
        file = request.files.get("file")
        if file:
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config["UPLOAD_FOLDER"], filename)
            file.save(filepath)
            uploaded_image = url_for("static", filename=f"uploads/{filename}")

            # Get predictions
            predictions = predict_image_class(model, filepath, class_indices)

            # Top prediction
            if predictions:
                top_pred = predictions[0]
                confidence = round(top_pred["confidence"], 2)
                result = top_pred["class"]

                # Split plant and disease
                if " - " in result:
                    plant_name, disease_name = result.split(" - ", 1)
                else:
                    plant_name = result
                    disease_name = None

                # Determine healthy/disease status
                if "Healthy" in result:
                    status = "healthy"
                    disease_name = None  # no disease
                else:
                    status = "disease"

                # Recommendation
                if confidence < THRESHOLD:
                    result = "Uncertain"
                    recommendation = "Model is not confident. Please upload a clearer image of a plant leaf."
                else:
                    recommendation = recommendation_dict.get(result, 
                        f"No specific treatment found for {result}. General advice: maintain proper watering, sunlight, and pest management."
                    )

            # Alternatives (skip top prediction)
            if len(predictions) > 1:
                alternatives = [(p["class"], round(p["confidence"], 2)) for p in predictions[1:]]

    return render_template(
        "index.html",
        plant_name=plant_name,
        disease_name=disease_name,
        status=status,
        confidence=confidence,
        alternatives=alternatives,
        recommendation=recommendation,
        uploaded_image=uploaded_image
    )



if __name__ == "__main__":
    app.run(debug=True)
