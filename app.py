import os
import json
import numpy as np
from PIL import Image as PILImage   # For preprocessing
import tensorflow as tf
from flask import Flask, render_template, request, redirect, url_for
from werkzeug.utils import secure_filename
import matplotlib.cm as cm
import cv2
from tensorflow.keras.models import Model  # Import Model
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image as RLImage, Table # For PDF
from reportlab.lib.styles import getSampleStyleSheet
import io
from flask import send_file
from flask import  session
import time

# Flask App
app = Flask(__name__)
app.secret_key = "your_secret_key"  # required for session
# Define upload folder 
UPLOAD_FOLDER = os.path.join("static", "uploads")
os.makedirs(UPLOAD_FOLDER, exist_ok=True)  # Create folder if not exists
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

# Working directory
working_dir = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(working_dir, "plant_disease_prediction_model.keras")  # Correct path

# Load the pre-trained model
model = tf.keras.models.load_model(model_path)

# Force build so model has input/output tensors
dummy_input = np.zeros((1, 224, 224, 3), dtype=np.float32)
_ = model(dummy_input)

# Load class indices
with open(os.path.join(working_dir, "class_indices.json")) as f:
    class_indices = json.load(f)

# Load recommendations
with open(os.path.join(working_dir, "recommendations.json"), encoding="utf-8") as f:
    recommendation_dict = json.load(f)

# Threshold for valid prediction (in percentage)
THRESHOLD = 70

# Function to preprocess image
def load_and_preprocess_image(image_path, target_size=(224, 224)):
    img = PILImage.open(image_path).convert("RGB")  # Ensure 3 channels
    img = img.resize(target_size)
    img_array = np.array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array.astype('float32') / 255.0
    return img_array

def predict_image_class(model, img_array, class_indices, top_k=3):
    predictions = model.predict(img_array)[0]  # Shape: (num_classes,)

    # Sort by probability
    top_indices = predictions.argsort()[::-1][:top_k]
    results = []
    for i in top_indices:
        results.append({
            "class": class_indices[str(i)],
            "confidence": float(predictions[i] * 100)  # Convert to %
        })
    return results  # List of dicts

def get_last_conv_layer(model):
    """Find the last Conv2D layer automatically."""
    for layer in reversed(model.layers):
        if isinstance(layer, tf.keras.layers.Conv2D):
            return layer.name
    raise ValueError("No Conv2D layer found in the model.")

# Grad-CAM function
def get_integrated_gradients(model, img_array, target_class_index=None, m_steps=50, baseline=None):
    """
    Integrated Gradients for model interpretability.
    Args:
        model: Trained keras model
        img_array: Preprocessed image, shape (1, H, W, C)
        target_class_index: Class index to explain (defaults to predicted class)
        m_steps: Steps for approximation
        baseline: Baseline image (defaults to black image)
    Returns:
        attribution heatmap (H, W)
    """
    if baseline is None:
        baseline = np.zeros_like(img_array).astype(np.float32)

    if target_class_index is None:
        preds = model(img_array)
        target_class_index = tf.argmax(preds[0]).numpy()

    # Scale inputs
    alphas = tf.linspace(0.0, 1.0, m_steps+1)

    # Collect gradients
    integrated_grads = tf.zeros_like(img_array, dtype=tf.float32)

    for alpha in alphas:
        interpolated = baseline + alpha * (img_array - baseline)
        with tf.GradientTape() as tape:
            tape.watch(interpolated)
            preds = model(interpolated)
            loss = preds[:, target_class_index]
        grads = tape.gradient(loss, interpolated)
        integrated_grads += grads

    # Average & scale by input difference
    avg_grads = integrated_grads / (m_steps+1)
    integrated_grads = (img_array - baseline) * avg_grads

    # Sum along color channels → (H, W)
    heatmap = tf.reduce_sum(tf.abs(integrated_grads), axis=-1)[0]
    heatmap = (heatmap - tf.reduce_min(heatmap)) / (tf.reduce_max(heatmap) - tf.reduce_min(heatmap) + 1e-8)

    return heatmap.numpy()

def generate_pdf(uploaded_image_path, heatmap_path, plant_name, disease_name, status, confidence, recommendation, alternatives):
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(buffer)
    styles = getSampleStyleSheet()
    elements = []

    # Title
    elements.append(Paragraph("🌱 Plant Disease Report", styles['Title']))
    elements.append(Spacer(1, 20))

    # Plant details
    details = f"<b>Plant:</b> {plant_name}<br/>"
    details += f"<b>Status:</b> {status.capitalize()}<br/>"
    if disease_name:
        details += f"<b>Disease:</b> {disease_name}<br/>"
    details += f"<b>Confidence:</b> {confidence}%"
    elements.append(Paragraph(details, styles['Normal']))
    elements.append(Spacer(1, 20))

    # Uploaded image
    if uploaded_image_path and os.path.exists(uploaded_image_path):
        elements.append(Paragraph("Uploaded Image:", styles['Heading2']))
        elements.append(RLImage(uploaded_image_path, width=200, height=200))
        elements.append(Spacer(1, 20))

    # Heatmap image
    if heatmap_path and os.path.exists(heatmap_path):
        elements.append(Paragraph("Heatmap (Integrated Gradients):", styles['Heading2']))
        elements.append(RLImage(heatmap_path, width=200, height=200))
        elements.append(Spacer(1, 20))

    # Alternatives
    if alternatives:
        data = [["Alternative", "Confidence %"]]
        for alt, conf in alternatives:
            data.append([alt, f"{conf}%"])
        table = Table(data)
        elements.append(Paragraph("Alternative Predictions:", styles['Heading2']))
        elements.append(table)
        elements.append(Spacer(1, 20))

    # Recommendation
    elements.append(Paragraph("Recommendation:", styles['Heading2']))
    elements.append(Paragraph(recommendation, styles['Normal']))

    doc.build(elements)
    buffer.seek(0)
    return buffer

# Routes
@app.route("/", methods=["GET", "POST"])
def index():
    plant_name = None
    disease_name = None
    status = None
    confidence = None
    alternatives = None
    recommendation = None
    uploaded_image = None
    heatmap_image = None  # Initialize gradcam_image

    if request.method == "POST":
        file = request.files.get("file")
        if file:
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config["UPLOAD_FOLDER"], filename)
            file.save(filepath)
            uploaded_image = url_for("static", filename=f"uploads/{filename}")

            # Get predictions
            img_array = load_and_preprocess_image(filepath)
            predictions = predict_image_class(model, img_array, class_indices)

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
                    disease_name = None  # No disease
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

            # Integrated Gradients for top prediction
                heatmap = get_integrated_gradients(model, img_array)

                # Apply heatmap on original image
                img = cv2.imread(filepath)
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                heatmap = cv2.resize(heatmap, (img.shape[1], img.shape[0]))
                heatmap = np.uint8(255 * heatmap)
                heatmap = cm.jet(heatmap)[:, :, :3] * 255

                superimposed_img = cv2.addWeighted(img, 0.6, heatmap.astype("uint8"), 0.4, 0)
                gradcam_path = os.path.join(app.config["UPLOAD_FOLDER"], "gradcam_" + filename)
                cv2.imwrite(gradcam_path, cv2.cvtColor(superimposed_img, cv2.COLOR_RGB2BGR))

                heatmap_image = url_for("static", filename=f"uploads/gradcam_{filename}")

            # Alternatives (skip top prediction)
            if len(predictions) > 1:
                alternatives = [(p["class"], round(p["confidence"], 2)) for p in predictions[1:]]

        # --- ✅ Save to session history ---
        if "history" not in session:
            session["history"] = []

        session["history"].append({
            "image": uploaded_image,
            "prediction": f"{plant_name} - {disease_name if disease_name else status}",
            "confidence": confidence,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
        })
        session.modified = True  # tell Flask to update

    return render_template(
        "index.html",
        plant_name=plant_name,
        disease_name=disease_name,
        status=status,
        confidence=confidence,
        alternatives=alternatives,
        recommendation=recommendation,
        uploaded_image=uploaded_image,
        heatmap_image=heatmap_image,
        history=session.get("history", [])
    )

@app.route("/download_report", methods=["POST"])
def download_report():
    data = request.form
    uploaded_image_path = os.path.join(app.root_path, data.get("uploaded_image_path", ""))
    heatmap_path = os.path.join(app.root_path, data.get("heatmap_path", ""))

    # --- Fix for alternatives parsing ---
    alt_raw = data.get("alternatives", "[]")
    if isinstance(alt_raw, list):
        alternatives = alt_raw
    else:
        try:
            alternatives = json.loads(alt_raw)
        except Exception:
            alternatives = []

    pdf_buffer = generate_pdf(
        uploaded_image_path=uploaded_image_path,
        heatmap_path=heatmap_path,
        plant_name=data.get("plant_name", "Unknown"),
        disease_name=data.get("disease_name", ""),
        status=data.get("status", "Unknown"),
        confidence=data.get("confidence", "0"),
        recommendation=data.get("recommendation", "No recommendation available"),
        alternatives=alternatives
    )

    return send_file(pdf_buffer, as_attachment=True, download_name="disease_report.pdf", mimetype="application/pdf")

@app.route("/clear_history", methods=["POST"])
def clear_history():
    session["history"] = []
    return redirect(url_for("index"))

if __name__ == "__main__":
    app.run(debug=True)
