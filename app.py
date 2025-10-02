from flask import Flask, request, render_template, send_from_directory, jsonify
import os
import cv2
import numpy as np
from skimage import color

app = Flask(__name__)

UPLOAD_FOLDER = 'static/processed'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Pastikan folder ada
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)


# ========== Route untuk upload manual (via form HTML biasa) ==========
@app.route("/", methods=["GET", "POST"])
def upload_file():
    if request.method == "POST":
        if "file" not in request.files:
            return "No file part"
        file = request.files["file"]
        if file.filename == "":
            return "No selected file"

        filepath = os.path.join(app.config["UPLOAD_FOLDER"], file.filename)
        file.save(filepath)

        # Baca gambar
        image = cv2.imread(filepath)

        # Konversi ke RGB
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Konversi ke CIELAB (OpenCV)
        image_lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)

        # Simpan hasil
        rgb_path = os.path.join(app.config["UPLOAD_FOLDER"], "rgb_" + file.filename)
        cv2.imwrite(rgb_path, cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR))

        lab_path = os.path.join(app.config["UPLOAD_FOLDER"], "lab_" + file.filename)
        cv2.imwrite(lab_path, image_lab)

        return render_template("result.html",
                               original=file.filename,
                               rgb="rgb_" + file.filename,
                               lab="lab_" + file.filename)

    return render_template("index.html")


# ========== Route untuk upload via kamera (JS fetch API) ==========
@app.route("/upload", methods=["POST"])
def upload_capture():
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files["file"]
    filepath = os.path.join(app.config["UPLOAD_FOLDER"], file.filename)
    file.save(filepath)

    # Baca gambar
    image = cv2.imread(filepath)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image_lab = color.rgb2lab(image_rgb)  # pakai LAB standar skimage

    # Ambil midpoint
    h, w, _ = image_rgb.shape
    mid_y, mid_x = h // 2, w // 2
    rgb_val = image_rgb[mid_y, mid_x].tolist()
    lab_val = image_lab[mid_y, mid_x].tolist()

    # Contoh ΔE2000 vs target warna (opsional)
    # target_lab = np.array([50, 0, 0])  # L*, a*, b*
    # deltaE = np.linalg.norm(lab_val - target_lab)
    deltaE = None  # kalau belum dipakai

    return jsonify({
        "rgb": rgb_val,
        "lab": lab_val,
        "deltaE": deltaE
    })


# ========== Route untuk akses file hasil ==========
@app.route("/static/processed/<filename>")
def processed_file(filename):
    return send_from_directory(app.config["UPLOAD_FOLDER"], filename)


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080, debug=True)
