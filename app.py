from flask import Flask, render_template, request, redirect, url_for
from werkzeug.utils import secure_filename
from keras.preprocessing import image
from model import run_model
import numpy as np
import os


# Global prediction variable
image_label = None

# Create and configure app
app = Flask( __name__ )
app.config.from_mapping(
    SECRET_KEY = 'dev',
    DATABASE=os.path.join( app.instance_path, 'flaskr.sqlite' ),
)
app.config["UPLOAD_FILE"] = ["PNG", "JPG", "JPEG"]

# Ensure instance folder exists
try:
    os.makedirs(app.instance_path)
except OSError:
    pass

# Check allowed file
def check_file(filename):
    if not "." in filename:
        return False
    ext = filename.rsplit(".", 1)[1]

    # Check with allowed filetpyes
    if ext.upper() in app.config["UPLOAD_FILE"]:
        return True
    else:
        return False

# Convert image to numpy array
def process_image(path):
    inputArray = []
    imageLoad = image.load_img(path, target_size=(28, 28, 1), color_mode="grayscale")
    imageLoad = image.img_to_array(imageLoad)
    imageNorm = imageLoad/255
    inputArray.append(imageNorm)

    return np.array(inputArray)

# Load model and make prediction
def model_predict(image_array):
    labels = ["Building","Forest","Glacier","Mountain","Sea","Street"]
    model = run_model.load_model()
    model.load_weights("./model/checkpoint/model_checkpoint")
    prediction = np.argmax(model.predict(image_array), axis = 1)

    return labels[prediction[0]]

# Render upload file page
@app.route('/', methods=["GET", "POST"])
def upload_image():

    if request.method == "POST":

        # Saves file in <static/images> folder
        if request.files:
            image = request.files["image"]
            imageName = secure_filename(image.filename)
            if not check_file(imageName):
                print("Invalid file type.")
                return redirect(request.url)
            else: 
                imagePath = os.path.join("./static/images", imageName) 
                image.save(imagePath)
                print("Image saved.")
                imageArray = process_image(imagePath)
                image_label = model_predict(imageArray)

            return render_template("upload_image.html", filename=imageName, imagelabel=image_label)

    return render_template("upload_image.html")

@app.route("/display/<filename>")
def display_image(filename):
    return redirect(url_for('static', filename="images/" + filename), code = 301)


if __name__ == "__main__":
    app.run()