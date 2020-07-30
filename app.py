from flask import Flask
from flask import render_template
from flask import request
from flask import redirect
from flask import url_for
from werkzeug.utils import secure_filename
from tensorflow import keras
import numpy
import os


# Global prediction variable
image_label = None

# Create and configure app
app = Flask( __name__ )
app.config.from_mapping(
    SECRET_KEY = 'dev',
    DATABASE=os.path.join( app.instance_path, 'flaskr.sqlite' ),
)
app.config['UPLOAD_FILE'] = ['PNG', 'JPG', 'JPEG']

# Ensure instance folder exists
try:
    os.makedirs(app.instance_path)
except OSError:
    pass

# Check allowed file
def check_file(filename):
    if not '.' in filename:
        return False
    ext = filename.rsplit('.', 1)[1]

    # Check with allowed filetpyes
    if ext.upper() in app.config['UPLOAD_FILE']:
        return True
    else:
        return False

# Convert image to numpy array
def process_image(path):
    inputArray = []
    imageLoad = keras.preprocessing.image.load_img(path, target_size=(28, 28, 3))
    imageLoad = keras.preprocessing.image.img_to_array(imageLoad)
    imageNorm = imageLoad/255
    inputArray.append(imageNorm)
    return numpy.array(inputArray)

# Load model and make prediction
def model_predict(image_array):
    labels = ['Building','Forest','Glacier','Mountain','Sea','Street']
    model = keras.models.load_model('./model/test_model.h5')
    model.load_weights('./model/checkpoint/model_checkpoint')
    prediction = numpy.argmax(model.predict(image_array), axis = 1)

    return labels[prediction[0]]

# Render upload file page
@app.route('/', methods=['GET', 'POST'])
def upload_image():

    if request.method == 'POST':

        # Saves file in <static/images> folder
        if request.files:
            image = request.files['image']
            imageName = secure_filename(image.filename)
            if not check_file(imageName):
                print('Invalid file type.')
                return redirect(request.url)
            else: 
                imagePath = os.path.join('./static/images', imageName) 
                image.save(imagePath)
                print('Image saved.')
                imageArray = process_image(imagePath)
                image_label = model_predict(imageArray)

            return render_template('upload_image.html', filename=imageName, imagelabel=image_label)

    return render_template('upload_image.html')

@app.route('/display/<filename>')
def display_image(filename):
    return redirect(url_for('static', filename='images/' + filename), code = 301)

if __name__ == '__main__':
    app.run()