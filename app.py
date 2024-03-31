from flask import Flask, request, render_template
from keras import Model
from werkzeug.utils import secure_filename
import os
from keras.models import load_model
import tensorflow as tf
import cv2
import numpy as np
import matplotlib.pyplot as plt
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads/'

# Importing Image module from PIL package
from PIL import Image
import PIL
class VAE(tf.keras.Model):
    def __init__(self, latent_dim):
        super(VAE, self).__init__()
        self.latent_dim = latent_dim
        self.encoder = self.build_encoder()
        self.decoder = self.build_decoder()

    def build_encoder(self):
        input_img = tf.keras.Input(shape=(64, 64, 3))
        x = tf.keras.layers.Conv2D(32, 3, activation='relu', padding='same')(input_img)
        x = tf.keras.layers.MaxPooling2D((2, 2), padding='same')(x)
        x = tf.keras.layers.Conv2D(64, 3, activation='relu', padding='same')(x)
        x = tf.keras.layers.MaxPooling2D((2, 2), padding='same')(x)
        x = tf.keras.layers.Conv2D(64, 3, activation='relu', padding='same')(x)
        x = tf.keras.layers.MaxPooling2D((2, 2), padding='same')(x)
        x = tf.keras.layers.Flatten()(x)
        z_mean = tf.keras.layers.Dense(self.latent_dim)(x)
        z_log_var = tf.keras.layers.Dense(self.latent_dim)(x)
        return tf.keras.Model(input_img, [z_mean, z_log_var], name="encoder")

    def build_decoder(self):
        latent_inputs = tf.keras.Input(shape=(self.latent_dim,))
        x = tf.keras.layers.Dense(8 * 8 * 64, activation='relu')(latent_inputs)
        x = tf.keras.layers.Reshape((8, 8, 64))(x)
        x = tf.keras.layers.Conv2DTranspose(64, 3, activation='relu', padding='same')(x)
        x = tf.keras.layers.UpSampling2D((2, 2))(x)
        x = tf.keras.layers.Conv2DTranspose(64, 3, activation='relu', padding='same')(x)
        x = tf.keras.layers.UpSampling2D((2, 2))(x)
        x = tf.keras.layers.Conv2DTranspose(32, 3, activation='relu', padding='same')(x)
        x = tf.keras.layers.UpSampling2D((2, 2))(x)
        decoded = tf.keras.layers.Conv2DTranspose(3, 3, activation='sigmoid', padding='same')(x)
        return tf.keras.Model(latent_inputs, decoded, name="decoder")

    def reparameterize(self, z_mean, z_log_var):
        batch = tf.shape(z_mean)[0]
        dim = tf.shape(z_mean)[1]
        epsilon = tf.keras.backend.random_normal(shape=(batch, dim))
        return z_mean + tf.exp(0.5 * z_log_var) * epsilon

    def call(self, inputs):
        z_mean, z_log_var = self.encoder(inputs)
        z = self.reparameterize(z_mean, z_log_var)
        reconstructed = self.decoder(z)
        return reconstructed


def vgg(file_path):
    image = cv2.imread(file_path)
    image = cv2.resize(image, (32, 32), interpolation=cv2.INTER_AREA)
    image = np.expand_dims(image, axis=0)
    sgd_binary_model = load_model(r'C:\Users\16307\PycharmProjects\project\models\vgg_16_model_8.keras')
    sgd_binary = sgd_binary_model.predict(image)
    maxindex = int(np.argmax(sgd_binary))
    emotion = (['Angry', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise'][maxindex])
    return emotion

def ann_binary(file_path):
    image = cv2.imread(file_path)
    image = cv2.resize(image, (32, 32), interpolation=cv2.INTER_AREA)
    image = np.expand_dims(image, axis=0)
    sgd_binary_model = load_model(r'C:\Users\16307\PycharmProjects\project\models\adam_binary_model_Model-1.keras')
    sgd_binary = sgd_binary_model.predict(image)
    maxindex = int(np.argmax(sgd_binary))
    emotion = (['Angry', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise'][maxindex])
    return emotion

def mini_batch(file_path):
    image = cv2.imread(file_path)
    image = cv2.resize(image, (48, 48), interpolation=cv2.INTER_AREA)
    image = np.expand_dims(image, axis=0)
    sgd_binary_model = load_model(r'C:\Users\16307\PycharmProjects\project\models\mini_batch_sgd_Model_5.keras')
    sgd_binary = sgd_binary_model.predict(image)
    maxindex = int(np.argmax(sgd_binary))
    emotion = (['Angry', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise'][maxindex])
    return emotion

def compare_optimizers(file_path):
    image = cv2.imread(file_path)
    image = cv2.resize(image, (32, 32), interpolation=cv2.INTER_AREA)
    image = np.expand_dims(image, axis=0)
    sgd_binary_model = load_model(r'C:\Users\16307\PycharmProjects\project\models\sgd_cat_model_Mode_2.keras')
    sgd_binary = sgd_binary_model.predict(image)
    maxindex = int(np.argmax(sgd_binary))
    emotion = (['Angry', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise'][maxindex])
    return emotion

def compare_optimizers_2(file_path):
    image = cv2.imread(file_path)
    image = cv2.resize(image, (32, 32), interpolation=cv2.INTER_AREA)
    image = np.expand_dims(image, axis=0)
    sgd_binary_model = load_model(r'C:\Users\16307\PycharmProjects\project\models\adam_binary_model_Model-1.keras')
    sgd_binary = sgd_binary_model.predict(image)
    maxindex = int(np.argmax(sgd_binary))
    emotion = (['Angry', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise'][maxindex])
    return emotion

def compare_optimizers_3(file_path):
    image = cv2.imread(file_path)
    image = cv2.resize(image, (32, 32), interpolation=cv2.INTER_AREA)
    image = np.expand_dims(image, axis=0)
    sgd_binary_model = load_model(r'C:\Users\16307\PycharmProjects\project\models\rmsprop_cat_model.keras')
    sgd_binary = sgd_binary_model.predict(image)
    maxindex = int(np.argmax(sgd_binary))
    emotion = (['Angry', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise'][maxindex])
    return emotion
def ann_categorical(file_path):
    image = cv2.imread(file_path)
    image = cv2.resize(image, (32, 32), interpolation=cv2.INTER_AREA)
    image = np.expand_dims(image, axis=0)
    sgd_binary_model = load_model(r'C:\Users\16307\PycharmProjects\project\models\adam_cat_model_4.keras')
    sgd_binary = sgd_binary_model.predict(image)
    maxindex = int(np.argmax(sgd_binary))
    emotion = (['Angry', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise'][maxindex])
    return emotion

def simple_cnn(file_path):
    image = cv2.imread(file_path)
    image = cv2.resize(image, (48, 48), interpolation=cv2.INTER_AREA)
    image = np.expand_dims(image, axis=0)
    model = load_model(r'C:\Users\16307\PycharmProjects\project\models\simple_cnn_sgd_Model_6.keras')
    sgd_binary = model.predict(image)
    maxindex = int(np.argmax(sgd_binary))
    emotion = (['Angry', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise'][maxindex])
    return emotion

def cnn_regularizer(file_path):
    image = cv2.imread(file_path)
    image = cv2.resize(image, (48, 48), interpolation=cv2.INTER_AREA)
    image = np.expand_dims(image, axis=0)
    model = load_model(r'C:\Users\16307\PycharmProjects\project\models\model_7_cnn_regularizer.keras')
    sgd_binary = model.predict(image)
    maxindex = int(np.argmax(sgd_binary))
    emotion = (['Angry', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise'][maxindex])
    return emotion


def rnn(file_path):
    image = cv2.imread(file_path)
    image = cv2.resize(image, (48, 48), interpolation=cv2.INTER_AREA)
    image = np.expand_dims(image, axis=0)
    model = load_model(r'C:\Users\16307\PycharmProjects\project\models\rnn_sgd_model9.keras')
    sgd_binary = model.predict(image)
    maxindex = int(np.argmax(sgd_binary))
    emotion = (['Angry', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise'][maxindex])
    return emotion


def cnn_lstm(file_path):
    image = cv2.imread(file_path)
    image = cv2.resize(image, (48, 48), interpolation=cv2.INTER_AREA)
    image = np.expand_dims(image, axis=0)
    model = load_model(r'C:\Users\16307\PycharmProjects\project\models\cnn_lstm_sgd_model10.keras')
    sgd_binary = model.predict(image)
    maxindex = int(np.argmax(sgd_binary))
    emotion = (['Angry', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise'][maxindex])
    return emotion

def autoencoder(file_path):
    # Read and preprocess the input image
    image = cv2.imread(file_path)
    image = cv2.resize(image, (48, 48), interpolation=cv2.INTER_AREA)
    image = image / 255.0  # Normalize pixel values to range [0, 1]
    image = np.expand_dims(image, axis=0)

    # Load the autoencoder model
    model = load_model(r'C:\Users\16307\PycharmProjects\project\models\autoencoder_saved_model_11.h5')

    # Generate the reconstructed image using the autoencoder
    reconstructed_image = model.predict(image)

    # Plot the original and reconstructed images

    reconstructed_image_pil = Image.fromarray((reconstructed_image[0] * 255).astype(np.uint8))



    im1 = reconstructed_image_pil.save("static/uploads/re_image.jpg")



    return 'static/uploads/re_image.jpg'


def denoise_autoencoder(file_path):
    image = cv2.imread(file_path)
    image = cv2.resize(image, (32, 32), interpolation=cv2.INTER_AREA)
    image = np.expand_dims(image, axis=0)
    model = load_model(r'C:\Users\16307\PycharmProjects\project\models\autoencoder_saved_model_13.h5')

    reconstructed_image = model.predict(image)

    # Plot the original and reconstructed images

    reconstructed_image_pil = Image.fromarray((reconstructed_image[0] * 255).astype(np.uint8))

    im1 = reconstructed_image_pil.save("static/uploads/re_image.jpg")

    return 'static/uploads/re_image.jpg'


def autoencoder_resnet(file_path):
    image = cv2.imread(file_path)
    image = cv2.resize(image, (48, 48), interpolation=cv2.INTER_AREA)
    image = np.expand_dims(image, axis=0)
    model = load_model(r'C:\Users\16307\PycharmProjects\project\models\autoencoder_saved_model_13.h5')
    sgd_binary = model.predict(image)
    maxindex = int(np.argmax(sgd_binary))
    emotion = (['Angry', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise'][maxindex])
    return emotion

def vae(file_path):
    image = cv2.imread(file_path)
    image = cv2.resize(image, (32, 32), interpolation=cv2.INTER_AREA)
    image = np.expand_dims(image, axis=0)
    latent_dim = 32
    vae = VAE(latent_dim)
    model = Model.load_weights(filepath=r'C:\Users\16307\PycharmProjects\project\models\vae_weights_model_14.h5', skip_mismatch=False)
    reconstructed_image = model.predict(image)

    # Plot the original and reconstructed images

    reconstructed_image_pil = Image.fromarray((reconstructed_image[0] * 255).astype(np.uint8))

    im1 = reconstructed_image_pil.save("static/uploads/re_image.jpg")

    return 'static/uploads/re_image.jpg'


@app.route('/', methods=['GET', 'POST'])
def upload_image():
    if request.method == 'POST':
        # Get the uploaded image file
        file = request.files['image']
        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)
        pdf=""

        # Get the selected value
        selected_value,accuracy = None,None
        if 'ann_binary' in request.form:
            accuracy = '37%'
            selected_value = ann_binary(file_path)
            pdf=""

        elif 'model2' in request.form:
            accuracys = ['Adam: 36%','SGD: 38%','Rmsprop: 34%']



            a=compare_optimizers(file_path)
            b = compare_optimizers_2(file_path)
            c = compare_optimizers_3(file_path)

            selected_value = [a,b,c]
            pdf="https://docs.google.com/document/d/e/2PACX-1vTNqk1Gx8gz-Jq6XJvY6UDfaQr9H-_b4bKJUWWr9ev3tkloo5TfRpNrldhanF2Hj-wmLXpNMcOs7EYO/pub?embedded=true"
            return render_template('result_3.html', file_path=file_path, pdf_path=pdf, selected_value=selected_value,
                               accuracy=accuracys)

        elif 'compare_optimizers' in request.form:
            accuracy = ['Adam: 36%','SGD: 38%','Rmsprop: 34%']
            selected_value = compare_optimizers(file_path)

        elif 'ann_categorical' in request.form:
            accuracy = 'not yet tested'
            selected_value = ann_categorical(file_path)

        elif 'mini_batch' in request.form:
            accuracy = '51%'
            selected_value = mini_batch(file_path)
            pdf = "https://docs.google.com/document/d/e/2PACX-1vQksnCHM8xfl47iCvHq5OFaMkGrGedUOZz5Rq8FmlUP9Pt8m75RGX1fAmZY18692bWVtDrUVo9hudMz/pub?embedded=true"


        elif 'cnn' in request.form:
            accuracy = '51%'
            selected_value = simple_cnn(file_path)
            pdf="https://kluniversityin-my.sharepoint.com/personal/2100030022_kluniversity_in/_layouts/15/Doc.aspx?sourcedoc={6af340ca-ac28-4616-a0bd-5cfd8790dbc7}&amp;action=embedview"

        elif 'cnn_regularizer' in request.form:
            accuracy = 'not yet tested'
            selected_value = cnn_regularizer(file_path)

        elif 'vgg' in request.form:
            accuracy = 'not yet tested'
            selected_value = vgg(file_path)
            pdf=""

        elif 'rnn' in request.form:
            accuracy = 'not yet tested'
            selected_value = rnn(file_path)

        elif 'cnn+lstm' in request.form:
            accuracy = 'not yet tested'
            selected_value = cnn_lstm(file_path)

        elif 'autoencoder' in request.form:
            accuracy = 'not yet tested'
            selected_value = autoencoder(file_path)
            pdf="https://docs.google.com/document/d/e/2PACX-1vQS9YZBVGUckNp6scEmfPz7jHjonmlIdnzZFph5TgeXziZZtknGg-9m0oBX5oes-A/pub?embedded=true"
            return render_template('result_2.html', file_path=file_path, pdf_path=pdf, selected_value=selected_value,
                                   accuracy=accuracy)


        elif 'denoise autoencoder' in request.form:
            accuracy = 'not yet tested'
            selected_value = denoise_autoencoder(file_path)
            pdf = "https://docs.google.com/document/d/e/2PACX-1vQS9YZBVGUckNp6scEmfPz7jHjonmlIdnzZFph5TgeXziZZtknGg-9m0oBX5oes-A/pub?embedded=true"
            return render_template('result_2.html', file_path=file_path, pdf_path=pdf, selected_value=selected_value,
                                   accuracy=accuracy)

        elif 'autoencoder+resnet' in request.form:
            accuracy = 'not yet tested'
            selected_value = autoencoder_resnet(file_path)
            pdf="https://docs.google.com/document/d/e/2PACX-1vT9dxMC1LpZkmGlT4pHPKxuT2ZYlWDueMoyv1Zwnc-8fjKL6PqLToqbv9IpuQJcyw/pub?embedded=true"

        elif 'vae' in request.form:
            accuracy = 'not yet tested'
            selected_value = vae(file_path)
            pdf = "https://docs.google.com/document/d/e/2PACX-1vQS9YZBVGUckNp6scEmfPz7jHjonmlIdnzZFph5TgeXziZZtknGg-9m0oBX5oes-A/pub?embedded=true"
            return render_template('result_2.html', file_path=file_path, pdf_path=pdf, selected_value=selected_value,
                                   accuracy=accuracy)

        # Render the result.html template with the file path and selected value
        return render_template('result.html', file_path=file_path,pdf_path=pdf ,selected_value=selected_value,accuracy=accuracy)

    # Render the index.html template for GET requests
    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)