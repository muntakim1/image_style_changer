from flask import Flask,render_template,request, redirect
import os
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = "true"
from skimage import io
import time
import numpy as np
import tensorflow as tf
import tensorflow_hub as hub
from utils import load_image
hub_handle = 'https://tfhub.dev/google/magenta/arbitrary-image-stylization-v1-256/2'
hub_module = hub.load(hub_handle)

app = Flask('__name__')
app.config["IMAGE_UPLOADS"] ="./static/images/"
app.config["ALLOWED_IMAGE_EXTENSIONS"] = ["JPEG", "JPG", "PNG", "GIF"]
content_img_size =(1920,1024)
style_img_size = (256, 256) 

def allowed_image(filename):

    if not "." in filename:
        return False

    ext = filename.rsplit(".", 1)[1]

    if ext.upper() in app.config["ALLOWED_IMAGE_EXTENSIONS"]:
        return True
    else:
        return False


@app.route("/", methods=["GET", "POST"])
def FileUpload():       
    k= request.form.get('number')
    if request.method == "POST":
        k= request.form.get('number')
        if request.files:

            image = request.files["image"]

            if image.filename == "":
                print("No filename")
                return redirect('/')

            if allowed_image(image.filename) and k!=None:
                
                i=0
                image.save(os.path.join(app.config["IMAGE_UPLOADS"], str(i)+".jpg"))
                

                content_image = load_image('./static/images/0.jpg', content_img_size)
                style_image = load_image('./static/styles/'+k+'.jpg', style_img_size)
                # print(content_image,style_image)
                # style_image = tf.nn.avg_pool(style_image, ksize=[3,3], strides=[1,1], padding='SAME')
                outputs = hub_module(tf.constant(content_image), tf.constant(style_image))
                stylized_image = outputs[0]
                print(stylized_image)
                img = np.squeeze(stylized_image)
                io.imsave("./static/processed/processed-0.jpg",img)
                print("Image saved")

                return render_template('index.html',k=k)

            else:
                print("That file extension is not allowed")
                return render_template('index.html',k=k)
            try:
                os.remove("./static/processed/processed-0.jpg")
            except OSError:
                pass
        print(k)
        return render_template('index.html')
    return render_template('index.html')

if __name__ == "__main__":

    app.run(debug=True)
