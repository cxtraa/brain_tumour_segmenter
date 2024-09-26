from flask import Flask, request, render_template, redirect, flash
from PIL import Image
import torch as t
from utils import predict, UNet
from torchvision import models
import os
from io import BytesIO
import base64

app = Flask(__name__)
app.secret_key = "ABCDEE"
model = UNet(in_channels=3, out_channels=2)
model.load_state_dict(t.load("./model_26092024-154656.pth"))
model.eval()

@app.route("/", methods=["GET", "POST"])
def index():

    if request.method == "POST":
        uploaded_image = request.files["image"]
        
        if uploaded_image:
            image = Image.open(uploaded_image)
            if image.mode != "RGB":
                return redirect(request.url)
            
            buffer_img = BytesIO()
            image.save(buffer_img, format="PNG")
            buffer_img.seek(0)
            uploaded_image_base64 = base64.b64encode(buffer_img.getvalue()).decode("utf-8")

            mask_image = predict(model=model, image=image)
            buffer_mask = BytesIO()
            mask_image.save(buffer_mask, format="PNG")
            buffer_mask.seek(0)
            mask_image_base64 = base64.b64encode(buffer_mask.getvalue()).decode("utf-8")

            return render_template("index.html", upload=uploaded_image_base64, mask=mask_image_base64)

    return render_template("index.html", image=None)

if __name__ == "__main__":
    app.run(debug=True)