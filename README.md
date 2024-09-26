## Finding brain tumours using image segmentation

This is a small personal project of mine that takes images from the [following dataset](https://www.kaggle.com/datasets/mateuszbuda/lgg-mri-segmentation).

It's built using the [U-Net architecture](https://en.wikipedia.org/wiki/U-Net), which consists of encoding, bottleneck, and decoding layers. The encoding layers are convolutions that progressively increase the number of channels and decrease the image size. The bottleneck does further processing on the many channels which contain features. The decoding layers do the reverse of the encoders; they reduce the number of channels and upsample the image, constructing a mask.

The notebook `segmentation.ipynb` walks you through the data processing phase, creating the model architecture with PyTorch, training the model, and visualising the results.

I turned this into a web application using `Flask`. To try it out, clone the repo, and create a virtual environment. Install the libraries using `pip` from `requirements.txt`. Then, type `python app.py` into your CLI, and the website will start running on your local server.
