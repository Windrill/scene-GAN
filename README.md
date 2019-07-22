# Scenery-Video-GANs
This Neural Network generates relatively realistic animations from scenic images. This repository includes the saved checkpoint files of the trained network and the model source code. Some examples outputs are stored in ./output/

Network Parameters
-------------------
Input image size: 80x45 pixels.
Output animation size: 80x45x32 pixels.

Data Source
-------------------
Training data is downloaded from Flickr and subsequently processed. This network is trained on approximately 7000 processed video clips each of the size 80x45x32 px, where the entire training process takes around half a day on a PC with GeForce GTX1050.
Video clips are mostly tagged with scenery related labels, such as 'sea'. Because the dataset is pulled directly from Flickr and not filtered, it is highly likely that the training dataset have variable quality.

Training
-------------------
The model is trained in two steps. A smaller model of size 40x23px is trained beforehand. The source code provided on the repository directly reads from the saved checkpoint file of a trained 40x23px Neural Network model.
Refer to "Progressive Growing of GANs" (https://arxiv.org/abs/1710.10196) for an in depth description on how two networks of different sizes are incorporated in this project.
The model is a Convolutional Neural Network, and is also a Generative Adversarial Network.

Loss Function
-------------------
Empirically, both the Wasserstein Loss function and the Mean Squared Error Loss function work well. The model code in this project uses the Mean Squared Error loss function.

Extensions
-------------------
The model can be modified trained in color easily with more training data. In such a situation the input image size would be 80x45x3, and the number of dimensions increase with the inclusion of an RGB channel. 7000 videos cannot accurately generate colored pixels in this model, and in most models.
The model's size can be easily extended under the condition that a GPU with a larger memory is available.

Examples
-------------------
See ./output/
Github cannot locally host gif images, some gif images will be uploaded separately and be post on this markdown file, and more examples will be added.
