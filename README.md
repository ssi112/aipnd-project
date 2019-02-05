# AI Programming with Python Project

Project code for Udacity's AI Programming with Python Nanodegree program. In this project, students first develop code for an image classifier built with PyTorch, then convert it into a command line application. The project involves creating an image classifier for 102 different flowers.

The first major step involves loading a pre-trained network. For my project I decided on [DenseNet](https://pytorch.org/docs/stable/torchvision/models.html) (Densely Connected Convolutional Networks). In particular Densenet-121. This and other pre-trained models are used as feature detectors for images that they were not originally trained on. The process of using a model in this manner is called _transfer learning_ (refer to the link below for more info).

Once the pre-trained model is loaded we have to replace it's classifier layer with our own and then train our classifier. _So while we do not need to train the model we do need to train our classifier._ In general, the pre-trained models come with feature detectors already trained to extract information about an image and then feed that to our classifier. Our classifier needs trained to understand that information for our specific images.

For this project we will have three image datasets: one for training our classifier, one for validating it and one for testing. All three sets need normalized to the means and standard deviations of the ImageNet images for which they were originally trained. For the means, it is [0.485, 0.456, 0.406] and the standard deviations is [0.229, 0.224, 0.225].

To help improve accuracy our training images will be randomly flipped and all three image sets will be shuffled as images are fed into our model. Refer to the Jupyter notebook for specific transformations and data loader settings.

Once our model is trained we want to save it for reuse later.


**Additonal Information**
 * Information about tuning Torch Vision models is found [here](https://pytorch.org/tutorials/beginner/finetuning_torchvision_models_tutorial.html).
 * Intro to Convolutional Neural Networks ([video](https://www.youtube.com/watch?v=2-Ol7ZB0MmU))
 * Intro to [Transfer Learning](https://machinelearningmastery.com/transfer-learning-for-deep-learning/)
 * Understanding Softmax and Negative Log-Likelihood can be found in this [notebook](https://ljvmiranda921.github.io/notebook/2017/08/13/softmax-and-the-negative-log-likelihood/)