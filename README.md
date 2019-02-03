# AI Programming with Python Project

Project code for Udacity's AI Programming with Python Nanodegree program. In this project, students first develop code for an image classifier built with PyTorch, then convert it into a command line application. The project involves creating an image classifier for 102 different flowers.

The first major step involves loading a pre-trained network. For my project I decided on [DenseNet](https://pytorch.org/docs/stable/torchvision/models.html) (Densely Connected Convolutional Networks). In particular Densenet-121.

Once the pre-trained model is loaded we have to replace it's classifier layer with our own and then train our classifier. _So while we do not need to train the model we do need to train our classifier._ In general, the pre-trained models come with feature detectors already trained to extract information about an image and then feed that to our classifier. Our classifier needs trained to understand that information for our specific images.

If interested, additional information about tuning Torch Vision models is found [here](https://pytorch.org/tutorials/beginner/finetuning_torchvision_models_tutorial.html).

