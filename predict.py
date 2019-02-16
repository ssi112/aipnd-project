# IMAGE CLASSIFIER COMMAND LINE APPLICATION
#   predict.py
#
# USAGE:
#   python predict.py --data_dir flowers --save_dir checkpoints --path_to_image /9/image_06410.jpg
#
# Some Example files for testing:
#   /3/image_06634.jpg
#   /7/image_07215.jpg
#   /33/image_06460.jpg
#   /71/image_04514.jpg
#
# PROGRAMMER: Steve S Isenberg
# DATE CREATED: February 14, 2019
# REVISED DATE: 
# PURPOSE: 
#   Uses a trained network to predict the flower name of the input image.
#   Receives a single file name 
#   Returns the flower name and top K most likely class probabilities
#
# Import modules
import torch
from torch import nn
from torch import optim
import json
from torchvision import datasets, transforms, models
import time
import numpy as np
import pandas as pd
import argparse
import os
from PIL import Image

# ***********************
#   WRITE THE FOLLOWING
# ***********************
# ✔ load the checkpoint
# ✔ process image
# ✔ class prediction
# ✔ show_prediction
# ✔ command line args

def load_saved_checkpoint(model_path):
    """
    loads a saved checkpoint and rebuilds the model
    """
    saved_model = torch.load(model_path)
    model = saved_model['model']
    model.classifier = saved_model['classifier']
    model.load_state_dict(saved_model['model_state'])
    model.class_to_idx = saved_model['model_class_index']
    optimizer = saved_model['optimizer_state']
    epochs = saved_model['epochs']
    for param in model.parameters():
        param.requires_grad = False
    return model, saved_model['model_class_index']


def process_image(image):
    ''' 
    Process a PIL image for use in a PyTorch model
    Scales, crops, and normalizes a PIL image for a PyTorch model,
    '''
    img_mean = np.array([0.485, 0.456, 0.406])
    img_std  = np.array([0.229, 0.224, 0.225])
    img = Image.open(image)
    img = img.resize((256,256))

    # just checking
    width, height = img.size
    # print('width={} height={}'.format(width, height))

    left = (width - 224) / 2
    top = (height - 224) / 2
    right = (width + 224) / 2
    bottom = (height + 224) / 2
    img = img.crop((left, top, right, bottom)) #crop out center

    # make this a numpy array
    img = np.array(img)
    # RGB values are 8-bit: 0 to 255
    # dividing by 255 gives us a range from 0.0 to 1.0
    img = img / 255
    img_norm = (img - img_mean) / img_std
    img_norm = np.transpose(img_norm, (2, 0, 1))
    return torch.Tensor(img_norm) # convert back to PyTorch tensor


def predict(image_path, model, topk=5):
    """
    Predict the class (or classes) of an image using a trained deep learning model.

    """
    # implement the code to predict the class from an image file
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    model.eval() # inference mode
    img = process_image(image_path)
    
    # without this get an error about sizes not matching
    # Input type (torch.FloatTensor) and weight type (torch.cuda.FloatTensor) should be the same
    img = img.to(device)
    
    # not sure why unsqueeze is needed, something to do with the batch
    # could not get answer from mentors or within student hub or knowledge
    img = img.unsqueeze(0)  

    with torch.no_grad():
        logits = model.forward(img)
        # https://pytorch.org/docs/stable/torch.html#torch.topk
        # returns the topk largest elements of the given input tensor
        probs, probs_labels = torch.topk(logits, topk)
        probs = probs.exp() # calc all exponential of all elements
        class_to_idx = model.class_to_idx
    
    # more errors, can't convert CUDA tensor to numpy. 
    # Use Tensor.cpu() to copy the tensor to host memory first.
    # thanks for the suggestion!
    probs = probs.cpu().numpy()
    probs_labels = probs_labels.cpu().numpy()
    
    # gets the indexes in numerical order: 0 to 101
    classes_indexed = {model.class_to_idx[i]: i for i in model.class_to_idx}
    # and still more errors - must be a list!
    classes_list = list()
    
    for label in probs_labels[0]:
        classes_list.append(classes_indexed[label])
        
    return (probs[0], classes_list)


def show_prediction(probs, classes):
    """
    Display probabilites and name from the image
    """
    with open('cat_to_name.json', 'r') as f:
        cat_to_name = json.load(f)
    flower_names = [cat_to_name[i] for i in classes]

    df = pd.DataFrame(
        {'flowers': pd.Series(data=flower_names),
         'probabilities': pd.Series(data=probs, dtype='float64')
        })
    print(df)


def get_input_args():
    """
    Retrieves and parses the 3 command line arguments provided by the user when
    they run the program from a terminal window. This function uses Python's 
    argparse module to created and defined these 3 command line arguments. If 
    the user fails to provide some or all of the 3 arguments, then the default 
    values are used. 

    Command Line Arguments:
      1. Location of folder with flower images: --data_dir with default value 'flower_data'
      2. Location of folder to save the checkpoints: --save_dir with default value 'checkpoints'
      3. Learning rate for the model: --learn_rate with default value '0.001'

    Returns:
     parse_args() -data structure that stores the command line arguments object  
    """

    # Create Parse using ArgumentParser
    parser = argparse.ArgumentParser()
    
    # command line options
    parser.add_argument('--data_dir', type = str, default = 'flowers/', 
                         help = 'Path to the folder of the flower images')
    parser.add_argument('--save_dir', type = str, default = 'checkpoints', 
                         help = 'Path to save the model checkpoints')
    parser.add_argument('--path_to_image', type = str, default = '/99/image_07833.jpg', 
                         help = 'Path to an image file')  
    return parser.parse_args()


def main():
    """
    """
    in_arg = get_input_args()
    data_dir = in_arg.data_dir
    test_dir = data_dir + 'test'
    image_file = in_arg.path_to_image
    print('')
    print('data_dir: {}'.format(data_dir))
    print('test_dir: {}'.format(test_dir))
    print('save_dir: {}'.format(in_arg.save_dir))
    print('image_file: {}\n'.format(test_dir+image_file))
    print('')
    # make sure checkpoint exists
    if os.path.exists(in_arg.save_dir+'/trainpy_checkpoint.pth'):
        model, class_to_idx = load_saved_checkpoint(in_arg.save_dir+'/trainpy_checkpoint.pth')
        probs, classes = predict(test_dir+image_file, model, topk=5)
        show_prediction(probs, classes)
    else:
        print('Oops, checkpoint does NOT exist! ({})'.format(in_arg.save_dir+'/trainpy_checkpoint.pth'))
    return 


# let's run this thing
if __name__ == '__main__':
    main()

