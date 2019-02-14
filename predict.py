# predict.py
#                                                                             
# PROGRAMMER: Steve S Isenberg
# DATE CREATED: February 12, 2019
# REVISED DATE: 
# PURPOSE: 
#   Uses a trained network to predict the flower name of the input image.
#   Receives a single file name 
#	Returns the flower name and top K most likely class probabilities
#
# Import modules
import torch
from torch import nn
from torch import optim
# import torch.nn.functional as F
from torchvision import datasets, transforms, models
import time
import numpy as np
import json
from model_util import get_input_args

# ***********************
#   WRITE THE FOLLOWING
# ***********************
# ✔ load the checkpoint
# ✔ process image
# ✔ class prediction
# ✔ show_prediction
# ✔ 

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
    # won't work properly without this
    img = img / 255
    img_norm = (img - img_mean) / img_std
    img_norm = np.transpose(img_norm, (2, 0, 1))
    return torch.Tensor(img_norm) # convert back to PyTorch tensor


def predict(image_path, model, topk=5):
    """
    Predict the class (or classes) of an image using a trained deep learning model.

    """
    # TODO: Implement the code to predict the class from an image file
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


def show_prediction():
	"""
	Display probabilites and name from the image
	"""

	# get the probabilites and classes
	probs, classes = predict(test_img, model, 5)

	# just checking
	print('probabilities: ', probs)
	print('top 5 classes: ', classes)

	flowers = [cat_to_name[i] for i in classes]


def main():
	"""
	"""
	in_arg = get_input_args()
	# print(in_arg)
	data_dir = in_arg.data_dir
	test_dir = data_dir + '/test'
	lr = float(in_arg.learn_rate.strip().strip("'"))
	print('')
	print('data_dir: {}'.format(data_dir))
	print('test_dir: {}'.format(test_dir))
	print('save_dir: {}\n'.format(in_arg.save_dir))
	print('')
	# model, class_to_idx = load_saved_checkpoint('checkpoints/model_checkpoint.pth')
	# show_prob_class(test_img, probs, classes, cat_to_name)
	return 


# let's run this thing
if __name__ == '__main__':
	main()


