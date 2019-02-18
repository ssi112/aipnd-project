# IMAGE CLASSIFIER COMMAND LINE APPLICATION
#   train.py
#
# Usage:
#   python train.py --data_dir flowers --save_dir rabbit_foot --learn_rate 0.01
#                                                                             
# PROGRAMMER: Steve S Isenberg
# DATE CREATED: February 12, 2019
# REVISED DATE: 
# PURPOSE: 
#   Train a network on a dataset and save the model checkpoint
#
# Import modules
import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
from torchvision import datasets, transforms, models
import time
import numpy as np
from torch.autograd import Variable
from collections import OrderedDict
import json
from model_util import get_input_args, make_folder

# ***********************
#   WRITE THE FOLLOWING
# ***********************
# ✔ define transforms for training, validation and testing datasets
# ✔ create label mapping
# ✔ build and train network classifier
# ✔ validate test dataset
# ✔ test the network - print accuracy
# ✔ save the checkpoint

data_dir = 'flowers'    # default - can be supplied by user
train_dir = data_dir + '/train'
valid_dir = data_dir + '/valid'
test_dir = data_dir + '/test'
model = models.densenet121(pretrained=True)
device = 'cuda'
data_transforms = {}
image_datasets = {}
dataloaders = {}
batch = 64
epochs = 3 # can change to 1 for testing to shorten run time
optimizer = 0
loss = 0

def transformations():
    """
    Load all the data: training, validation and test datasets
    Resizes and crops the data, including random flippingof training set
    """
    global data_transforms, image_datasets, dataloaders
    global data_dir, train_dir, valid_dir, test_dir
    print('begin transformations()...\n')
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                  std=[0.229, 0.224, 0.225])
    data_transforms = {
        'train_set': transforms.Compose([transforms.Resize(256),
                                         transforms.RandomResizedCrop(224),
                                         transforms.RandomHorizontalFlip(),
                                         transforms.ToTensor(),
                                         normalize]),
        
        'valid_set': transforms.Compose([transforms.Resize(256),
                                         transforms.CenterCrop(224),
                                         transforms.ToTensor(),
                                         normalize]),
        
        'test_set':  transforms.Compose([transforms.Resize(256),
                                         transforms.CenterCrop(224),
                                         transforms.ToTensor(),
                                         normalize])
    }

    # Load the datasets with ImageFolder
    image_datasets = {
        'train_data': datasets.ImageFolder(train_dir, transform = data_transforms['train_set']),
        'valid_data': datasets.ImageFolder(valid_dir, transform = data_transforms['valid_set']),
        'test_data':  datasets.ImageFolder(test_dir,  transform = data_transforms['test_set'])
    }

    # Using the image datasets and the trainforms, define the dataloaders
    dataloaders = {
        'train_loader': torch.utils.data.DataLoader(image_datasets['train_data'], batch_size=batch, shuffle=True),
        'valid_loader': torch.utils.data.DataLoader(image_datasets['valid_data'], batch_size=batch, shuffle=True),
        'test_loader' : torch.utils.data.DataLoader(image_datasets['test_data'],  batch_size=batch, shuffle=True)
    }
    # return


def label_mapping():
    """
    dictionary mapping the integer encoded categories to the actual names of the flowers
    """
    print('begin label_mapping()...\n')
    with open('cat_to_name.json', 'r') as f:
        cat_to_name = json.load(f)
    # Not a requirement for project - used to check labels
    # prints dictionary sorted by category name
    # this can be commented out, but is here for a check
    #for cat_key, cat_value in sorted(cat_to_name.items(), key=lambda x: x[1]): 
    #   print("{}: {}".format(cat_key, cat_value))
    # return


def run_validation(model, valid_data_loader, criterion):
    valid_loss = valid_accuracy = 0
    for images, labels in valid_data_loader:
        # Variable() - Wraps a tensor and records the operations applied to it
        # https://pytorch.org/docs/0.3.1/_modules/torch/autograd/variable.html
        images = Variable(images)
        labels = Variable(labels)
        images, labels = images.to(device), labels.to(device)
        
        output = model.forward(images)
        valid_loss += criterion(output, labels).item()
        
        ps = torch.exp(output)  # softmax log probability
        equality = (labels.data == ps.max(dim = 1)[1])
        # mean() is not a method on the tensor, 
        # so convert the type to a float tensor to use mean()
        valid_accuracy += equality.type_as(torch.FloatTensor()).mean()
    return valid_loss, valid_accuracy


def build_train_network(learning_rate=0.001):
    """
    Uses PyTorch pretrained densenet121 CNN
    Ref: https://pytorch.org/docs/stable/torchvision/models.html
    Parm:
        learning_rate
    """
    global model    # make sure we use global version
    global device, optimizer
    global data_transforms, image_datasets, dataloaders
    print('begin build_train_network()...\n')
    # Freeze parameters so we don't back propagate through them
    # turns off gradient descent so we don't train them again when we use our dataset
    for param in model.parameters():
        param.requires_grad = False

    # the classifier downloaded from the model as follows:
    # Linear(in_features=1024, out_features=1000, bias=True)
    # update for our dataset
    from collections import OrderedDict
    classifier = nn.Sequential(OrderedDict([
                   # note we have to match the number of final output features to our classifier
                   # as the model is designed - 1024 in this case
                  ('fc1', nn.Linear(1024, 1000)),
                  ('relu', nn.ReLU()),
                  ('fc2', nn.Linear(1000, 102)), # 102 output categories
                  ('output', nn.LogSoftmax(dim = 1))
                  ]))

    # update the model classifier
    model.classifier = classifier

    criterion = nn.NLLLoss()

    # Train the classifier
    optimizer = optim.Adam(model.classifier.parameters(), lr = learning_rate)

    # print( model.state_dict() )

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('\nUsing device:{}\n'.format(device))
    """
    *******************************************************************************
    Actual training of the classifier
    """
    print_every = 20
    steps = 0

    start_time = time.time()
    print('   begin training...\n')
    model = model.train()
    for epoch in range(epochs):
        model.to(device)
        running_loss = 0
        for ii, (images, labels) in enumerate(dataloaders['train_loader']):
            steps += 1
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            
            # Forward and backward passes
            outputs = model.forward(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            if steps % print_every == 0:
                # Make sure network is in evaluation mode for inference
                # turns dropout OFF
                model.eval()
                model.to(device)
                # Turn off gradients for validation, saves memory and 
                # computations when doing validation
                with torch.no_grad():
                    valid_loss, valid_accuracy = run_validation(model, dataloaders['valid_loader'], criterion)
                print("Epoch: {} of {}... ".format(epoch+1, epochs),
                      "Training Loss: {:.4f}".format(running_loss / print_every),
                      "Validation Loss: {:.4f}...".format(valid_loss / len(dataloaders['valid_loader'])),
                      "Validation Accuracy: {:.4f}...".format(valid_accuracy / len(dataloaders['valid_loader']))
                     )
                running_loss = 0
                # Make sure training is back on - turns dropout ON
                model.train()
    # output the training and validation stats
    end_time = time.time()
    elapsed_time = end_time - start_time
    print('Completed {:02d} epoch{} using device: {}'.format(epochs, 's' if epochs > 1 else '', device))
    print("\n** Elapsed Runtime:",
              str(int((elapsed_time/3600)))+":"+str(int((elapsed_time%3600)/60))+":"
              +str(int((elapsed_time%3600)%60)) )
    # return


def test_network():
    """
    Validate using test dataset
    Run test images through the network and measure the accuracy
    Parms:
        model 
    """
    global model    # use global version
    global data_transforms, image_datasets, dataloaders
    print('begin test_network()...\n')
    correct = 0
    total = 0
    with torch.no_grad():
        model.to(device)
        # Make sure network is in evaluation mode for inference turns dropout off
        model.eval()
        for ii, (inputs, labels) in enumerate(dataloaders['test_loader']):
            inputs, labels = inputs.to('cuda'), labels.to('cuda')
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    print('Accuracy of the network on test images: %d %%' % (100 * correct / total))
    # return 


def save_checkpoint(ckpt_path):
    """
    Save the model for later reuse
    """
    global model    # use global version
    global optimizer, loss
    print('begin save_checkpoint()...\n')
    input_size = model.state_dict()['classifier.fc1.weight'].size()[1] #1024
    output_size = model.state_dict()['classifier.fc2.bias'].size()[0] #102
    batch_size = dataloaders['train_loader'].batch_size

    model_check_point = {
        'input_size': input_size,
        'output_size': output_size,
        'batch_size': batch,
        'epochs': epochs,
        'arch_name': 'densenet121',
        'classifier': model.classifier,
        'model_class_index': image_datasets['train_data'].class_to_idx,
        'model_state': model.state_dict(),
        'optimizer_state': optimizer.state_dict(),
        'model_loss': loss
    }

    # if folder doesn't exist it will create it
    make_folder(ckpt_path)
    # trainpy_checkpoint.pth here to distinguish it from jupyter notebook
    torch.save(model_check_point, ckpt_path+'/trainpy_checkpoint.pth')
    # return


def main():
    global data_dir, train_dir, valid_dir, test_dir
    in_arg = get_input_args()
    # print(in_arg)
    data_dir = in_arg.data_dir
    train_dir = data_dir + '/train'
    valid_dir = data_dir + '/valid'
    test_dir = data_dir + '/test'
    lr = float(in_arg.learn_rate.strip().strip("'"))
    print('')
    print('data_dir: {}'.format(data_dir))
    print('learning_rate: {}'.format(lr))
    print('save_dir: {}\n'.format(in_arg.save_dir))
    transformations()
    label_mapping()
    build_train_network(lr)
    test_network()
    # just in case the checkpoint folder doesn't exist...
    make_folder(in_arg.save_dir)
    save_checkpoint(in_arg.save_dir)
    print('\ndone...')
    return 


# let's run this thing
if __name__ == '__main__':
    main()
