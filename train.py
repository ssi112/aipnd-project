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

"""
************************************************************************
               A Note About the Number of Epochs
************************************************************************
Running more epochs will improve accuracy up to a point. Training loss
will decrease slightly with more epochs, but eventually validation loss
loss plateau and at times increase. A test run of 20 epochs was done,
accuracy test yielded a slight increase: 91% versus 89% as opposed to  
running with only three epochs. It looked like validation loss leveled
out after about nine epochs. Also, run time was consideraly long with 
20 epochs - over 38 minutes.
************************************************************************
"""

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
epochs = 3
hidden_units = 512
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


def build_train_network(arch, learning_rate=0.001, hidden_units=512):
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
    # *******************************************************************************
    # cannot hard code the number of in_features as the model can change
    # update for our dataset depending on model chosen by user
    # ResNet, Inception: input_size = model.fc.in_features
    # VGG: input_size = model.classifier[0].in_features
    # DenseNet: input_size = model.classifier.in_features
    # SqueezeNet: input_size = model.classifier[1].in_channels
    # AlexNet: alexnet.classifier[1].in_features
    # *******************************************************************************
    if (arch == 'vgg16'):
        num_features = model.classifier[0].in_features
    else:
        num_features = model.classifier.in_features
    from collections import OrderedDict
    classifier = nn.Sequential(OrderedDict([
                  ('fc1', nn.Linear(num_features, 512)),
                  ('relu', nn.ReLU()),
                  ('hidden', nn.Linear(512, hidden_units)),                       
                  ('fc2', nn.Linear(hidden_units, 102)),
                  ('output', nn.LogSoftmax(dim = 1))
                  ]))

    # update the model classifier
    model.classifier = classifier

    criterion = nn.NLLLoss()

    # Train the classifier
    optimizer = optim.Adam(model.classifier.parameters(), lr = learning_rate)

    # print( model.state_dict() )
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


def save_checkpoint(ckpt_path, arch):
    """
    Save the model for later reuse
    """
    global model    # use global version
    global optimizer, loss
    print('begin save_checkpoint()...\n')
    input_size = model.state_dict()['classifier.fc1.weight'].size()[1] 
    output_size = model.state_dict()['classifier.fc2.bias'].size()[0] 
    batch_size = dataloaders['train_loader'].batch_size

    print('input size = ', input_size)
    print('output size = ', output_size)
    print('batch size = ', batch_size)

    model_check_point = {
        'input_size': input_size,
        'output_size': output_size,
        'batch_size': batch,
        'epochs': epochs,
        'arch_name': arch,
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
    global model, device, epochs, hidden_units
    in_arg = get_input_args()
    # print(in_arg)
    data_dir = in_arg.data_dir
    train_dir = data_dir + '/train'
    valid_dir = data_dir + '/valid'
    test_dir = data_dir + '/test'
    lr = float(in_arg.learn_rate.strip().strip("'"))
    if (in_arg.arch == 'densenet121'):
        model = models.densenet121(pretrained=True)
    elif (in_arg.arch == 'vgg16'):
        model = models.vgg16(pretrained=True)
    else:
        print('\nERROR *** Train.py supports two models: densenet121 and vgg16 ***')
        print('      *** Please correct and try again ***\n')
        return
    if (in_arg.to_device == 'gpu'):
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = 'cpu'
    epochs = in_arg.epochs
    hidden_units = in_arg.hidden_units
    print('')
    print('data_dir: {}'.format(data_dir))
    print('save_dir: {}'.format(in_arg.save_dir))
    print('learning_rate: {}'.format(lr))
    print('arch: {}'.format(in_arg.arch))
    print('hidden_units: {}'.format(in_arg.hidden_units))
    print('epochs: {}'.format(in_arg.epochs))
    print('device: {}'.format(in_arg.to_device))
    transformations()
    label_mapping()
    build_train_network(in_arg.arch, lr, hidden_units)
    test_network()
    # just in case the checkpoint folder doesn't exist...
    make_folder(in_arg.save_dir)
    save_checkpoint(in_arg.save_dir, in_arg.arch)
    print('\ndone...')
    return 


# let's run this thing
if __name__ == '__main__':
    main()
