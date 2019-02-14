# model_util.py
#                                                                             
# PROGRAMMER: Steve S Isenberg
# DATE CREATED: February 12, 2019
# REVISED DATE: 
# PURPOSE: 
#   Utility functions for loading data and preprocessing images
#
#
# Import modules
import argparse
import os
import errno

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
    parser.add_argument('--data_dir', type = str, default = 'flower_data/', 
                         help = 'Path to the folder of the flower images')
    parser.add_argument('--save_dir', type = str, default = 'checkpoints', 
                         help = 'Path to save the model checkpoints')
    parser.add_argument('--learn_rate', type = str, default = '0.001', 
                         help = 'Model learning rate')  
    return parser.parse_args()


def make_folder(save_dir):
    """
    make the folder if it does not exist
    if it does ignore the error
    """
    try:
        os.makedirs(save_dir)
    except OSError as e:
        if e.errno != errno.EEXIST:
            raise
    

