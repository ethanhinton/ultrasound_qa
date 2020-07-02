from DICOMimages import DICOMimage, curvedDICOMimage, linearDICOMimage
from os import path
from pathlib import Path

# VARIABLES
PATH = ""

def main():
    for file in path.abspath(PATH):

def analyse(path):
    image = DICOMimage(path)
    if image.islinear() == True:
        del image
        image = linearDICOMimage(path)
    else:
        del image
        image = curvedDICOMimage(path)
    
