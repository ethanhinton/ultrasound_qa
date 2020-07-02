import os
from pathlib import Path
from DICOMimages import curvedDICOMimage


FILENAME = 'I2LBHP18'

def main():
    path = os.path.join(Path.cwd(), FILENAME)
    im1 = curvedDICOMimage(path)
    print(im1.circle_centre())
    im1.crop()
    im1.showimage()
    print(im1.sectorcoords)

if __name__ == '__main__':
    main()

