from DICOMimages import DICOMimage, curvedDICOMimage, linearDICOMimage
from pathlib import Path
from os import path, listdir

#FILE = 'Images\\I2LBHP18'
#FILE = 'Images\\I96DQQOI'
FILE = 'Images\\I2LBHP2A'
PATH = path.join(Path.cwd(), FILE)
#PATH = 'C:\\Users\\eth4n\\Desktop\\Python Projects\\ultrasound-defect-detector\\Linear Serious'


def analyse(path):
    image = DICOMimage(path)
    if image.islinear():
        del image
        image = linearDICOMimage(path)
        if image.region != None:
            image.alternative_crop()
            # image.main_crop()
            image.crop_sides()
            # image.crop_bottom()
        else:
            pass
    else:
        del image
        image = curvedDICOMimage(path)
        image.refactor()
        image.crop_bottom()

    image.showimage()

analyse(PATH)








#IMAGE LOOP
# n = 0
# # for image in listdir(PATH):
# #     try:
# #         print(image.)
# #         analyse(path.join(PATH, image))
# #
# #     except KeyError:
# #         print(image + "failed!")
# #         n += 1
# #
# # print(n)





