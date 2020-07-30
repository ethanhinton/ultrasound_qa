from DICOMimages import DICOMimage, curvedDICOMimage, linearDICOMimage
from pathlib import Path
from os import path
import csv

FILE = 'Images\\J95FISO0'
PATH = path.join(Path.cwd(), FILE)

# Main function which analyses images and sends them to a csv file (REF:5)
def analyse(path):
    image = DICOMimage(path)
    if image.islinear():
        del image
        image = linearDICOMimage(path)
        if image.region == None:
            image.alternative_crop()
            image.alt_crop_sides()
            image.crop_bottom()
            image.showimage()
        else:
            image.main_crop()
            image.crop_sides()
            image.crop_bottom()
            image.showimage()
    else:
        del image
        image = curvedDICOMimage(path)
        image.refactor()
        image.crop_bottom()
        image.showimage()

    cov, skew, L_low, CL_low, C_low, CR_low, R_low = image.analyse()

    # CSV file setup
    with open('output.csv', mode='w') as output:
        output_writer = csv.writer(output, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        HEADERS = ['Date', 'Type', 'Seriel', 'Manufacturer', 'Scanner Model', 'COV', 'Skew', 'Left Low', 'Centre-Left Low', 'Centre Low', 'Centre-Right Low', 'Right Low']
        output_writer.writerow(HEADERS)
        output_writer.writerow([image.date, image.type, image.seriel, image.manufacturer, image.scannermodel, cov, skew, L_low, CL_low, C_low, CR_low, R_low])
        output.close()


analyse(PATH)