import pydicom
import matplotlib.pyplot as plt
import numpy as np
import sys
import datetime



class DICOMimage:

    def __init__(self, path):
        self.path = path
        self.data = pydicom.read_file(path)
        self.pixels = self.data.pixel_array
        self.type = self.data[0x18, 0x6031].value
        self.date = DICOMimage.reformat_date(self.data[0x8, 0x23].value)
        self.seriel = self.data[0x18, 0x1000].value
        self.manufacturer = self.data[0x8, 0x70].value
        self.scannermodel = self.data[0x8, 0x1090].value

        # Extracts image location data from DICOM header in form (x1, y1, x2, y2, x(centre of image))
        self.region = [self.data[0x18, 0x6011][0][0x18, 0x6018].value,
                       self.data[0x18, 0x6011][0][0x18, 0x601A].value,
                       self.data[0x18, 0x6011][0][0x18, 0x601C].value,
                       self.data[0x18, 0x6011][0][0x18, 0x601E].value,
                       self.data[0x18, 0x6011][0][0x18, 0x6020].value + self.data[0x18, 0x6011][0][0x18, 0x6018].value]

    def showimage(self):
        plt.imshow(self.pixels)
        plt.show()

    # Prints pixel array in full
    def pixelarray(self):
        np.set_printoptions(threshold=sys.maxsize)
        return self.pixels

    def islinear(self):
        if self.type == "LINEAR":
            return True
        else:
            return False

    @staticmethod
    def reformat_date(date):
        year = int(date[:4])
        if date[4] == 0:
            month = int(date[5])
        else:
            month = int(date[4:6])
        if date[6] == 0:
            day = int(date[7])
        else:
            day = int(date[6:8])
        return datetime.date(year, month, day).strftime('%d,%m,%Y')

    @staticmethod
    def nonzero_threshold(pixels, threshold):
        nonzeros = sum(1 for pixel in pixels if pixel.any() !=0)
        if nonzeros/len(pixels) >= threshold:
            return True
        else:
            return False



class linearDICOMimage(DICOMimage):

    #crops the image to remove information from the outside FINISH!!!!!!!
    def main_crop(self):
        pixels = self.pixels
        region = self.region
        self.region[4] = region[4] - region[0]
        self.pixels = pixels[region[1]:region[3], region[0]:region[2]]
        self.data.PixelData = self.pixels.tobytes()
        
    def crop_sides(self):
        pixels = self.pixels
        centre = self.region[4]

        for h in range(pixels.shape[1] - centre):
            values = []
            for n in range(pixels.shape[0] - 1):
                values.append(pixels[n, centre + h])
            if self.nonzero_threshold(values, 0.05):
                continue
            else:
                self.pixels = pixels[:, centre - (h-2):centre + (h-2)]
                break


class curvedDICOMimage(DICOMimage):

    def __init__(self, path):
        super().__init__(path)
        # Finds the coordinates of the top two points of the curved image and the coordinates of the middle of the sector
        self.sectorcoords = self.find_top_values(), self.find_middle_value()
        self.centre = self.circle_centre()



    # finds the coordinates of the two points at the top of the curved image (labelled x1,y1 and x2,y2 in diagram) CHANGE FOR REFERENCE PIXELS
    def find_top_values(self):
        xmiddle = self.region[4]
        height = self.region[1] + 1
        for index, pixel in enumerate(self.pixels[height, xmiddle:]):
            if pixel.all() != 0:
                return [height,xmiddle - index], [height, xmiddle + index]

    # finds the x and y coordinates of the middle of the top arc of the image (labelled xm, ym in diagram)
    def find_middle_value(self):
        xmiddle = self.region[4]
        s_height = self.region[1]
        for index, pixel in enumerate(self.pixels[s_height:, xmiddle]):
            if pixel.all() != 0:
                return [s_height + index, xmiddle]

    # Finds the centre of the circle that the image arcs follow (i.e. the origin of the signal)
    def circle_centre(self):
        x1, x2 = self.sectorcoords[0]
        middle = self.sectorcoords[1]
        m = middle[0] - x1[0]
        l = middle[1] - x1[1]
        r1 = (l**2 + m**2) / (2*m)
        h1 = int(np.sqrt(r1**2 - l**2))
        return [middle[0] - m - h1 , middle[1]]


    def refactor(self):
        print('cartesian top left point --> ' + str(self.sectorcoords[0][0]))
        print('cartesian centre --> ' + str(self.centre))
        imageleft = self.zero_coords(self.sectorcoords[0][0])
        imageright = self.zero_coords(self.sectorcoords[0][1])
        print('cartesian top left point zeroed --> ' + str(imageleft))
        print('cartesian top left point zeroed --> ' + str(imageright))
        rho_min = self.cart2pol(imageleft[0], imageleft[1])[0]
        phi_max = self.cart2pol(imageleft[0], imageleft[1])[1]
        print('polar min radius --> ' + str(rho_min))
        print('polar max angle --> ' + str(abs(phi_max)))
        rho_max = self.pixels.shape[0] - self.centre[0]
        print('vertical pixels --> ' + str(self.pixels.shape[0]))
        print('polar max radius --> ' + str(rho_max))
        arc_length = int(2 * phi_max * rho_max)
        phi_increment = (2 * phi_max) / arc_length
        print('arc length --> ' + str(arc_length))
        print('angle increment --> ' + str(phi_increment))


        for i in range(int(rho_min), int(rho_max)):
            cartesian = self.pol2cart(i, phi_max)
            cart_reset = self.reset_coords(cartesian)
            pixel_val = self.nearest_neighbour(cart_reset[0], cart_reset[1])
            print(pixel_val)


    def zero_coords(self, point):
        return (point[0] - self.centre[0], point[1] - self.centre[1])

    def reset_coords(self, point):
        return (point[0] + self.centre[0], point[1] + self.centre[1])

    def nearest_neighbour(self, y, x):
        y_round = int(round(y))
        x_round = int(round(x))
        return self.pixels[y_round, x_round]




    @staticmethod
    def cart2pol(x, y):
        rho = np.sqrt(x ** 2 + y ** 2)
        phi = np.arctan2(y, x)
        return (rho, phi)

    @staticmethod
    def pol2cart(rho, phi):
        x = rho * np.cos(phi)
        y = rho * np.sin(phi)
        return (x, y)