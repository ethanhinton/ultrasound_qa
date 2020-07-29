import pydicom
import matplotlib.pyplot as plt
import numpy as np
import sys
import datetime
from skimage import measure



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
        self.grayscale()

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

    #If image is RGB, this method converts it to grayscale
    def grayscale(self):
        pixels = self.pixels
        if self.data[0x28, 0x2].value == 3:
            self.pixels = pixels[:, :, 2]

    def condense(self, List, factor):
        new_length = int(len(List) / factor)
        standard_deviation = []
        mean_list = []
        old_list = List[:]
        for i in range(new_length):
            mean = []
            for element in range(factor):
                mean.append(List.pop(0))
            mean_list.append(np.mean(mean))
            standard_deviation.append(np.std(mean) / self.minmax(old_list))
        return standard_deviation, mean_list

    def crop_bottom(self):
        if self.type == "LINEAR":
            image = self.pixels
        else:
            image = self.refactored

        width = int(image.shape[1] / 10)
        mean_values = self.middle_values(image, width)
        factor_constant = 15
        factor = int(len(mean_values) / factor_constant)
        standard_dev_list, mean_list = self.condense(mean_values, factor)
        index_cut = self.cutoff_index(standard_dev_list, mean_list, factor)

        if self.type == "LINEAR":
            self.pixels = image[:index_cut, :]
        else:
            self.refactored = image[:index_cut, :]

        plt.imshow(self.refactored)
        plt.show()

    def analyse(self):
        if self.type == "LINEAR":
            pixels = self.pixels
        else:
            pixels = self.refactored

        columns = [sum(pixels[:,i]) for i in range(pixels.shape[1])]
        print(columns)
        cov = self.cov(columns)
        skew = self.skew(columns)
        low_L = self.low(columns[:int(len(columns) / 10)], columns)
        low_CL = self.low(columns[int(len(columns) / 10):int(len(columns) * (3 / 10))], columns)
        low_C = self.low(columns[int(len(columns) * (3 / 10)):int(len(columns) * (7 / 10))], columns)
        low_CR = self.low(columns[int(len(columns) * (7 / 10)):int(len(columns) * (9 / 10))], columns)
        low_R = self.low(columns[int(len(columns) / 10):], columns)
        print(cov, skew, low_L, low_CL, low_C, low_CR, low_R)
        return cov, skew, low_L, low_CL, low_C, low_CR, low_R


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

    @staticmethod
    def cutoff_index(standard_dev_list, mean_list, factor):
        for index, value in enumerate(standard_dev_list):
            if value < 0.015 and mean_list[index] < 230:
                new_index = index * factor
                return new_index
            elif index == len(standard_dev_list) - 1:
                return None

    @staticmethod
    def minmax(val_list):
        min_val = min(val_list)
        max_val = max(val_list)
        return max_val - min_val

    @staticmethod
    def middle_values(image, width):
        middle = int(image.shape[1] / 2)
        mean_values = []
        for i in range(-(int(width / 2)), (int(width / 2))):
            values = []
            for pixel in range(image.shape[0]):
                values.append(image[pixel, middle + i] / width)
            if mean_values == []:
                mean_values = values
            else:
                for pixel in range(len(mean_values)):
                    mean_values[pixel] += values[pixel]
        return mean_values

    @staticmethod
    def cov(columns):
        std = np.std(columns)
        mean = np.mean(columns)
        return (std / mean) * 100

    @staticmethod
    def skew(columns):
        n = len(columns)
        mean = np.mean(columns)
        m1 = 0
        m3 = 0
        for column in columns:
            m1 += (column - mean) ** 2
            m3 += (column - mean) ** 3

        m1 = m1 * (1/(n-1))
        m3 = m3 * (1/n)
        skew = m3 / (m1 ** (3/2))
        return skew

    @staticmethod
    def low(segment, columns):
        median = np.median(columns)
        lowest = 0
        for element in segment:
            if lowest == 0:
                lowest = (element - median) / median
            elif ((element - median) / median) < lowest:
                lowest = (element - median) / median
        return abs(lowest) * 100

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

    def alt_crop_sides(self):
        pixels = self.pixels
        centre = int(pixels.shape[1] / 2)
        for h in range(pixels.shape[1] - centre):
            values = []
            for n in range(pixels.shape[0] - 1):
                values.append(pixels[n, centre + h])
            if self.nonzero_threshold(values, 0.05):
                continue
            else:
                pixels = pixels[:,:centre + (h)]
                break

        for h in range(pixels.shape[1] - centre):
            values = []
            for n in range(pixels.shape[0] - 1):
                values.append(pixels[n, centre - h])
            if self.nonzero_threshold(values, 0.05):
                continue
            else:
                pixels = pixels[:,centre - h:]
                break
        self.pixels = pixels


    def alternative_crop(self):
        image = self.pixels

        # get the pixel information
        cutoff = int(0.08 * image.shape[0])
        image = image[cutoff:, :]

        # convert to a black and white image
        bw = (image > 0)

        # find connected white regions
        labels = measure.label(bw, connectivity=1)
        properties = measure.regionprops(labels)

        # empty area list to add to and then find the biggest area

        maxArea = 0
        maxIndex = 0

        for prop in properties:
            # print('Label: {} >> Object size: {}'.format(prop.label, prop.area))
            if prop.area > maxArea:
                maxArea = prop.area
                maxIndex = prop.label

        bboxCoord = properties[maxIndex - 1].bbox
        minx = bboxCoord[1]
        miny = bboxCoord[0]
        maxx = bboxCoord[3]
        maxy = bboxCoord[2]

        if miny > int(bw.shape[0] / 6):
            bw = bw[:miny, :]
            labels = measure.label(bw, connectivity=1)
            properties = measure.regionprops(labels)
            maxArea = 0
            maxIndex = 0

            # loop over the connected white regions and select the largest region size
            for prop in properties:
                if prop.area > maxArea:
                    maxArea = prop.area
                    maxIndex = prop.label

            # crop the original image to the bounding box of the maximum white region

            bboxCoord = properties[maxIndex - 1].bbox

            minx_new = bboxCoord[1]
            miny_new = bboxCoord[0]
            maxx_new = bboxCoord[3]
            maxy_new = bboxCoord[2]

            if maxy_new - miny_new > 0.05 * pixels.shape[0]:
                croppedImage = image[miny_new:maxy_new, minx_new:maxx_new]
            else:
                croppedImage = image[miny:maxy, minx:maxx]
        else:
            croppedImage = image[miny:maxy, minx:maxx]

        self.pixels = croppedImage
        # save header as one channel
        self.data[0x28, 0x2].value = 1
        plt.imshow(croppedImage)
        plt.show()


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
        STRETCH = 2


        print('cartesian top left point --> ' + str(self.sectorcoords[0][0]))
        print('cartesian centre --> ' + str(self.centre))
        imageleft = self.zero_coords(self.sectorcoords[0][0])
        imageright = self.zero_coords(self.sectorcoords[0][1])
        print('cartesian top left point zeroed --> ' + str(imageleft))
        print('cartesian top left point zeroed --> ' + str(imageright))
        rho_min = int(self.cart2pol(imageleft[1], imageleft[0])[0])
        phi_max = self.cart2pol(imageleft[1], imageleft[0])[1]
        phi_min = self.cart2pol(imageright[1], imageright[0])[1]
        print('polar min radius --> ' + str(rho_min))
        print('polar max angle --> ' + str(abs(phi_max)))
        rho_max = int(self.pixels.shape[0] - self.centre[0])
        print('vertical pixels --> ' + str(self.pixels.shape[0]))
        print('polar max radius --> ' + str(rho_max))
        arc_length = abs(int(2 * (phi_max - phi_min) * rho_max))
        phi_increment = (phi_max - phi_min) / arc_length
        print('arc length --> ' + str(arc_length))
        print('angle increment --> ' + str(phi_increment))

        #Creates a blank linear numpy array to input refactored data into
        refactored = np.ndarray(shape=(STRETCH * (rho_max - rho_min), arc_length))
        print(refactored.shape)

        x = 0
        for j in range(arc_length):
            phi = phi_max - (j * phi_increment)
            y = 0
            for i in range(STRETCH * rho_min, STRETCH * (rho_max - 1)):
                i = i / STRETCH
                cartesian = self.pol2cart(i, phi)
                #print(cartesian)
                cart_reset = self.reset_coords(cartesian)
                # print(j)
                # print(cart_reset[0])
                pixel_val = self.nearest_neighbour(cart_reset[0], cart_reset[1])
                refactored[y,x] = pixel_val

                y += 1
            x += 1

        plt.imshow(refactored)
        plt.show()
        self.refactored = refactored







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
        return (y, x)

