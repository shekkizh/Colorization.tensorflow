"""
Code ideas from https://github.com/Newmu/dcgan and tensorflow mnist dataset reader
"""
import numpy as np
from scipy import misc
from skimage import color


class BatchDatset:
    files = []
    images = []
    image_options = {}
    batch_offset = 0
    epochs_completed = 0

    def __init__(self, records_list, image_options={}):
        """
        Intialize a generic file reader with batching for list of files
        :param records_list: list of files to read -
        :param image_options: A dictionary of options for modifying the output image
        Available options:
        resize = True/ False
        resize_size = #size of output image
        color=LAB, RGB, HSV
        """
        print("Initializing Batch Dataset Reader...")
        print(image_options)
        self.files = records_list
        self.image_options = image_options
        self._read_images()

    def _read_images(self):
        self.images = np.array([self._transform(filename) for filename in self.files])
        print (self.images.shape)

    def _transform(self, filename):
        try:
            image = misc.imread(filename)
            if len(image.shape) < 3:  # make sure images are of shape(h,w,3)
                image = np.array([image for i in range(3)])

            if self.image_options.get("resize", False) and self.image_options["resize"]:
                resize_size = int(self.image_options["resize_size"])
                resize_image = misc.imresize(image,
                                             [resize_size, resize_size])
            else:
                resize_image = image

            if self.image_options.get("color", False):
                option = self.image_options['color']
                if option == "LAB":
                    resize_image = color.rgb2lab(resize_image)
                elif option == "HSV":
                    resize_image = color.rgb2hsv(resize_image)
        except:
            print ("Error reading file: %s of shape %s" % (filename, str(image.shape)))
            raise

        return np.array(resize_image)

    def get_records(self):
        return self.images

    def reset_batch_offset(self, offset=0):
        self.batch_offset = offset

    def next_batch(self, batch_size):
        start = self.batch_offset
        self.batch_offset += batch_size
        if self.batch_offset > self.images.shape[0]:
            # Finished epoch
            self.epochs_completed += 1
            print("****************** Epochs completed: " + str(self.epochs_completed) + "******************")
            # Shuffle the data
            perm = np.arange(self.images.shape[0])
            np.random.shuffle(perm)
            # Start next epoch
            start = 0
            self.batch_offset = batch_size

        end = self.batch_offset
        images = self.images[start:end]
        return np.expand_dims(images[:, :, :, 0], axis=3), images

    def get_random_batch(self, batch_size):
        indexes = np.random.randint(0, self.images.shape[0], size=[batch_size]).tolist()
        images = self.images[indexes]
        return np.expand_dims(images[:, :, :, 0], axis=3), images
