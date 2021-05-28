from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, concatenate, Permute, Activation, MaxPooling2D, Conv2D, UpSampling2D, BatchNormalization, Reshape, ZeroPadding2D
import numpy as np
import cv2
import glob
import itertools
from scipy.io import loadmat
import matplotlib.colors as mcolors
from PIL import ImageColor


class SegmentationUnet:
    def __init__(self,  preproc=1, needCheckBackground=False):
        self.labelsType = 0  # mat file
       # self.labelsType = 1  # png file
        self.preproc = preproc
        self.needCheckBackground = needCheckBackground
        self.useChangingAlg = 1
        self.numClasses = 2
        self.height = 480
        self.width = 352
        self.batchSize = 2
        # self.colors = \
        #     [(255, 255, 255), (255, 0, 0), (255, 255, 0), (0, 255, 0), (0, 0, 255), (0, 255, 255)]
        colors = list(mcolors.CSS4_COLORS.values())[0:self.numClasses]
        self.colors = []
        for i in range(0, self.numClasses):
            self.colors.append(ImageColor.getrgb(colors[i]))
        self.buildUnet()

    def loadModel(self, pathToWeights):
        self.model.load_weights(pathToWeights)

    def process(self, imgUploaded):
        self.img = self.prepareSize(imgUploaded, False)
        self.preprocImage()
        labPred = self.model.predict(np.array([self.img]))[0]
        labPred = labPred.reshape((self.outputHeight, self.outputWidth, self.numClasses)).argmax(axis=2)
        segmentationRes = np.zeros((self.outputHeight, self.outputWidth, 3))
        for chann in range(0, 3):
            for classInd in range(0, self.numClasses):
                segmentationRes[:, :, chann] = ((self.colors[classInd][chann]) * (labPred[:, :] == classInd)).astype('uint8') + segmentationRes[:,
                                                                                                                                                :, chann]
        segmentationRes = segmentationRes.astype(np.uint8)
        imageSegmentationSmall = cv2.resize(segmentationRes, (self.width, self.height))
        return imageSegmentationSmall

    def buildUnet(self):
        input = Input(shape=(self.height, self.width, 3))
        u = Conv2D(64, (5, 5), padding='same', activation='relu', data_format='channels_last')(input)
        u = Conv2D(64, (3, 3), padding='same', activation='relu', data_format='channels_last')(u)
        u = MaxPooling2D((2, 2), strides=(2, 2), data_format='channels_last')(u)
        part1 = u
        u = Conv2D(128, (3, 3), activation='relu', padding='same', data_format='channels_last')(u)
        u = Conv2D(128, (3, 3), activation='relu', padding='same', data_format='channels_last')(u)
        u = MaxPooling2D((2, 2), strides=(2, 2), data_format='channels_last')(u)
        part2 = u
        u = Conv2D(256, (3, 3), padding='same', activation='relu', data_format='channels_last')(u)
        u = Conv2D(256, (3, 3), padding='same', activation='relu', data_format='channels_last')(u)
        u = Conv2D(256, (3, 3), padding='same', activation='relu', data_format='channels_last')(u)
        u = MaxPooling2D((2, 2), strides=(2, 2), data_format='channels_last')(u)
        part3 = u
        u = Conv2D(512, (3, 3), padding='same', activation='relu', data_format='channels_last')(u)
        u = Conv2D(512, (3, 3), padding='same', activation='relu', data_format='channels_last')(u)
        u = Conv2D(512, (3, 3), padding='same', activation='relu', data_format='channels_last')(u)
        u = MaxPooling2D((2, 2), strides=(2, 2), data_format='channels_last')(u)
        # end of down side

        u = (ZeroPadding2D((1, 1), data_format='channels_last'))(u)
        u = (Conv2D(512, (3, 3), padding='valid', data_format='channels_last'))(u)
        u = (BatchNormalization())(u)

        u = (UpSampling2D((2, 2), data_format='channels_last'))(u)
        u = (concatenate([u, part3], axis=3))
        u = (ZeroPadding2D((1, 1), data_format='channels_last'))(u)
        u = (Conv2D(256, (3, 3), padding='valid', data_format='channels_last'))(u)
        u = (BatchNormalization())(u)

        u = (UpSampling2D((2, 2), data_format='channels_last'))(u)
        u = (concatenate([u, part2], axis=3))
        u = (ZeroPadding2D((1, 1), data_format='channels_last'))(u)
        u = (Conv2D(128, (3, 3), padding='valid', data_format='channels_last'))(u)
        u = (BatchNormalization())(u)

        u = (UpSampling2D((2, 2), data_format='channels_last'))(u)
        u = (concatenate([u, part1], axis=3))
        u = (ZeroPadding2D((1, 1), data_format='channels_last'))(u)
        u = (Conv2D(64, (3, 3), padding='valid', data_format='channels_last'))(u)
        u = (BatchNormalization())(u)
        u = Conv2D(self.numClasses, (3, 3), padding='same', data_format='channels_last')(u)

        outputHeight = Model(input, u).output_shape[1]
        outputWidth = Model(input, u).output_shape[2]
        u = (Reshape((self.numClasses, outputHeight * outputWidth)))(u)
        u = (Permute((2, 1)))(u)
        u = (Activation('softmax'))(u)
        self.model = Model(input, u)
        self.outputWidth = outputWidth
        self.outputHeight = outputHeight
        return 0

    def preprocImage(self,):
        self.img = self.img.astype(np.float32)
        self.img = cv2.resize(self.img, (self.width, self.height))
        if self.preproc == 1:
            self.img = self.img - 127.5
        elif self.preproc == 2:
            self.img[:, :, 0] = self.img[:, :, 0] - 127  # todo calc meaned color for dataset
            self.img[:, :, 1] = self.img[:, :, 1] - 127
            self.img[:, :, 2] = self.img[:, :, 2] - 127
        # self.img = np.rollaxis(self.img, 2, 0)
        return 0

    def prepareSize(self, imScreen, isAnno):
        color = [255, 255, 255]
        hi, wi, _ = imScreen.shape
        outputSize = (self.width, self.height)
        # if hi<self.height or wi<self.width:
        #     raise ValueError("Input image size so small. height: "+str(hi)+" width: "+str(wi) +" Please use height>=480 width>=352")
        wo, ho = outputSize
        if wi == wo and hi == ho:
            return imScreen
        if hi / float(wi) > ho / float(wo):
            # big height, height same
            wNeed = wo / float(ho) * hi

            delta_w = wNeed - wi
            left, right = delta_w // 2, round(delta_w) - (delta_w // 2)

            imRes = cv2.copyMakeBorder(imScreen, 0, 0, int(left), int(right), cv2.BORDER_CONSTANT, value=color)

        else:
            # big width or square width same
            hNeed = ho / float(wo) * wi

            delta_h = hNeed - hi
            top, bottom = delta_h // 2, round(delta_h) - (delta_h // 2)

            imRes = cv2.copyMakeBorder(imScreen, int(top), int(bottom), 0, 0, cv2.BORDER_CONSTANT, value=color)
        if not isAnno:
            imRes = cv2.resize(imRes, outputSize, interpolation=cv2.INTER_AREA)
        else:
            imRes = cv2.resize(imRes, outputSize, interpolation=cv2.INTER_CUBIC)
        return imRes

    def batchGenerator(self):
        while True:
            imBatch = []
            labBatch = []
            for t in range(0, self.batchSize):
                imPath, segPath = next(self.cylceZip)
                self.img = cv2.imread(imPath, cv2.IMREAD_COLOR)
                self.img = np.asarray(self.img, dtype=object)
                self.preprocImage()
                imBatch.append(self.img)
                if self.labelsType == 1:
                    imgSeg = cv2.imread(segPath, cv2.IMREAD_COLOR)
                    imgSeg = cv2.resize(imgSeg, (self.outputWidth, self.outputHeight))
                    lab = np.zeros((imgSeg.shape[0], imgSeg.shape[1], self.numClasses))
                    imgSeg = imgSeg[:, :, 0]
                    for classInd in range(0, self.numClasses):
                        lab[:, :, classInd] = (imgSeg == classInd).astype(int)
                    labBatch.append(np.reshape(lab, (imgSeg.shape[0] * imgSeg.shape[1], self.numClasses)))
                elif self.labelsType == 0:
                    # imgSeg = loadmat(segPath)['groundtruth']
                    # imgSeg = cv2.resize(imgSeg, (self.outputWidth, self.outputHeight))
                    # lab = np.zeros((imgSeg.shape[0], imgSeg.shape[1], self.numClasses))
                    # for classInd in range(0, self.numClasses):
                    #     lab[:, :, classInd] = (imgSeg == classInd).astype(int)
                    # lab = np.asarray(lab, dtype=object)
                    # labBatch.append(lab)
                    imgSeg = loadmat(segPath)['groundtruth']
                    imgSeg = cv2.resize(imgSeg, (self.outputWidth, self.outputHeight))
                    lab = np.zeros((imgSeg.shape[0], imgSeg.shape[1], self.numClasses))
                    for classInd in range(0, self.numClasses):
                        lab[:, :, classInd] = (imgSeg == classInd).astype(int)
                    labBatch.append(np.reshape(lab, (imgSeg.shape[0] * imgSeg.shape[1], self.numClasses)))
            yield np.array(imBatch), np.array(labBatch)

    def train(self, imgsPath, labelsPath, epochs, weightsPath):
        self.model.compile(loss='categorical_crossentropy', optimizer='adadelta')
        print(imgsPath)
        print(labelsPath)

        imagesAllPath = glob.glob(imgsPath + "*.jpg")
        if self.labelsType == 0:
            labelsAllPath = glob.glob(labelsPath + "*.mat")
        elif self.labelsType == 1:
            labelsAllPath = glob.glob(labelsPath + "*.png")

        arr = np.arange(len(imagesAllPath))
        np.random.shuffle(arr)
        imagesAllPath = np.asarray(imagesAllPath)[arr]
        labelsAllPath = np.asarray(labelsAllPath)[arr]
        self.cylceZip = itertools.cycle(zip(imagesAllPath, labelsAllPath))

        for epoch in range(0, epochs // 10):
            self.model.fit_generator(self.batchGenerator(), 256, epochs=10)
            self.model.save_weights(weightsPath + str((epoch + 1) * 10) + ".dat")
