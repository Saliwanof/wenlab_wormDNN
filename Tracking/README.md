## Assist Worm Tracking with Deep Convolutional Neural Network

This project aims to assist worm tracking using deep learning methods instead of classical image processing algorithms. 

The worm tracking system mainly consists of a camera ,a moving stage and an analysis&control unit. When working, the camera captures a photo of the worm and delivers the photo to the analysis unit, where several parameters are extracted and used to guide feedback control. The ideal performance of the analysis&control unit is that the camera tightly follows the head of the worm as a result of the movement of the stage. In former practice, the idea of image processing is to thin the worm body to a curve and take the one of those endpoints that is nearest to the head position of last frame as the head of current frame. The analysis is done by Wenbin's code. This algorithms usually performs very well, except when the worm cross with itself and the camera mistaken its target after the head comes out. Although there may be only a few times of self cross-section, this may lead to a wrong tracking target in the following frames if it is not corrected.

Thus the deep convolutional neural network is introduced to solve this problem. The algorithm is as below:
1. downsize the raw image of size (768,1024) to size (192,256)
2. thin the worm body to a cu rve
3. ignore the loops in the curve and find two endpoints
4. take each endpoint as center and make two (50,50) square sections of the image
5. use a CNN to judge how likely the section includes head or tail or neither
6. keep the camera following the endpoint more like head

The python library **keras** is used as a frontend to set up the network. Other python libraries, tensorflow-gpu, h5py, numpy and scipy are used in data and network preprocessing.The framework of the artificial neural network used in this task is a VGG convnet followed by a multilayer perceptron. Before training, the network is initialized with pretrained VGG-16 weights for convNet part and random weights for MLP part.
