For worm motion status recognition. Keras is used here.
### Intro
Worm motion status is classified as five categories, marked by indicies 1 to 5
* 1 Forward
* 2 Backward
* 3 Crossing during forward
* 4 Crossing during backward
* 5 Pause

### Algorithm
* Classification with ResNet-18 without pretrained weights
* He-norm initialization
* L2 weight regularization
* Early stopping
* Learning rate decay

### Data Augmentation
* Brightness enhancement/reduction
* Rotation
* Flipping
* Scaling

### Data Preparation
1. Center-crop raw image/video to square
2. Scale the image to (224, 224)
3. Chose an interval, e.g. 10 frames (1/3 second)
4. Put images of three timesteps, e.g. (frame -10, 0, 10), into three channels of a (224, 224, 3) image as one sample
5. Category of this sample is the motion status of worm in frame 0
