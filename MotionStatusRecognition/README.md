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
