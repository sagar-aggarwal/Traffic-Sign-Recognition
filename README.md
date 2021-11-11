# Traffic Sign Recognition

Winter Training Project
* Developed a traffic-sign recognition modules using [The German Traffic Sign Recognition Benchmark](https://benchmark.ini.rub.de/gtsrb_news.html). 
* Implemented an LeNet to classify 43 traffic-signs.

## Dataset and Preprocessing

* Images are 32 (width) x 32 (height) x 3 (RGB color channels)
* Training set is composed of 34799 images
* Validation set is composed of 4410 images
* Test set is composed of 12630 images  There are 43 classes (e.g. Speed Limit 20km/h, No entry, Bumpy road, etc.)
* Image normalisation

### Data Distribution and Sample

![ScreenShot](/images/data.png) ![ScreenShot](/images/dist.png)
![ScreenShot](/images/norm.png) 

## Training

While training the network, the following criteria needs to be kept in mind. 
* Cost Function
* Optimizer (Adam, Adamax, RMSprop) 
* Number of epochs and Batch Size (128 selected)
* Learning Rate Decrease

## Files

* train.py - running the keras model
* test.py - testing the accuracy
* utils.py - preprocessing function for histogram normalistaion
