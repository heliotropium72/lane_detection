# Lane Detection
[![Udacity - Self-Driving Car NanoDegree](https://s3.amazonaws.com/udacity-sdc/github/shield-carnd.svg)](http://www.udacity.com/drive)

### Term 1, Project 4: Advanced Lane Finding
### Keywords: Computer Vision, Camera Calibration, Perspective Transform


[//]: # (Image References)

[image1]: ./Figures/Distortion.png "Undistorted"
[image_test]: ./Figures/Test_images.png "Test images"
[images_test_d]: ./Figures/Test_images_detected.png "Test images"
[video1]: ./Videos/video1_detected.mp4 "Video"

---

In this project the lane lines on a highway course are detected based on computer vision and directly drawn on the images of the frontal camera. Checkout the [project rubric](https://review.udacity.com/#!/rubrics/571/view) for more details.
The software pipeline contains the following steps

* Camera calibration matrix and distortion coefficients given a set of chessboard images.
* Distortion correction to raw images.
* Use color transforms, gradients, etc., to create a thresholded binary image.
* Perspective transform to rectify binary image ("birds-eye view").
* Detection of lane pixels and fit to find the lane boundary.
* Curvature of the lane and vehicle position with respect to center.
* Warping of the detected lane boundaries back onto the original image.
* Visual display of the lane boundaries and numerical estimation of lane curvature and vehicle position.

The pipeline was first tested on 8 test images and then on the video streams ...
xx 



---

### Camera Calibration

The file `undistort.py` contains all code necessary for the camera calibration and the function `undistort(<single image>)` which is used in the pipeline of the video stream.
The camera calibration is determined using a 9x6 chessboard. 20 images of this chessboard can be found in the folder "camera_cal". The camera calibration is performed by mapping the corner points of the chessboard `objpts` (which are symmetrical)
to their image points `imgpts`.
The points in the image plane and the 3d-plane are saved in the lists `imgpts` and `objpts` for all 20 images.
The `objpts` is easily created due to the symmetry of the chessboard. The chessboard is assumed to be fixed on the xy-plane (z=0) with the first point in the origin.
The measure of the coordinate system is such that the distance between two corners corresponds to 1 measure i.e. the points are [(0,0,0), (1,0,0), (2,0,0), ..., (9,6,0)]. The set of one set of object points in one image is saved in `objp`.
The `imgpts` contains the position of the corners in the image plane òf a single image `corners` which are detected making using of the OpenCV function cv2.findChessboardCorners(). They are further refined to subpixel level by using cv2.cornerSubPix().
After every succesful detection of the corners, the set of `corners` and `objp is appended to `imgpts` and `objpts` respectively.

I then used the output `objpts` and `imgpts` to compute the camera calibration and distortion coefficients using the `cv2.calibrateCamera()` function.
The camera matrix of the camera used in this project is

```
		1156.94	0		665.948
mtx = 	0		1152.13	388.786
		0		0		1
```

and the distortion parameters are `dist` = [-0.238, -0.085, -0.0008, -0.0001, 0.106].

I applied this distortion correction to the test image using the `cv2.undistort()` function and obtained this result: 

![alt text][image1]

The camera calibration can then be applied on every image of the camera by calling

```python
def undistort(img, mtx=mtx, dist=dist):
    undist = cv2.undistort(img, mtx, dist, None, mtx)
    return undist
```
The camera matrix `mtx` and the distortion parameters `dist` are the ones which were determines before.

---

### Lane detection pipeline

The lane detection pipeline applied to a single image is summarised in the following graph. In the following the single steps are described in more detail.

![alt text][image]

#### 1. Distortion correction
As a first step, the images are distortion corrected using the function described above

#### 2. HLS colour space
Next, the image is converted to the HLS colour space using ```cv2.cvtColor(image, cv2.COLOR_RGB2HLS).astype(np.float)```.
The HLS space has the advantage that the image is separated in hue (H), lightness (L) and saturation (S) channels. Differences in illumination (e.g. tree shadows) will not affect the saturation channel.
Hence, thresholds of the saturation channel are more universal than thresholds of a colour (red, green, blue) channel.

![alt text] [image_hls]

#### 3. Threshold of saturation channel
...
...
...
binary output image

#### 4. Perspective Transform
The binary images are transformed to a birds-eye view using the function ```birdseye(image, src, dst)```. For this end, four points spanning a straight lane (on test image 0 ) are selected. These are transformed to the corners of a new, birds-eye view image.
These source and destination points were hard-coded:

| Source        | Destination   | 
|:-------------:|:-------------:| 
| 585, 460      | 320, 0        | 
| 203, 720      | 320, 720      |
| 1127, 720     | 960, 720      |
| 695, 460      | 960, 0        |
 

#### 5. Lane-line pixel selection
sliding window depending if previous lane is provided or not.

#### 6. 2nd order polynomial describing the lanes in birdseye view
In birdseye view, the lane lines can be described by a second order polynomial. The polynomials are fitted to all detected lane-line pixels. The area between the left and right line polynomial is then ascribed to the lane.

#### 7. Derived parameters
Several parameters can be calculated from the resulting lane. These parameters can then be used to check whether the result is reasonable.
First, the curvature for every lane line is calculated from the polynomial.
...
Offset from lane center.
...

The parameters will be directly plotted into the resulting image

#### 8. Sanity check
... checks ... restart if necessary ...


#### 10. Pipeline applied to the 8 test images
![alt text][images_test_d]

---

### Implementation
The code is organised in two main classes: The `Line` class containing information about either the left or the right lane-line and the `Lane`class which has corresponding pair of left and right lane-lines as input.
The `Lane` class further contains a memory `Lane.previous`, so that the newly detected lane-lines can be compared to the previous ones.

---

### Result
The above pipeline was then applied to the video provided by Udacity. The result can be seen [here] (./Videos/video1_detected.mp4)


---

### Discussion: Challenges and improvements

The selection of the source points for the perspective tranform is important as the quality of the warped images and therefore polynomial fit is highly sensitive to it. These points were selected by hand. It might be possible to 
improve the pipeline by barely tweaking these source points. E.g. by reducing the vertical extension of the lane, the pipeline becomes more robust on the one hand but the detected lane is shorter on the other hand.

The pipeline cannot handle missing stripes. For example, when the right lane is often interrupted and thus detected lane-line pixel are only in the lower half of the birds-eye view image,
the polynomial fit is likely to be off for the pixel towards the upper end of the lane. This could be possibly improved by applying a rolling average over the k last images or by tuning the thresholds even more.
