# Lane Detection
[![Udacity - Self-Driving Car NanoDegree](https://s3.amazonaws.com/udacity-sdc/github/shield-carnd.svg)](http://www.udacity.com/drive)

### Term 1, Project 4: Advanced Lane Finding
### Keywords: Computer Vision, Camera Calibration, Perspective Transform


[//]: # (Image References)

[image1]: ./Figures/Distortion.png "Undistorted"
[image_test]: ./Figures/Test_images.png "Test images"
[images_test_d]: ./Figures/Test_images_detected.png "Test images with detected lanes"
[image_pipe]: ./Figures/Pipeline/pipeline_0.png "Pipline"
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

The pipeline was first tested on 8 test images and then on the video streams.

![alt text][image_test]

---

### Implementation
The code is organised in three files: 
- `undistort.py` Camera calibration and distortion correction functions
- `lane_detection.py` Main program containing the lane detection pipeline
- `lane.py` Definition of the two classes `Line` and `Lane`

The `Line` class contains information about either the left or the right lane-line of a single image. An object of the `Lane` class consists of two elements of the `Line` class corresponding 
to a left and right lane-lines. Further, it contains a memory of previous lane detections `Lane.previous`, so that the newly detected lane-lines can be compared to the previous ones.
Functions which operate on a single line (polynomial fit, curvature calculation, ...) are methods of the `Line` class, while functions needing input from both lines (positional offset, plot, ...) are methods of the `Lane`class.
In the following only the functional behavior of the code is discussed, but the source code is available in the above files for detailed inspection.

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
	1156.94	0	665.948
mtx = 	0	1152.13	388.786
	0	0	1
```

and the distortion parameters are `dist` = [-0.238, -0.085, -0.0008, -0.0001, 0.106].

For an image of the calibration set, this distortion correction looks like the following. Their is also a third image illustrating the perspective transfrom based on the four outermost corners.

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

The lane detection pipeline applied to a single image is summarized in the following graph. In the following the single steps are described in more detail.
Here is an example image; the pipeline images for all test images can be found in the folder ["Figures/Pipeline"](./Figures/Pipeline/).

![alt text][image_pipe]

#### 1. Distortion correction
As a first step, the images are distortion corrected using the function described above

#### 2. HLS colour space
Next, the image is converted to the HLS colour space using ```cv2.cvtColor(image, cv2.COLOR_RGB2HLS).astype(np.float)```.
The HLS space has the advantage that the image is separated in hue (H), lightness (L) and saturation (S) channels. Differences in illumination (e.g. tree shadows) will not affect the saturation channel.
Hence, thresholds of the lightness and saturation channel are more universal than thresholds of a colour (red, green, blue) channel.
The HLS decomposition of the test images can be found in the folder ["Figures/HLS"](./Figures/HLS/).

#### 3. Threshold of lightness and saturation channel
The lightness channel measures the amount of white. It can be efficiently used with a high threshold to detect white line markings (which are not obscured by shadows). Although it can only a detect a limited fraction of the lane 
i.e. intensive white stripes, it does it with high confidentiality and hence is a good starting point to make the pipeline more robust.
The saturation channel detects the intensive yellow lines and partly the white lines. A threshold of the saturation channel is therefore used as backbone of the pipeline.
Optionally, a threshold of the sobel line detection of the lightness channel could be used. However, in the following code it was not used.
The result of the different thresholds is a combined binary images containing potential lane line pixels.

#### 4. Perspective Transform
The binary images are transformed to a birds-eye view using the function ```birdseye(image, src, dst)```. For this end, four points spanning a straight lane (on test image 0 ) are selected. These are transformed to the corners of a new, birds-eye view image.
These source and destination points were hard-coded:

| Source        | Destination   | 
|:-------------:|:-------------:| 
| 578, 460      | 150, 0        | 
| 235, 690      | 150, 720      |
| 1069, 690     | 1150, 720      |
| 705, 460      | 1150, 0        |
 
These areas are displayed in red in the above pipeline figure. The birds-eye view of the test images can be found in the folder ["Figures/Birdseye"](./Figures/Birdseye/).
 
#### 5. Lane-line pixel selection
When no previous lane is provided or the previous fit was insufficient, the lane line pixels are detected from scratch. For that end a sliding window approach is chosen (function `_sliding_window()`).
The windows are sliding in vertical/y direction. For every window, the histogram in horizontal/z direction is taken and the two peaks (one for each lane line) are detected. All points around a margin (here +- 50 pixel) are assumed to be lane line pixel.

If a previous lane is provided, new lane-line pixels are only searched in a small window (+- 50 pixel margin) around the previous lane.

#### 6. 2nd order polynomial describing the lanes in birdseye view
In birdseye view, the lane lines can be described by a second order polynomial. The polynomials are fitted to all detected lane-line pixels. The area between the left and right line polynomial is then ascribed to the lane.
The user can choose between only fitting the lane-line pixel of the current image (`Line.fit_lane`) or including also lane-line pixel of the (last three) previously detected lanes (`Line.fit_lane_incl_previous`)
This functionality is inside the `Line` class.

The fit result is displayed in yellow in the above pipeline figure.

#### 7. Derived parameters
Several parameters can be calculated from the resulting lane. These parameters can then be used to check whether the result is reasonable.
First, the curvature for every lane line is calculated from the polynomial. Then the offset from lane center. They are displayed in the upper left corner of the result image.

#### 8. Sanity check
The derived parameters have to fullfill several conditions to be considered reasonable (`Lane.check_sanity`):
- the lane was detected on both sides
- the car is driving in the lane center (+- 1m)
- the left and right curvatures are similar (+- 250m) or the line is straight (>1000m)
- the curvature is in the same direction (check difference in fit coefficients of left and right lane)

If any of the conditions does not hold, the sliding window approach is first reseted i.a. the lane-line pixels are searched from scratch.
If the lane detection is still unsuccessful, the previous fit is applied and the image is skipped.

#### 10. Pipeline applied to the 8 test images
![alt text][images_test_d]

---

### Result
The above pipeline was then applied to the video provided by Udacity. The result can be seen [here](./Videos/video1_detected.mp4)


---

### Discussion: Challenges and improvements

- Perspective Transform
The selection of the source points for the perspective transform is important as the quality of the warped images and therefore polynomial fit is highly sensitive to it. These points were selected by hand. It might be possible to 
improve the pipeline by barely tweaking these source points. E.g. by reducing the vertical extension of the lane, the pipeline becomes more robust on the one hand but the detected lane is shorter on the other hand.

- Missing stripes
The pipeline cannot handle missing stripes. For example, when the right lane is often interrupted and thus detected lane-line pixel are only in the lower half of the birds-eye view image,
the polynomial fit is likely to be off for the pixel towards the upper end of the lane. This could be possibly improved by applying a rolling average over the k last images or by tuning the thresholds even more.

- Defined shadows
The shadow of a wall as in the udacity video 2 is still a challenge for the pipeline as the gradient is very strong. Such a shadow is more likely to be detected as lane line than the actual line
