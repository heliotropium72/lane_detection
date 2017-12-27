# Lane Detection
[![Udacity - Self-Driving Car NanoDegree](https://s3.amazonaws.com/udacity-sdc/github/shield-carnd.svg)](http://www.udacity.com/drive)
Term 1, Project 4: Advanced Lane Finding
Keywords: Computer Vision, Camera Calibration, Perspective Transform

---

In this project the lane lines on a highway course are detected based on computer vision and directly drawn on the images of the frontal camera. Checkout the [project rubric](https://review.udacity.com/#!/rubrics/571/view) for more details.
The software pipline contains the following steps

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


[//]: # (Image References)

[image1]: ./Figures/Distortion.png "Undistorted"
[image2]: ./test_images/test1.jpg "Road Transformed"
[image3]: ./examples/binary_combo_example.jpg "Binary Example"
[image4]: ./examples/warped_straight_lines.jpg "Warp Example"
[image5]: ./examples/color_fit_lines.jpg "Fit Visual"
[image6]: ./examples/example_output.jpg "Output"
[video1]: ./project_video.mp4 "Video"
---

### Camera Calibration

#### 1. Briefly state how you computed the camera matrix and distortion coefficients. Provide an example of a distortion corrected calibration image.
The file `undistort.py` contains all code necessary for the camera calibration and the function `undistort(<singel image>)` which is used in the pipeline of the video stream.
The camera calibration is determined using a 9x6 chessboard. 20 images of this chessboard can be found in the folder "camera_cal". The camera calibration is performed by mapping the corner points of the chessboard `objpts` (which are symmetrical)
to their image points `imgpoints`.
The points in the image plane and the 3d-plane are saved in the lists `imgpoints` and `objpoints` for all 20 iamges.
The `objpoints` is easily created due to the symmetry of the chessboard. The chessboard is assumed to be fixed on the xy-plane (z=0) with the first point in the origin.
The measure of the coordinate system is such that the distance between two corners corresponds to 1 measure i.e. the points are [(0,0,0), (1,0,0), (2,0,0), ..., (9,6,0)]. The set of one set of object points in one image is saved in `objp`.
The `imgpts` contains the position of the corners in the image plane òf a single image `corners` which are detected making using of the OpenCV function cv2.findChessboardCorners(). They are further refined to subpixel level by using cv2.cornerSubPix().
After every succesful detection of the corners, the set of `corners` and `objp is appended to `imgpts` and `objpts` respectively.

I then used the output `objpoints` and `imgpoints` to compute the camera calibration and distortion coefficients using the `cv2.calibrateCamera()` function.
The camera matrix of the camera used in this project is

		1156.94	0		665.948
mtx = 	0		1152.13	388.786
		0		0		1

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


### Implementation

#### 1. Provide an example of a distortion-corrected image.

To demonstrate this step, I will describe how I apply the distortion correction to one of the test images like this one:
![alt text][image2]

#### 2. Describe how (and identify where in your code) you used color transforms, gradients or other methods to create a thresholded binary image.  Provide an example of a binary image result.

I used a combination of color and gradient thresholds to generate a binary image (thresholding steps at lines # through # in `another_file.py`).  Here's an example of my output for this step.  (note: this is not actually from one of the test images)

![alt text][image3]

#### 3. Describe how (and identify where in your code) you performed a perspective transform and provide an example of a transformed image.

The code for my perspective transform includes a function called `warper()`, which appears in lines 1 through 8 in the file `example.py` (output_images/examples/example.py) (or, for example, in the 3rd code cell of the IPython notebook).  The `warper()` function takes as inputs an image (`img`), as well as source (`src`) and destination (`dst`) points.  I chose the hardcode the source and destination points in the following manner:

```python
src = np.float32(
    [[(img_size[0] / 2) - 55, img_size[1] / 2 + 100],
    [((img_size[0] / 6) - 10), img_size[1]],
    [(img_size[0] * 5 / 6) + 60, img_size[1]],
    [(img_size[0] / 2 + 55), img_size[1] / 2 + 100]])
dst = np.float32(
    [[(img_size[0] / 4), 0],
    [(img_size[0] / 4), img_size[1]],
    [(img_size[0] * 3 / 4), img_size[1]],
    [(img_size[0] * 3 / 4), 0]])
```

This resulted in the following source and destination points:

| Source        | Destination   | 
|:-------------:|:-------------:| 
| 585, 460      | 320, 0        | 
| 203, 720      | 320, 720      |
| 1127, 720     | 960, 720      |
| 695, 460      | 960, 0        |

I verified that my perspective transform was working as expected by drawing the `src` and `dst` points onto a test image and its warped counterpart to verify that the lines appear parallel in the warped image.

![alt text][image4]

#### 4. Describe how (and identify where in your code) you identified lane-line pixels and fit their positions with a polynomial?

Then I did some other stuff and fit my lane lines with a 2nd order polynomial kinda like this:

![alt text][image5]

#### 5. Describe how (and identify where in your code) you calculated the radius of curvature of the lane and the position of the vehicle with respect to center.

I did this in lines # through # in my code in `my_other_file.py`

#### 6. Provide an example image of your result plotted back down onto the road such that the lane area is identified clearly.

I implemented this step in lines # through # in my code in `yet_another_file.py` in the function `map_lane()`.  Here is an example of my result on a test image:

![alt text][image6]

---

### Pipeline (video)

#### 1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (wobbly lines are ok but no catastrophic failures that would cause the car to drive off the road!).

Here's a [link to my video result](./project_video.mp4)

---

### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?
