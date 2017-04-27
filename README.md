## Advanced Lane Finding
[![Udacity - Self-Driving Car NanoDegree](https://s3.amazonaws.com/udacity-sdc/github/shield-carnd.svg)](http://www.udacity.com/drive)

Overview:
---

In this project, I've written a software pipeline to identify the lane boundaries in a video. The steps include, camera calibration, transforming images to HLS and taking the S binary output, merging it with gradient along the horizontal axis. Then applying sliding window algorithm to get the initial lane detection. Then using it to find the lane lines in subsequent frames. Applying lot of sanity checks like lane lines being parallel, and radius of curvature matching and so on. 

The Project
---

The goals / steps of this project are the following:

* Compute the camera calibration matrix and distortion coefficients given a set of chessboard images.
* Apply a distortion correction to raw images.
* Use color transforms, gradients, etc., to create a thresholded binary image.
* Apply a perspective transform to rectify binary image ("birds-eye view").
* Detect lane pixels and fit to find the lane boundary.
* Determine the curvature of the lane and vehicle position with respect to center.
* Warp the detected lane boundaries back onto the original image.
* Output visual display of the lane boundaries and numerical estimation of lane curvature and vehicle position.

The detailed project writeup is at:
---
[writeup](/writeup.pdf)

The IPython notebook having the entire code is at :
---
[advanced_lane_finding.ipynb](/advanced_lane_finding.ipynb)

The HTML version of the notebook: 
[html of notebook](/advanced_lane_finding.html)

The output (lane marked) of test images: 
[lane marked test images](/output_images)

Lane marked Video output:
---

[lane marked video](/video_lane_marked.mp4)

Challenge Lane marked Video output:
---

[Challenge lane marked video](/challenge_lane_marked.mp4)


