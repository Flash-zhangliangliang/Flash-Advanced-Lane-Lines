## Advanced Lane Finding project of Flash( Zhang Liangliang )

My goals:
---

In this project, my goals is two:

( 1 ) to write a software pipeline to identify the lane boundaries in a video;

( 2 ) to create a detailed writeup of the project.


The Project
---

The direction of my project is the following:

* There are three jupyter file:
  * Step_by_Step: Including calibrate_camera(), Testing each fuction and parameter step by step;
  * Without_Testing: No calibrate_camera(), no testing, Just main steps in my processing;
  * Pip_Line: Building my pip line.
* The calibration parameters and perspective parameters are recorded in calibration_parameters.pkl and perspective_parameters.pkl.
* writeup_flash.md is my writeup report
* All my fuctions can be found in file ad_lane/ad_lane_lines.py
* The images in camera_cal direction is used by calibrate_camera() function
* The images in output_images direction is the processing result of the images in testing_images
* The video in output_video direction is the processing result of the video in testing_video/project_video.mp4
