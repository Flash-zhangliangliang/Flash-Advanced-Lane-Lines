## Writeup of Flash( Zhang Liangliang )

---

**Advanced Lane Finding Project**

The goals / steps of this project are the following:

* Compute the camera calibration matrix and distortion coefficients given a set of chessboard images.
* Apply a distortion correction to raw images.
* Use color transforms, gradients, etc., to create a thresholded binary image.
* Apply a perspective transform to rectify binary image ("birds-eye view").
* Detect lane pixels and fit to find the lane boundary.
* Determine the curvature of the lane and vehicle position with respect to center.
* Warp the detected lane boundaries back onto the original image.
* Output visual display of the lane boundaries and numerical estimation of lane curvature and vehicle position.

[//]: # (Image References)

[image1]: ./writeup_images/undistorted_image.png "Undistorted"
[image2]: ./writeup_images/test2.jpg "Road Transformed"
[image3]: ./writeup_images/sobel_x_filter.png "sobel_x_filter"
[image4]: ./writeup_images/mag_threshold.png "mag_threshold"
[image5]: ./writeup_images/dir_threshold.png "dir_threshold"
[image6]: ./writeup_images/s_channal_filter.png "s_channal_filter"
[image7]: ./writeup_images/combine_filters.png "combine_filter"
[image8]: ./writeup_images/warp_perspective_image.png "warp_perspective_image"
[image9]: ./writeup_images/find_line_fit.png "find_line_fit"
[image10]: ./writeup_images/resault.png "resault"

[video1]: ./output_video/project_video_flash.mp4 "Video"

## [Rubric](https://review.udacity.com/#!/rubrics/571/view) Points

### Here I will consider the rubric points individually and describe how I addressed each point in my implementation.

---

### Camera Calibration

#### 1. how i computed the camera matrix and distortion coefficients.
The code for this step is contained in 8 line of file called `ad_lane_lines.py`.

```
def calibrate_camera(nx, ny, cal_images_path):
    # prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
    obj_p = np.zeros((ny * nx, 3), np.float32)
    obj_p[:, :2] = np.mgrid[0:nx, 0:ny].T.reshape(-1, 2)

    # Arrays to store object points and image points from all the images.
    obj_points = []  # 3d points in real world space
    img_points = []  # 2d points in image plane.

    # Make a list of calibration images
    images = glob.glob(cal_images_path)
    img_size = None
    # Step through the list and search for chessboard corners
    for f_name in images:
        img = cv2.imread(f_name)
        if img_size is None:
            img_size = (img.shape[1], img.shape[0])
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Find the chessboard corners
        ret, corners = cv2.findChessboardCorners(gray, (nx, ny), None)

        # If found, add object points, image points
        if ret is True:
            obj_points.append(obj_p)
            img_points.append(corners)

            # Draw and display the corners
            cv2.drawChessboardCorners(img, (nx, ny), corners, ret)
            cv2.imshow('img', img)
            cv2.waitKey(500)

    cv2.destroyAllWindows()

    if img_size is not None:
        ret, mtx, dist, r_vec_s, t_vec_s = cv2.calibrateCamera(obj_points, img_points, img_size, None, None)
    else:
        dist = None
        mtx = None

    return mtx, dist
```

I start by preparing "object points", which will be the (x, y, z) coordinates of the chessboard corners in the world. Here I am assuming the chessboard is fixed on the (x, y) plane at z=0, such that the object points are the same for each calibration image.  Thus, `objp` is just a replicated array of coordinates, and `objpoints` will be appended with a copy of it every time I successfully detect all chessboard corners in a test image.  `imgpoints` will be appended with the (x, y) pixel position of each of the corners in the image plane with each successful chessboard detection.

I then used the output `objpoints` and `imgpoints` to compute the camera calibration and distortion coefficients using the `cv2.calibrateCamera()` function.  I applied this distortion correction to the test image using the `cv2.undistort()` function and obtained this result:

![alt text][image1]

At last I saved the camera calibration parameters in file `calibration_parameters.pkl`, and for next time I can read the parameters directly without run functions.

### Pipeline (single images)

#### 1. Provide an example of a distortion-corrected image.

To demonstrate this step, I will describe how I apply the distortion correction to one of the test images like this one:
![alt text][image2]

#### 2. Describe how (and identify where in your code) you used color transforms, gradients or other methods to create a thresholded binary image.  Provide an example of a binary image result.

I used a combination of color and gradient thresholds to generate a binary image (thresholding steps at lines 58 - 197 in `al_lane_lines.py`).

First I use the x abs_sobel_thresh to generate a binary image:

```
def abs_sobel_thresh(img, sobel_kernel=3, orient='x', thresh=(0, 255)):
    # Apply the following steps to img
    # 1) Convert to gray scale
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    # 2) Take the derivative in x or y given orient = 'x' or 'y'
    if orient == 'x':
        sobel = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
    elif orient == 'y':
        sobel = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel)
    # 3) Take the absolute value of the derivative or gradient
    abs_sobel = np.absolute(sobel)
    # 4) Scale to 8-bit (0 - 255) then convert to type = np.uint8
    scaled_sobel = np.uint8(255 * abs_sobel / np.max(abs_sobel))
    # 5) Create a mask of 1's where the scaled gradient magnitude
    # is > thresh_min and < thresh_max
    binary_output = np.zeros_like(scaled_sobel)
    binary_output[(scaled_sobel >= thresh[0]) & (scaled_sobel <= thresh[1])] = 1
    # 6) Return this mask as your binary_output image
    # binary_output = gray # Remove this line
    return binary_output
```
```
gradx = al.abs_sobel_thresh(wrap_img, orient='x', sobel_kernel=3, thresh=(10, 230))
```
And I get a result look like this:

![alt text][image3]

It seems that it lose track of the lane line where the road color and the line color are light.

Then I use the magnitude threshholds to see how well it does to capture the lane line:

```
def mag_thresh(img, sobel_kernel=3, thresh=(0, 255)):
    # Apply the following steps to img
    # 1) Convert to gray scale
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    # 2) Take the gradient in x and y separately
    sobel_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
    sobel_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel)
    # 3) Calculate the magnitude
    grad_mag = np.sqrt(sobel_x ** 2 + sobel_y ** 2)
    # 4) Scale to 8-bit (0 - 255) and convert to type = np.uint8
    scale_factor = np.max(grad_mag) / 255
    grad_mag = (grad_mag / scale_factor).astype(np.uint8)
    # 5) Create a binary mask where mag thresholds are met
    binary_output = np.zeros_like(grad_mag)
    binary_output[(grad_mag >= thresh[0]) & (grad_mag <= thresh[1])] = 1
    # 6) Return this mask as your binary_output image
    # binary_output = np.copy(img) # Remove this line
    return binary_output
```

```
mag_binary = al.mag_thresh(wrap_img, sobel_kernel=9, thresh=(50, 100))
```

And I get a result look like this:

![alt text][image4]

Still not capable of capture the lane line where both the road and lane color are light.

So I turn to direction threshholds:

```
def dir_threshold(img, sobel_kernel=3, thresh=(0, np.pi / 2)):
    # Apply the following steps to img
    # 1) Convert to gray scale
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    # 2) Take the gradient in x and y separately
    sobel_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
    sobel_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel)
    # 3) Take the absolute value of the x and y gradients
    abs_sobel_x = np.absolute(sobel_x)
    abs_sobel_y = np.absolute(sobel_y)
    # 4) Use np.arctan2(abs_sobel_y, abs_sobel_x) to calculate the direction of the gradient
    grad_dir = np.arctan2(abs_sobel_y, abs_sobel_x)
    # 5) Create a binary mask where direction thresholds are met
    binary_output = np.zeros_like(grad_dir)
    binary_output[(grad_dir >= thresh[0]) & (grad_dir <= thresh[1])] = 1
    # 6) Return this mask as your binary_output image
    # binary_output = np.copy(img) # Remove this line
    return binary_output
```

```
dir_binary = al.dir_threshold(wrap_img, sobel_kernel=21, thresh=(0.7, 1.3))
```
Here is the result I got:
![alt text][image5]

It seems a bit too much blur.

Seems gradient thresholds are not capable to handle this situation

How about color thresholds:

```
def hls_select(img, channel='S', thresh=(90, 255)):
    # 1) Convert to HLS color space
    # 2) Apply a threshold to the S channel
    # 3) Return a binary image of threshold result
    hls = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
    if channel == 'S':
        x = hls[:, :, 2]
    elif channel == 'H':
        x = hls[:, :, 0]
    elif channel == 'L':
        x = hls[:, :, 1]
    else:
        print('illegal channel !!!')
        return
    binary_output = np.zeros_like(x)
    binary_output[(x > thresh[0]) & (x <= thresh[1])] = 1
    return binary_output
```

```
l_binary = al.hls_select(wrap_img, channel='L', thresh=(180, 255))
s_binary = al.hls_select(wrap_img, channel='S', thresh=(180, 255))
h_binary = al.hls_select(wrap_img, channel='H', thresh=(0, 255))
```

And here is the result I got:
![alt text][image6]

The s color channel did a great job on capature the lane where both lane and road color are light.
But still it lose track of the lane on some place.

It seems difference thresholds capature lane line under difference situation.
So I used a combination of color and gradient thresholds to generate a binary image at the end.

```
def luv_select(img, thresh=(0, 255)):
    luv = cv2.cvtColor(img, cv2.COLOR_RGB2LUV)
    l_channel = luv[:, :, 0]
    binary_output = np.zeros_like(l_channel)
    binary_output[(l_channel > thresh[0]) & (l_channel <= thresh[1])] = 1

    return binary_output


def lab_select(img, thresh=(0, 255)):
    lab = cv2.cvtColor(img, cv2.COLOR_RGB2Lab)
    b_channel = lab[:, :, 2]
    binary_output = np.zeros_like(b_channel)
    binary_output[(b_channel > thresh[0]) & (b_channel <= thresh[1])] = 1

    return binary_output


def combine_filters(img):
    grad_x = abs_sobel_thresh(img, orient='x', sobel_kernel=3, thresh=(10, 230))
    mag = mag_thresh(img, sobel_kernel=3, thresh=(30, 150))
    dir_thresh = dir_threshold(img, sobel_kernel=3, thresh=(0.7, 1.3))
    hls_thresh = hls_select(img, thresh=(180, 255))
    lab_thresh = lab_select(img, thresh=(155, 200))
    luv_thresh = luv_select(img, thresh=(225, 255))
    combined_lsx = np.zeros_like(grad_x)
    combined_lsx[((grad_x == 1) & (mag == 1)) | ((dir_thresh == 1) & (hls_thresh == 1)) |
                 (lab_thresh == 1) | (luv_thresh == 1)] = 1

    return combined_lsx
```

```
binary = al.combine_filters(wrap_img)
```

And after try a few difference pramemter and combination, I finaly got the result like this:

![alt text][image7]


#### 3. Describe how (and identify where in your code) you performed a perspective transform and provide an example of a transformed image.

The code for perspective transform is in the line 9-23 of file `Step_by_Step.ipynb`

I chose to hardcode the source and destination points to calculate the transform matrix:

```python
trape = np.array([[232, 700], [596, 450], [685, 450], [1078, 700]], np.int32)
trape = trape.reshape(-1, 1, 2)
src = np.float32(np.reshape(trape, (-1, 2)))

dst = np.float32([[350, 700], [350, 0], [950, 0], [950, 700]])
```

This resulted in the following source and destination points:

| Source        | Destination   |
|:-------------:|:-------------:|
| 232, 700      | 350, 700      |
| 596, 450      | 350, 0        |
| 685, 450      | 950, 0        |
| 1078, 700     | 950, 700      |

I verified that my perspective transform was working as expected by drawing the `src` and `dst` points onto a test image and its warped counterpart to verify that the lines appear parallel in the warped image.

```
img_size = (img.shape[1], img.shape[0])
M = cv2.getPerspectiveTransform(src, dst)
M_back = cv2.getPerspectiveTransform(dst, src)
```

![alt text][image8]


#### 4. Describe how (and identify where in your code) you identified lane-line pixels and fit their positions with a polynomial?

The code for identified lane-line pixels is in the line 200-272 of file `ad_lane_lines.py`

I use the  Peaks in a Histogram method to identified the x position of the lane lines in binary_wraped image.

Then I will use the sliding window to indentified the pixel that belong to the line:

```
def find_line_fit(img, n_windows=9, margin=100, min_pix=50):
    histogram = np.sum(img[img.shape[0] // 2:, :], axis=0)
    # Create an output image to draw on and  visualize the result
    out_img = np.dstack((img, img, img)) * 255
    # Find the peak of the left and right halves of the histogram
    # These will be the starting point for the left and right lines
    midpoint = np.int(histogram.shape[0] / 2)
    left_x_base = np.argmax(histogram[:midpoint])
    right_x_base = np.argmax(histogram[midpoint:]) + midpoint

    # Set height of windows
    window_height = np.int(img.shape[0] / n_windows)
    # Identify the x and y positions of all nonzero pixels in the image
    nonzero = img.nonzero()
    nonzero_y = np.array(nonzero[0])
    nonzero_x = np.array(nonzero[1])
    # Current positions to be updated for each window
    left_x_current = left_x_base
    right_x_current = right_x_base
    # Create empty lists to receive left and right lane pixel indices
    left_lane_ind = []
    right_lane_ind = []

    # Step through the windows one by one
    for window in range(n_windows):
        # Identify window boundaries in x and y (and right and left)
        win_y_low = img.shape[0] - (window + 1) * window_height
        win_y_high = img.shape[0] - window * window_height
        win_x_left_low = left_x_current - margin
        win_x_left_high = left_x_current + margin
        win_x_right_low = right_x_current - margin
        win_x_right_high = right_x_current + margin
        # Draw the windows on the visualization image
        cv2.rectangle(out_img, (win_x_left_low, win_y_low), (win_x_left_high, win_y_high),
                      (0, 255, 0), 2)
        cv2.rectangle(out_img, (win_x_right_low, win_y_low), (win_x_right_high, win_y_high),
                      (0, 255, 0), 2)
        # Identify the nonzero pixels in x and y within the window
        good_left_ind = ((nonzero_y >= win_y_low) &
                         (nonzero_y < win_y_high) &
                         (nonzero_x >= win_x_left_low) &
                         (nonzero_x < win_x_left_high)).nonzero()[0]
        good_right_ind = ((nonzero_y >= win_y_low) &
                          (nonzero_y < win_y_high) &
                          (nonzero_x >= win_x_right_low) &
                          (nonzero_x < win_x_right_high)).nonzero()[0]
        # Append these indices to the lists
        left_lane_ind.append(good_left_ind)
        right_lane_ind.append(good_right_ind)
        # If you found > min_pix pixels, recenter next window on their mean position
        if len(good_left_ind) > min_pix:
            left_x_current = np.int(np.mean(nonzero_x[good_left_ind]))
        if len(good_right_ind) > min_pix:
            right_x_current = np.int(np.mean(nonzero_x[good_right_ind]))

    # Concatenate the arrays of indices
    left_lane_ind = np.concatenate(left_lane_ind)
    right_lane_ind = np.concatenate(right_lane_ind)

    # Extract left and right line pixel positions
    left_x = nonzero_x[left_lane_ind]
    left_y = nonzero_y[left_lane_ind]
    right_x = nonzero_x[right_lane_ind]
    right_y = nonzero_y[right_lane_ind]

    # to plot
    out_img[nonzero_y[left_lane_ind], nonzero_x[left_lane_ind]] = [255, 0, 0]
    out_img[nonzero_y[right_lane_ind], nonzero_x[right_lane_ind]] = [0, 0, 255]

    # Fit a second order polynomial to each
    left_fit = np.polyfit(left_y, left_x, 2)
    right_fit = np.polyfit(right_y, right_x, 2)
    return left_fit, right_fit, out_img

## Generate x and y values for plotting
def get_fit_xy(img, left_fit, right_fit):
    plot_y = np.linspace(0, img.shape[0]-1, img.shape[0])
    left_fit_x = left_fit[0] * plot_y ** 2 + left_fit[1] * plot_y + left_fit[2]
    right_fit_x = right_fit[0] * plot_y ** 2 + right_fit[1] * plot_y + right_fit[2]
    return left_fit_x, right_fit_x, plot_y
```

```
left_fit, right_fit, out_img = al.find_line_fit(binary)
left_fitx, right_fitx, ploty = al.get_fit_xy(binary, left_fit, right_fit)
```

![alt text][image9]

#### 5. Describe how (and identify where in your code) you calculated the radius of curvature of the lane and the position of the vehicle with respect to center.

The code for  calculated the radius of curvature of the lane and the position of the vehicle with respect to center is in the line 283-304 of file `ad_lane_lines.py`

I difine a function that take a binary_wraped image,the left and right lane polynomial coefficients and output the radius of curvature of the lane and the position of the vehicle with respect to center.

```
def cal_cur_and_pos(binary_warped, left_fit_x, right_fit_x, plot_y):
    # Define conversions in x and y from pixels space to meters
    ym_per_pix = 30 / 720  # meters per pixel in y dimension
    xm_per_pix = 3.7 / 700  # meters per pixel in x dimension
    y_eval = np.max(plot_y)
    # Fit new polynomials to x,y in world space
    left_fit_cr = np.polyfit(plot_y * ym_per_pix, left_fit_x * xm_per_pix, 2)
    right_fit_cr = np.polyfit(plot_y * ym_per_pix, right_fit_x * xm_per_pix, 2)
    # Calculate the new radii of curvature
    left_cur = ((1 + (2 * left_fit_cr[0] * y_eval * ym_per_pix + left_fit_cr[1]) ** 2) ** 1.5) / \
                np.absolute(2 * left_fit_cr[0])
    right_cur = ((1 + (2 * right_fit_cr[0] * y_eval * ym_per_pix + right_fit_cr[1]) ** 2) ** 1.5) / \
                 np.absolute(2 * right_fit_cr[0])

    curvature = ((left_cur + right_cur) / 2)

    lane_width = np.absolute(left_fit_x[719] - right_fit_x[719])
    lane_xm_per_pix = 3.7 / lane_width
    veh_pos = (((left_fit_x[719] + right_fit_x[719]) * lane_xm_per_pix) / 2.)
    cen_pos = ((binary_warped.shape[1] * lane_xm_per_pix) / 2.)
    distance_from_center = veh_pos - cen_pos
    return curvature, distance_from_center
```

#### 6. Provide an example image of your result plotted back down onto the road such that the lane area is identified clearly.

I implemented this step in lines 307-339 in my code in `ad_lane_lines.py` in the function `image_add()` `draw_values`.  Here is an example of my result on a test image:

![alt text][image10]

---

### Pipeline (video)

#### 1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (wobbly lines are ok but no catastrophic failures that would cause the car to drive off the road!).

Here's a [link to my video result](./output_video/project_video_flash.mp4)

---

### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

The biggest problem I faced in my implementation of this project is to fine the best method in useing a combination of color and gradient thresholds to generate a binary image.

I tryed my best and looked for help from internet, and at the end I found a combination to solve the problem, just as that in this writeup.
