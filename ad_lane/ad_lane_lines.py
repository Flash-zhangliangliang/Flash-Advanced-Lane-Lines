import numpy as np
import cv2
import glob
import pickle
import matplotlib.image as mpimg


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


def cal_undistorted(img, mtx, dist):
    # 根据系数校正图像
    # returns the undistorted image
    undistorted = cv2.undistort(img, mtx, dist, mtx)
    return undistorted


# Define a function that applies Sobel x or y,
# then takes an absolute value and applies a threshold.
# Note: calling your function with orient='x', thresh_min=5, thresh_max=100
# should produce output like the example image shown above this quiz.
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


# Define a function that applies Sobel x and y,
# then computes the magnitude of the gradient
# and applies a threshold
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


# Define a function that applies Sobel x and y,
# then computes the direction of the gradient
# and applies a threshold.
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


def r_select(img, thresh=(200, 255)):
    r = img[:, :, 0]
    binary = np.zeros_like(r)
    binary[(r > thresh[0]) & (r <= thresh[1])] = 1
    return binary


def yellow_white_mask(img,
                      yellow_hsv_low=np.array([0, 100, 100]),
                      yellow_hsv_high=np.array([80, 255, 255]),
                      white_hsv_low=np.array([0, 0, 160]),
                      white_hsv_high=np.array([255, 80, 255])):
    image_hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    mask_yellow = cv2.inRange(image_hsv, yellow_hsv_low, yellow_hsv_high)
    mask_white = cv2.inRange(image_hsv, white_hsv_low, white_hsv_high)
    mask_yw_image = cv2.bitwise_or(mask_yellow, mask_white)
    return mask_yw_image


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


def combine_filters(img):
    grad_x = abs_sobel_thresh(img, orient='x', sobel_kernel=3, thresh=(20, 255))
    l_binary = hls_select(img, channel='L', thresh=(100, 200))
    s_binary = hls_select(img, channel='S', thresh=(100, 255))
    yw_binary = yellow_white_mask(img)
    yw_binary[(yw_binary != 0)] = 1
    combined_lsx = np.zeros_like(grad_x)
    combined_lsx[((l_binary == 1) & (s_binary == 1) | (grad_x == 1) | (yw_binary == 1))] = 1
    return combined_lsx


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


# Generate x and y values for plotting
def get_fit_xy(img, left_fit, right_fit):
    plot_y = np.linspace(0, img.shape[0]-1, img.shape[0])
    left_fit_x = left_fit[0] * plot_y ** 2 + left_fit[1] * plot_y + left_fit[2]
    right_fit_x = right_fit[0] * plot_y ** 2 + right_fit[1] * plot_y + right_fit[2]
    return left_fit_x, right_fit_x, plot_y


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


def image_add(wrap_img, origin_img, left_fit_x, right_fit_x, plot_y, m_back):
    warp_zero = np.zeros_like(wrap_img).astype(np.uint8)
    color_warp = np.dstack((warp_zero, warp_zero, warp_zero))

    # Recast the x and y points into usable format for cv2.fillPoly()
    pts_left = np.array([np.transpose(np.vstack([left_fit_x, plot_y]))])
    pts_right = np.array([np.flipud(np.transpose(np.vstack([right_fit_x, plot_y])))])
    pts = np.hstack((pts_left, pts_right))

    # Draw the lane onto the warped blank image
    cv2.fillPoly(color_warp, np.int_([pts]), (0, 0, 255))

    # Warp the blank back to original image space using inverse perspective matrix
    img_size = (color_warp.shape[1], color_warp.shape[0])  # 先是宽度，再是高度
    new_warp = cv2.warpPerspective(color_warp, m_back, img_size)
    # Combine the result with the original image
    result = cv2.addWeighted(origin_img, 1, new_warp, 0.3, 0)
    return result


def draw_values(img, curvature, distance_from_center):
    font = cv2.FONT_HERSHEY_SIMPLEX
    radius_text = "Radius of Curvature: %sm" % (round(curvature))
    cv2.putText(img, radius_text, (100, 100), font, 2, (255, 255, 255), 2)

    if distance_from_center > 0:
        pos_flag = 'right'
    else:
        pos_flag = 'left'

    center_text = "Vehicle is %.3fm %s of center" % (abs(distance_from_center), pos_flag)
    cv2.putText(img, center_text, (100, 150), font, 2, (255, 255, 255), 2)
    return img


def advanced_lane_pip(test_image):
    file_para = open('calibration_parameters.pkl', 'rb')
    dist_pickle = pickle.load(file_para)
    mtx = dist_pickle['mtx']
    dist = dist_pickle['dist']
    file_para.close()

    origin_img = mpimg.imread(test_image)

    test_img = cal_undistorted(origin_img, mtx, dist)

    file_para = open('perspective_parameters.pkl', 'rb')
    dist_pickle = pickle.load(file_para)
    m = dist_pickle['M']
    m_back = dist_pickle['M_back']
    file_para.close()

    img_size = (test_img.shape[1], test_img.shape[0])  # 先是宽度，再是高度
    wrap_img = cv2.warpPerspective(test_img, m, img_size)

    binary = combine_filters(wrap_img)

    left_fit, right_fit, out_img = find_line_fit(binary)
    left_fit_x, right_fit_x, plot_y = get_fit_xy(binary, left_fit, right_fit)

    curvature, distance_from_center = cal_cur_and_pos(binary, left_fit_x, right_fit_x, plot_y)

    result = image_add(binary, test_img, left_fit_x, right_fit_x, plot_y, m_back)
    result = draw_values(result, curvature, distance_from_center)

    return result


