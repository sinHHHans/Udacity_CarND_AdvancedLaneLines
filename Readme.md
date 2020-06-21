## Writeup


**Advanced Lane Finding Project**

![][Intro]


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

[calibration_image]: ./writeup_images/Calib.png "Calibration Image"
[undistorted_calibration_image]: ./writeup_images/Calib_Done.png "Calibrated Image"

[test_image]: ./writeup_images/test6.png "Test Image"
[undistorted_test_image]: ./writeup_images/test6_undistorted.png "Test image undistorted"
[test_image_diff]: ./writeup_images/diff_image_distortion.png "Diff image"

[3d_edges_raw]: ./writeup_images/lane_edges_3d.png "Lane 3D"
[3d_edges_scaled]: ./writeup_images/lane_edges_3d_scaled.png "Lane 3D scaled"
[scaling_plot]: ./writeup_images/Scaling_Far_Intensities.PNG "Scaling"
[overview_edges_scaling]: ./writeup_images/OverviewEdgesScaling.png "Scaling Overview"
[overview_edges]: ./writeup_images/OverviewEdges.png "Edges Overview"
[Overview_S_Channel]: ./writeup_images/OverviewSChannel.png "S-Channel Overview"
[Overview_H_Channel]: ./writeup_images/OverviewHChannel.png "H-Channel Overview"

[Radon_good]: ./writeup_images/Radon_good.png "Radon Good"
[Radon_bad]: ./writeup_images/Radon_bad.png "Radon Bad"

[OverviewFiltered]: ./writeup_images/OverviewFiltered.png "Overview filtered"

[Transformations]: ./writeup_images/Transformation.png "BEV Samples"

[half_image]: ./writeup_images/lanes_test_half.png "Half"
[half_image_hist]: ./writeup_images/lanes_test_half_hist.png "Half hist"

[Sliding_window]: ./writeup_images/Sliding_Window.png "Sliding window"
[result]: ./writeup_images/Result.png "Result"
[Radius]: ./writeup_images/Radius.PNG "Radius"
[Intro]: ./writeup_images/Intro.PNG "Radius"
[Test1]: ./writeup_images/test_done.PNG "Test"
[Test2]: ./writeup_images/test2_done.PNG "Test"
[Test3]: ./writeup_images/test3_done.PNG "Test"




## [Rubric](https://review.udacity.com/#!/rubrics/571/view) Points

### Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  

---

### Writeup

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one.  You can submit your writeup as markdown or pdf.  [Here](https://github.com/udacity/CarND-Advanced-Lane-Lines/blob/master/writeup_template.md) is a template writeup for this project you can use as a guide and a starting point.  

You're reading it!

### Camera Calibration

#### 1. Briefly state how you computed the camera matrix and distortion coefficients. Provide an example of a distortion corrected calibration image.

I calibrate the camera using all the provided calibration images. I do the whole processing in the function 
``` python
def calibrate_camera(path_to_calibration_images):
# ...
 ```
in my Notebook, "Pipeline.ipynb"
 As the calibration Chessboards are 9x6, I generate a grid of that size in numpy:
 
 ``` python
     objp= np.zeros((9*6,3),np.float32)
     objp[:,:2] = np.mgrid[0:9, 0:6].T.reshape(-1,2)
```
These are my world points in x,y,z=0. Then for each image I try to find the chessboard corners using openCV and refine them.

``` python
    ret, corners = cv2.findChessboardCorners(gray, (9,6), None)
    # ...
    if ret:
        corners2 = cv2.cornerSubPix(gray,corners,(11,11),(-1,-1),criteria)
```

After iterating all the calibration images, I create the camera parameters by calling:

``` python
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1],None,None)
```

A calibration looks like this:

![][calibration_image]

The same image but undistorted looks like this: 

![][undistorted_calibration_image]

### Pipeline (single images)

#### 1. Provide an example of a distortion-corrected image.

To demonstrate this step, I will describe how I apply the distortion correction to one of the test images like this one:

![][test_image]

The result looks like this:

![][undistorted_test_image]

Differences can be seen when comparing the trees and the hood. To get a better understanding of the difference, here is a
diff-image between distorted and undistorted:

![][test_image_diff]

It can be seen, that the effect of the camera lense is larger at the outer areas 
and smaller, actually zero, in the center. The circular nature of the lense can be seen as well.

#### 2. Describe how (and identify where in your code) you used color transforms, gradients or other methods to create a thresholded binary image.  Provide an example of a binary image result.

I try many approaches to separate lanes lanes in the images. All of this happens in the find_lanes()-function.
* Find edges using a sobel on x-direction with increased intensity in the background

``` python
    image_edge = scale_distant_values(sobelx(as_gray(image_bev), ksize=11))
    thresh_edge = (25, 255)

    image_edge = scale_distant_values(sobelx(as_gray(image_bev), ksize=11))
    image_edge = smoothen(image_edge)
    image_edge //= ((np.max(image_edge) // 255))
    binary_image_edge = np.zeros_like(image_edge)
    binary_image_edge[(image_edge > thresh_edge[0]) & (image_edge <= thresh_edge[1])] = 1
    binary_image_edge_opened = as_opened(binary_image_edge)
    binary_image_edge_opened_closed = as_closed(binary_image_edge_opened, iterations=15).astype("uint8")
```
Description:
First I convert the image, which was already projected into Birds Eye View(BEV) to a grayscale image. Then I try to find edges in the X-Direction.

Afterwards, I scale the result by calling scale_distant_values().It enhances the pixels that are in the top of the image (far away), since the intensity is not as high as the ones in the bottom(close to the camera).

``` python 
def scale_distant_values(image, factor=3.5, order=2):
    max_val = np.max(image)
    dtype = image.dtype
    op_image = np.copy(image)
    for row in list(range(op_image.shape[0])):
        scaler = 1 + factor*(((op_image.shape[0]-row)/op_image.shape[0])**order)
        op_image[row, :] = np.minimum((op_image[row, :] * scaler), max_val)
    return op_image
```

A short explanation of this:
This function uses a quadratic function. The principle is explained below:
When looking at a 3D plot of an edge detected image, the problem becomes clear:

![][3d_edges_raw]

The edges far away have less intensity. The close parts of the detected lanes, which are the black peaks, are more intense.
So the image is scaled in the way as shown in the following plot:

![][scaling_plot]

The y axis is the scaling factor, in relation to the image row shown as x-axis. The result s shown here:

![][3d_edges_scaled]

An overview for the test images can be seen below:

![][overview_edges_scaling]

Back to the edge detection. After scaling up the far away pixels in the detected edges, I smoothen the image to have a more
connected area when transfering into a binary image. I simply call     
``` python 
    image_edge = smoothen(image_edge)
```
 which effectively averages the image with a 9x9 kernel in a non gaussian way. After that, I normalize the values to be between 0 and 255 (uint8)
 Afterwards I create a new image of the same size and create the binary information, according to the threshold defined above.
 
 The following two lines increase the quality of the binary image a bit.
 ``` python
    binary_image_edge_opened = as_opened(binary_image_edge)
    binary_image_edge_opened_closed = as_closed(binary_image_edge_opened, iterations=15).astype("uint8")
 ```
 
 Closing reduces single pixels, as introduces by noise and bumps on the road surface. the closing operation fill small gaps within the remaining lanes.
 
 The whole pipeline for the edges approach can be found in the following overview image:
 
 ![][overview_edges]

* Transfering into HLS color space and filtering for the S-Channel.

I change the colorspace by calling:
``` python
    # Color transormation
    image_hls = cv2.cvtColor(image_bev, cv2.COLOR_RGB2HLS)
    image_s = image_hls[:,:,2]
    image_h = image_hls[:,:,0]
```

Then I create a binary image of the S-Channel. Here the explanation in in the code.

``` python
    # Set the threshold for binary selection for anything above 100.
    thresh_s = (100, 255) 
    
    # Create an empty image with correct size
    binary_image_s = np.zeros_like(image_s)
     
    # Do the actual binary selection
    binary_image_s[(image_s > thresh_s[0]) & (image_s <= thresh_s[1])] = 1
    
    # Perform an "Opening" operation. This reduces single images in the image
    binary_image_s_opened = as_opened(binary_image_s)
    
    # Perform a lot of closing, to aggregate scattered data. This accumulates the lane lines again
    binary_image_s_opened_closed = as_closed(binary_image_s_opened, iterations=15)
```


See the overview image on the effects of this:

![][Overview_S_Channel]

* Transfering into HLS color space and filtering for the S-Channel with increased intensity in the background.

I do the same from above, but with an S-Channel representation, whose far pixels are scaled up again by scale_distant_values.
The effect can be seen in the overview image above. It can be seen, that in many cases it really makes a big difference.
It is however important to do a good filtering afterwards.

* Transfering into HLS color space and filtering for the H-Channel.
This works just like the S-Chanenel part, but uses another Threshold.
Namely:
``` python
    thresh_h = (15, 100)
```

The result can be seen in the following overview:

![][Overview_H_Channel]


* Combining all the above into one image, selecting by a Radon Transform based approach.

As could be found in the above overviews, each result has a dedicated radon transformation image. 
Here comes the more tricky part of my pipeline.

An image that shows clear lanes has a very deterministic radon transformation, as there are points that lie on straight lines.

![][radon_good]

In an image that has a less good result in terms of lane extraction has a totally different radon transformation, as can be seen in this image:

![][radon_bad]

The shape of the waves in the image and their number contains all of the information of the original image. 
For simplicity though, I only consider the peaks, because they need to be very close the the edges if lanes 
are depicted from BEV. 

I use this knowledge for determining if an intermediate result, e.g. the Sobel edges provide a good estimate for lanes.
The logic for this is, that if at least one peak exists detached from the borders, the image is not a good estimate and will further on be ignored.

In the overall overview I marked the ones that were removed by this logic.


![][OverviewFiltered]

All the other images that remain are then added up, to form a "combo"-image. These can be found on the right in the above overview.


Note that the combo is not binary anymore, as pixels that occur in more than one image are 
added up and generate a higher impact, which is what I want because the more images show a pixel, the 
more likely it is that this pixel is actually a lane.

#### 3. Describe how (and identify where in your code) you performed a perspective transform and provide an example of a transformed image.

For transforming the image into BEV, I perform a perspective transform. This is done twice. Once in the beginning of detecting the lanes, and once again in the end, when I project my
solution for the lane boundaries back into the original image. A sequence of this can be found in the following:

![][Transformations]

I wrapped this function in the following:

``` python
def transform_BEV(image_undistorted, img_size, offset, inverse=False):
    """
    This function transforms an image into a Birds Eye View(BEV). The raw image that is provided should already be undistorted.
    It returns a BEV representation of the image.
    
    :param image_undistorted: The image that shall be transformed 
    :param img_size: The size of the source image
    :param offset: an offset to the borders. This part will be omitted in the destination image
    :param inverse: If true, the function performs the inverse transformation from BEV into normal perspective.
    
    :return: Transformed image
    """
    src = trapezoid = np.float32([(569,450),(714,450),(1200,650), (100,650)])
    dst = np.float32([[offset, offset], 
                      [img_size[0]-offset, offset], 
                      [img_size[0]-offset, img_size[1]-offset], 
                      [offset, img_size[1]-offset]])
    
    # Swap if inverse is wanted
    if inverse:
        # print(src[0], dst[0])
        temp = np.copy(src)        
        src = dst
        dst = temp
        
        img_size = tuple(reversed(img_size))
        # print(src[0], dst[0])
        
    M = cv2.getPerspectiveTransform(src, dst)
    # print(img_size)
    return cv2.warpPerspective(image_undistorted, M, img_size)

```
The source points are selected by hand, so it returns a good estimate on the road surface. 
If `` inverse==True ``, the function actually returns the inverse transformation. 
I tested this by seeing that the BEV image has parallel lines in the test images of the straight driving.

#### 4. Describe how (and identify where in your code) you identified lane-line pixels and fit their positions with a polynomial?

Starting from the combined image from above, I try to find the starting points of the lanes. I do this by taking the sum of the columns of the lower half of the combined image.
Pixels that were detected by many images account more into the resulting peaks in a histogram. The histogram and the respective half image can be found below:

![alt text][half_image]

![alt text][half_image_hist]


The code for this can be found in ``find_lane_start()``. That function is called by ``find_lane_pixels()``, a function which classifies pixels in an image to two lanes, a left and a right one.

The logic on a a high level is shown in the following pseudo code:

``` 
find_lanes():
    create combo-image
    search_around_poly():
        if no previous solution exists:
            fit_polynomial():
                find_lane_pixels():
                    find_lane_start()
                    follow found starting points
                    return: lane_pixels
                fit the found pixels to a 2nd order polynomial
        else if previous solution exists:
             fit pixels close to previous solution to 2nd order polynomial
        
        return: found polynomials

    lanes = found polynomials
    return lanes    
  
```
The procedure of following lane starts with moving windows can best be seen in the following image. 

*Note: The yellow lines can be ignored here, they have no meaning*

![alt text][Sliding_window]

The windows are set sequentially starting from the bottom and are centered at the center of the number of pixels. Once this is done,
the pixels can be used to find a polynomial per lane line.
Once these polynomials are found by either searching the pixels from start, or following an already found solution, the lanes are known.

A solution look as follows:

![alt text][result]

Additionally I do a sanity check, in order to validate if the found solution makes sense. I do this by taking three x values per lane line.
For one I calculate the pixel distance between them in order to check if the distance is approximately that of a lane on the road. Additionally I compare the standard
deviation of the distances between the three points, as this is a measure of parallelism. A high std indicates that the found lanes are not parallel.

```` python
def lane_sanity_filter(lanes, last_lanes):
    """
    This function checks if found lanes are reasonable. 
    If yes, it returns a True and the found lanes.
    If no, it returns a FAlse and the previous lanes
    
    :param lanes: Found lanes
    :param last_lanes: A deque of the last few lanes
    
    :return: boolean if all is fine, and a set of last lanes
    """
    # ...
````

#### 5. Describe how (and identify where in your code) you calculated the radius of curvature of the lane and the position of the vehicle with respect to center.

This is done in two dedicated functions. First I find the position of the vehicle in ```get_position() ``` For that, I compare the middle of the two lanes with the middle of the image. 
I approximate the ratio of pixels to meters as 7.5 meters per 720 pixels and can then calculate the offset in meters. I return the offset in meters and the offset between -1 and +1.

The normed offset is helpful for calculating the radius of the road better, which is done in ```get_radius() ```.

I calculate the radius by the formula from the lecture for each lane: 

![alt text][Radius]

Then I calculate a weighted mean, considering the lane which the vehicle is closer to more, and the other one less.

#### 6. Provide an example image of your result plotted back down onto the road such that the lane area is identified clearly.

The complete pipeline is applied to each video image in the following code.

```` python 
# Preprocessing, independant of individual frame
lanes_ring_buffer = collections.deque(maxlen=1)
camera_params = calibrate_camera("camera_cal/")
font = cv2.FONT_HERSHEY_SIMPLEX 
np.set_printoptions(precision=2)

def process_image(image):
    
    # Undistort
    image_undist = correct_lense(image, camera_params)
    
    # BEV transformation
    image_bev = transform_BEV(image_undist, img_size=(image.shape[0], image.shape[1]), offset=50)
    
    # Find lanes. _ are the intermediate images for testing, omitted here
    lanes, _ = find_lanes(image_bev, lanes_ring_buffer)
   
    # Apply the last found lane to the ring buffer, so it can be used in the next frame as a basis 
    lanes_ring_buffer.append(lanes)
    
    # Draw the lanes
    background = np.zeros_like(image_bev[:,:,0]).astype(np.uint8)
    background = np.dstack((background, background, background))  
    draw_lanes_on_bev(background, lanes)

    # Transform the found lanes back to original perspective. Note that inverse is set to True here
    back_trafo = transform_BEV(background, img_size=imsize, offset=50, inverse=True)
    
    # Combine the result with the original image
    back_trafo = back_trafo.reshape((image.shape[0], image.shape[1], 3))
    result = cv2.addWeighted(image, 1, back_trafo, 0.3, 0)
    
    # Calculate radius and position and print on output image
    offset_normed, offset_meters = get_position(lanes, image.shape[0] // 2)

    if offset_normed is not None:
        radius = get_radius(lanes, offset_normed)
    
        radius_str = "Radius of road = {:.2f}".format(radius) + "m"
        position_str = "Vehicle is "
        if offset_meters > 0:
            position_str += ("{:.2f}".format(abs(offset_meters)) + " to the right of center")
        else:
            position_str += ("{:.2f}".format(abs(offset_meters)) + " to the left of center")
    else:
        radius_str = "No lanes found!"
        position_str = "No lanes found!"

    cv2.putText(result, radius_str,(40,60), font, 1.5,(255,255,255),2,cv2.LINE_AA)
    cv2.putText(result, position_str,(40,140), font, 1.5,(255,255,255),2,cv2.LINE_AA)
    
    return result
````

Examples on test images are as follows.

![][Test1]

![][Test2]

![][Test3]


### Pipeline (video)

#### 1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (wobbly lines are ok but no catastrophic failures that would cause the car to drive off the road!).

Here's a [link to my video result](./project_video.mp4)

---

### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

Due to the Radon Transformation approach I took, I feel confident in finding lane estimates in a varying range of lighting and colors where 
shadows are involved. The major drawback here is, that the radon transformation is computationally very expensive and therefore my solution
is in no means capable if performing in real-time. I tried to do the same with a Fast-Fourier-Transformation(fft2). I think this is also possible, and
much faster, however the classification is tougher.

Another drawback is, that I do not yet use the added pixels after finding the starting points, but instead just count the non-zero pixels.
This is obviously the default approach from the lecture, and I think this can be increased by a lot 
if I use the valuable information of my combination image. Also I could use many more approaches to finding lane guesses.
