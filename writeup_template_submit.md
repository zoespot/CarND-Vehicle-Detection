# Udacity Self Driving Car Nano degree Term 1

## **Vehicle Detection Project**
---
The goals / steps of this project are the following:

* Performed a Histogram of Oriented Gradients (HOG) feature extraction on a labeled training set of images and train a classifier Linear SVM classifier
* Applied a color transform and append binned color features, as well as histograms of color, to HOG feature vector.
* Implemented a sliding-window technique and use your trained classifier to search for vehicles in images.
* Constructed a pipeline on the video stream, created a heat map of recurring detections frame by frame to reject outliers and smoothed the bounding boxes for detected vehicles.

---
### Writeup / README

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one.  You can submit your writeup as markdown or pdf.  

Writeup for project is [writeup-project](https://github.com/zoespot/CarND-Advanced-Lane-Lines/blob/master/writeup_template_submit.md).

Output video for project is
![output-project-video](https://github.com/zoespot/CarND-Advanced-Lane-Lines/blob/master/output_videos/output_project_video_smooth.mp4)

Code for project is [code-project](https://github.com/zoespot/CarND-Advanced-Lane-Lines/blob/master/vehicle_detect_smooth.ipynb).

### Histogram of Oriented Gradients (HOG)

#### 1. Explain how (and identify where in your code) you extracted HOG features from the training images.

The code for this step is contained in the `get_hog_features` function (in the 3rd cell of the IPython notebook).

I started by reading in all the `vehicle` and `non-vehicle` images.  Here is an example of one of each of the `vehicle` and `non-vehicle` classes with their respective HOG image:

![alt text](https://github.com/zoespot/CarND-Advanced-Lane-Lines/blob/master/output_images/car_notcar_hog.png)

I then explored different color spaces and different `skimage.hog()` parameters (`orientations`, `pixels_per_cell`, and `cells_per_block`).  I grabbed random images from each of the two classes and displayed them to get a feel for what the `skimage.hog()` output looks like.

Above plot is with `RGB` color space with `hist_bins =16`, `spatial_size =(16,16)`, and HOG parameters of `orientations=8`, `hog_channel =0 `, `pixels_per_cell=(8, 8)`, `cells_per_block=(2, 2)`, `cell_per_block =2`. Below is an example using the `YCrCb` color space and HOG parameters of `orientations=9` (other parameters are the same):

![alt text](https://github.com/zoespot/CarND-Advanced-Lane-Lines/blob/master/output_images/car_notcar_hog_YCrCb.png)

#### 2. Explain how you settled on your final choice of HOG parameters.

I tried various combinations of parameters and the final choice is as below:
```
color_space = 'YCrCb' #'RGB' #can be RGB,HSV,LUV,HLS,YUV,YCrCb
orient =9
pix_per_cell =8
cell_per_block =2
hog_channel ='ALL'
spatial_size = (16,16)
hist_bins = 16
spatial_feat =True
hist_feat =True
hog_feat = True
```
The choice is based on grid sweep of possible ranges and pick the one generates best accuracy for the linear SVM classifier.

#### 3. Describe how (and identify where in your code) you trained a classifier using your selected HOG features (and color features if you used them).

I trained a linear SVM using all the HOG and color space histograms and spatial features from **all** the `vehicle` and `non-vehicle` images.  The training and test sets are normalized and random split with 80/20%. The classifier finally achieved 99.7% accuracy.
```
200.397052526474 Seconds to compute feature...
Using: 9 orientations 8 pixels per cell and 2 cells per block
Feature vector length: 6108
28.88 Seconds to train SVC...
Test Accuracy of SVC =  0.997
```
### Sliding Window Search

#### 1. Describe how (and identify where in your code) you implemented a sliding window search.  How did you decide what scales to search and how much to overlap windows?

I decided to first limit the area of interest to the bottom half of the image, as the top half usually contain air, trees, hills, but no cars. Then I experiments different scale and overlap settings for several different images.

![alt text](https://github.com/zoespot/CarND-Advanced-Lane-Lines/blob/master/output_images/window_scale_false_negative.png)

![alt text](https://github.com/zoespot/CarND-Advanced-Lane-Lines/blob/master/output_images/window_scale2.png)


So finally I choose to combine 3 different scales in my code to generate heat map. The smaller scale searches are limited to the upper area vertically to detect far-end cars, and larger scale search in the lower area for near-end cars. I use `threshold=1` for heatmap combination. It slowed down the search by three times (0.5s per image to 1.5s per image). However it's required to cover all the cases while excluding false positive detections.

To further improve detection of false positive, **ROI is further reduced to trapezoidal road lanes area both horizontally and vertically.** As we don't need to detect cars in the reverse direction with fence in between.

```
scales =  [1.19, 1.4, 1.19]
cells_per_steps = [2,1,1]
ystarts = [340,375,400]
ystops =[532,631, 720]
xstarts =[512,480,384]
xstops =[1100,1280,1280]
heatmap_combined = np.zeros_like(img[:,:,0])

for iii in range(len(scales)):
    win_scale = scales[iii]
    cells_per_step = cells_per_steps[iii]
    ystart =ystarts[iii]
    ystop =ystops[iii]
    xstart =xstarts[iii]
    xstop =xstops[iii]
    out_img,heat_map = find_cars(img, ystart=ystart, ystop=ystop, xstart=xstart, xstop=xstop,cells_per_step=cells_per_step, scale=win_scale)
    heatmap_combined[heat_map.nonzero()]+=1
heatmap_combined= apply_threshold(heatmap_combined, threshold=1)
draw_img = draw_labeled_bboxes(np.copy(img), heatmap_combined)
```
![alt text](https://github.com/zoespot/CarND-Advanced-Lane-Lines/blob/master/output_images/window_scales_sweep_test5.png)

#### 2. Show some examples of test images to demonstrate how your pipeline is working.  What did you do to optimize the performance of your classifier?

Ultimately I searched on three scales using YCrCb 3-channel HOG features plus spatially binned color and histograms of color in the feature vector, which provided a nice result.  Here are some example images:

![alt text](https://github.com/zoespot/CarND-Advanced-Lane-Lines/blob/master/output_images/window_scale_result.png)

---

### Video Implementation

#### 1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (somewhat wobbly or unstable bounding boxes are ok as long as you are identifying the vehicles most of the time with minimal false positives.)
Here's a [link to my video result](https://github.com/zoespot/CarND-Advanced-Lane-Lines/blob/master/output_videos/output_project_video_smooth.mp4)

An earlier implementation without smoothing is at here [link to my video result without smoothing](https://github.com/zoespot/CarND-Advanced-Lane-Lines/blob/master/output_videos/output_project_video_smooth.mp4). It more unstable, but individual detection is more accurate in some frames.

Both of them detects cars well without much false detections.

#### 2. Describe how (and identify where in your code) you implemented some kind of filter for false positives and some method for combining overlapping bounding boxes.

I recorded the positions of positive detections in each frame of the video with Vehicle() class.  From the positive detections I created a heatmap and then thresholded that map to identify vehicle positions.  I then used `scipy.ndimage.measurements.label()` to identify individual blobs in the heatmap.  I then assumed each blob corresponded to a vehicle.  I constructed bounding boxes to cover the area of each blob detected.  

### Here are six frames and their corresponding heatmaps:

![alt text](https://github.com/zoespot/CarND-Advanced-Lane-Lines/blob/master/output_images/pipeline_test6.png)

### Here the resulting bounding boxes are drawn onto sequential frames in a series:
![alt text](https://github.com/zoespot/CarND-Advanced-Lane-Lines/blob/master/output_images/window_scale_series.png.png)
![alt text](https://github.com/zoespot/CarND-Advanced-Lane-Lines/blob/master/output_images/window_scale_series3.png)

---

### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

The problems I encountered in the project and their solutions:
* **Window Scale Choice** One set of scales works for one set of images, but failed in another set. Finally I have to choose 3 different scales.
* **False Positive Detection** It was difficult to exclude some of the false positive detection, even I already have increased the heatmap threshold and used the Vehicle Class objects to record and set threshold for the multiple detections in successive frames. Once the thresholds are set high enough to exclude the false positives, it will miss some real positives. Finally I have to further restrict the ROI (region of interests) manually to the trapezoidal road region and use intermediate thresholds.
* **Smooth Detection with Close Cars Separation** I use Vehicle Class objects to records the detected cars characteristics. The final boxes are average of 10 frames. The difficulty of using Class objects is to identify which heatmap blobs in current frame belongs which car in the previous detected cars objects. Also once two cars get close or overlaps, the bounding box merges. However, the problem is when they are separated again, how to separate the bounding boxes which links to the previous cars objects. The implementation are included in the `draw_labeled_bboxes_heatmap` function in the `vehicle_detect_smooth.ipynb` code.

Limitations:
* **SLOW** The current algorithm needs around 2.1-2.4 seconds to process one frame. It's much slower than the real-time expectation. It's due to the excessive number of scales used in order to cover small and large cars and excluding false positives. Even though I have already used HOG preprocessing over the whole image and sub-sampling, it still quite time consuming, since for each scale, HOG features are re-computed. It may help to reduce the number of frames per seconds sampling for real-time requirement.

* **Generalization** Though my output video performed well on the existing project, there are many manual parameters tunings as ROI, heatmap and frame successive detection threshold, sliding windows scales etc. They may need to be retuned for different road conditions.

Vehicle detection with convolutional neural network, e.g. YOLO algorithm may perform better and easy to generalize to different road conditions.
