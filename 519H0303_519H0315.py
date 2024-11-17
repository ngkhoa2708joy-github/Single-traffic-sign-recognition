import cv2 as cv
import numpy as np
import glob
from skimage.exposure import is_low_contrast
from matplotlib import pyplot as plt
import numpy as np
import numpy.random as rng

#handle file png in dir
png_files = glob.glob(r"*.png")

for file in png_files:
    # width after resize
    fixed_width = 200
    
    # Fuction to increase the low-contrasted images
    def contrast_enhance(img):
        img_lab = cv.cvtColor(img, cv.COLOR_BGR2Lab)
        L, a, b = cv.split(img_lab)

        # Apply histogram equalization and contrast stretching on L channel
        min_val, max_val = np.min(L), np.max(L)
        L_scaled = (L - min_val) * (255 / (max_val - min_val))
        L_eq = cv.equalizeHist(L_scaled)

        img_lab_merge = cv.merge((L_eq, a, b))
        img_enhanced = cv.cvtColor(img_lab_merge, cv.COLOR_Lab2BGR)
        return img_enhanced
          
    #signs often have a black symbol on a white background with a red crossed circle
    #white color
    lower_white = (0, 0, 0)
    upper_white = (0, 0, 255)
    
    #Red color 
    lower_red1 = (0, 40, 50)
    upper_red1 = (10, 255, 210)
    lower_red2 = (165, 40, 50)
    upper_red2 = (179, 255, 210)

    # Black colors
    lower_black = (0, 0, 0)
    upper_black = (179, 255, 5)

    # Blue color 
    lower_blue = (90, 40, 50)
    upper_blue = (120, 255, 210)

    # Function to make color segmentation
    def color_seg(img, kernel_size=None):
        """
        img: image in bgr
        kernel_size: None (default:(3, 3))
        
        """
        hsv_img = cv.cvtColor(img, cv.COLOR_BGR2HSV)

        # Define color ranges in HSV (adjust ranges as needed)
        # Consider adding more color ranges for better coverage
        mask_red1 = cv.inRange(hsv_img, lower_red1, upper_red1)
        mask_red2 = cv.inRange(hsv_img, lower_red2, upper_red2)
        mask_black = cv.inRange(hsv_img, lower_black, upper_black)
        mask_white = cv.inRange(hsv_img, lower_white, upper_white)  

        # Combine masks using bitwise OR (potential sign regions)
        mask_combined = cv.bitwise_or(mask_red1, mask_red2)
        mask_combined = cv.bitwise_or(mask_combined, mask_black)
        mask_combined = cv.bitwise_or(mask_combined, mask_white)

        # Apply noise reduction before morphological operations (optional)
        # You can uncomment these lines and adjust blur_kernel_size as needed
        # blur_kernel_size = 5
        # mask_combined = cv2.blur(mask_combined, (blur_kernel_size, blur_kernel_size))

        # Morphological operations with optional kernel size
        if kernel_size is not None:
            kernel = np.ones(kernel_size, np.uint8)
        else:
            kernel = np.ones((3, 3), np.uint8)

        # Apply opening and closing operations for noise reduction
        mask_combined = cv.morphologyEx(mask_combined, cv.MORPH_OPEN, kernel)
        mask_combined = cv.morphologyEx(mask_combined, cv.MORPH_CLOSE, kernel)

        return mask_combined
    
    
    def check_aspect_ratio(approx, tol):
        # Extract width and height from bounding rectangle
        x, y, w, h = cv.boundingRect(approx)
        aspect_ratio = abs(w / h)

        # Check if aspect ratio is close to 1 (rectangle) within tolerance
        return abs(aspect_ratio - 1) <= tol     
    
    # circle detection
    hough_dict = {
        "dp": 1,
        "minDist": 20,
        # Adaptive param1 based on average image intensity (heuristic)
        "param1": None,  # Set to None for internal calculation
        "param2": 50,
        "minRadius": 5,  # Adjust based on expected minimum circle size
        "maxRadius": 100  # Adjust based on expected maximum circle size
    }
    
    def cnt_circle(img, hough_dict):
        """
        Detects circles in an image using the Hough Circle Transform with 
        adaptive param1 and noise reduction (optional).

        Args:
            img: Grayscale image (assumed to be pre-processed)
            hough_dict: Dictionary containing parameters for the Hough Circle Transform.

        Returns:
            cnt: Largest detected circle contour (None if none found).
        """
        mask = np.zeros_like(img)

        # Calculate average intensity for adaptive param1
        average_intensity = np.mean(img)
        if hough_dict["param1"] is None:
            hough_dict["param1"] = max(hough_dict["param1"] * (average_intensity / 255), 50)
        else:
            print("Warning: hough_dict['param1'] is not set to None for adaptive calculation.")

        circles = cv.HoughCircles(img, 
                                    cv.HOUGH_GRADIENT, 
                                    hough_dict["dp"], 
                                    hough_dict["minDist"], 
                                    param1=hough_dict["param1"], 
                                    param2=hough_dict["param2"],
                                    minRadius=hough_dict["minRadius"], 
                                    maxRadius=hough_dict["maxRadius"])

        if circles is None:
            return None
        else:
            list_circles = circles[0]
            largest_circles = max(list_circles, key=lambda x: x[2])
            center_x, center_y, r = largest_circles

            cv.circle(mask, (int(center_x), int(center_y)), int(r), 255)
            cnts = cv.findContours(mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
            cnt = cnts[0]

            if len(cnts[0]) > 0:
                return max(cnt, key=cv.contourArea)
            else:
                return cnt[-1]
    
    # rectangle detection    
    def cnt_rect(cnts, coef=0.1, aspect_ratio_tol=0.1):
        contour_list = []
        for cnt in cnts:
            peri = cv.arcLength(cnt, True)
            approx = cv.approxPolyDP(cnt, coef * peri, True)

            # Check for 4 vertices and aspect ratio
            if len(approx) == 4 and check_aspect_ratio(approx, aspect_ratio_tol):
                contour_list.append(cnt)

        if not contour_list:
            return None
        else:
            LC = max(contour_list, key=cv.contourArea)
            return LC
    
    def auto_canny(img, method, sigma=0.5, block_size=31, C=7):
        if len(img.shape) == 3:
             # Convert color image to grayscale
            img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
        if method=="otsu":
            Th, _ = cv.threshold(img, 0, 255, cv.THRESH_OTSU) 
            
        elif method == "adaptive":
            Th = cv.adaptiveThreshold(img, 255, cv.ADAPTIVE_THRESH_MEAN_C, cv.THRESH_BINARY, block_size, C)
        else:
            raise ValueError("Invalid thresholding method specified!")
        
        lowTh = (1-sigma) * Th #The low threshold 
        highTh = (1+sigma) * Th #The high threshold 
        
        edge_image = cv.Canny(img, lowTh, highTh)
        return edge_image, highTh

    

    # combine the results of 2 shape detectors
    def integrate_circle_rect(rect_cnt, circle_cnt, cnt):
        # Filter out small and very large contours (adjust thresholds as needed)
        filtered_cnt = [c for c in cnt if cv.contourArea(c) > 50 and cv.contourArea(c) < 10000]

        # Prioritize detections from specific detectors (adjust based on your needs)
        if circle_cnt is not None and cv.contourArea(circle_cnt) > 100:
            return circle_cnt
        elif rect_cnt is not None and cv.contourArea(rect_cnt) > 100:
            return rect_cnt

        # If no detections from specific detectors or they are small, use the largest contour
        if len(filtered_cnt) > 0:
            return max(filtered_cnt, key=cv.contourArea)
        else:
            return None

    # combine the results of edge detector + color based segmentation
    def integrate_edge_color(output1, output2):
        if not isinstance(output1, np.ndarray): # isinstance(object, type)
            output1 = np.array(output1)
        
        if not isinstance(output2, np.ndarray):
            output2 = np.array(output2)
        
        if len(output1)==0 and len(output2)==0:
            return np.array([])
    
        elif len(output1)==0 and output2.shape[-1]==2:
            return output2
    
        elif len(output2)==0 and output1.shape[-1]==2:
            return output1
    
        else:
            if cv.contourArea(output1[0]) > cv.contourArea(output2[0]):
                return output1
            else:
                return output2
    
        
    def show_img(window_name, img, adjust=False):
        if adjust:
            cv.namedWindow(window_name, cv.WINDOW_NORMAL)
        else:
            cv.namedWindow(window_name)
            
        cv.imshow(window_name, img)
        key = cv.waitKey(0)
        if key == 27:  # Esc key
            cv.destroyAllWindows()
        
        
    # Read image
    img = cv.imread(file)
    img_copy = img.copy()
    img_copy = cv.GaussianBlur(img_copy, (5, 5), 0)

    #Processing Image
    img_denoised = cv.medianBlur(img_copy, 3)
    if is_low_contrast(img_denoised):
        img_denoised = contrast_enhance(img_denoised)
        
    #Resize the image
    ratio = fixed_width / img.shape[1]
    img_resized = cv.resize(img_denoised, None, fx=ratio, fy=ratio, interpolation=cv.INTER_AREA)

    #change to grayscale
    gray = cv.cvtColor(img_resized, cv.COLOR_BGR2GRAY)

    #Edge detection 
    #edge, canny_th2 = auto_canny(gray, "adaptive",block_size=31, C=7)
    edge, canny_th2 = auto_canny(gray, "otsu")


    # Perform shape detectors
    cnts = cv.findContours(edge, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
    cnt = cnts[0]
    rect = cnt_rect(cnt)

    hough_dict["param1"] = canny_th2
    circle = cnt_circle(gray, hough_dict)

    output1 = integrate_circle_rect(rect, circle, cnt)

    # perform color segmentation
    color_segmented = color_seg(img_resized)

    # perform rectangular object detection
    cnts = cv.findContours(color_segmented, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
    cnt = cnts[0]
    rect = cnt_rect(cnt)

    # perform circular object detection
    hough_dict["param1"] = 200
    circle = cnt_circle(color_segmented, hough_dict)

    output2 = integrate_circle_rect(rect, circle, cnt)
    results = integrate_edge_color(output1, output2)

    x, y, w, h = cv.boundingRect(results)
    color = (0, 156, 0)
    cv.rectangle(img_resized, (x, y), (x+w, y+h), color, 3)
    show_img("final_dectection", img_resized)
