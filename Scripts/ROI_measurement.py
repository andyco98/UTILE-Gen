import cv2
import numpy as np
import os

 

# Function to calculate area and equivalent diameter
def calculate_area_and_diameter(img):
    area = cv2.countNonZero(img)
    equivalent_diameter = np.sqrt(4 * area / np.pi)
    return area, equivalent_diameter

 

# Function to calculate aspect ratio
def calculate_aspect_ratio(img):
    contours, _ = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contour = max(contours, key=cv2.contourArea)
    _, _, w, h = cv2.boundingRect(contour)
    aspect_ratio = float(w) / h
    return aspect_ratio

 

def calculate_solidity(img):
    contours, _ = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contour = max(contours, key=cv2.contourArea)
    hull = cv2.convexHull(contour)
    hull_area = cv2.contourArea(hull)
    contour_area = cv2.contourArea(contour)
    if hull_area == 0:
        return 0  # Or return any other special value to indicate undefined solidity
    else:
        solidity = float(contour_area) / hull_area
        return solidity

 

def calculate_orientation(img):
    contours, _ = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contour = max(contours, key=cv2.contourArea)  # Select the contour with the greatest area
    if len(contour) < 5:
        return None  # Or any other value to indicate undefined orientation
    else:
        (_, _), (_, _), angle = cv2.fitEllipse(contour)
        return angle

 

# Function to calculate extent
def calculate_extent(img):
    contours, _ = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contour = max(contours, key=cv2.contourArea)
    contour_area = cv2.contourArea(contour)
    x, y, w, h = cv2.boundingRect(contour)
    rect_area = w * h
    extent = float(contour_area) / rect_area
    return extent

 

# Function to calculate perimeter
def calculate_perimeter(img):
    contours, _ = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contour = max(contours, key=cv2.contourArea)
    perimeter = cv2.arcLength(contour, True)
    return perimeter

 

def calculate_roundness(img):
    area, equivalent_diameter = calculate_area_and_diameter(img)
    perimeter = calculate_perimeter(img)
    if perimeter == 0:
        return None  # Or any other value to indicate undefined roundness
    else:
        roundness = 4 * np.pi * area / (perimeter ** 2)
        return roundness

 

def draw_metrics(img, metrics, position=(50, 50), font=cv2.FONT_HERSHEY_SIMPLEX, 
                 font_scale=1, color=(255, 255, 255), thickness=2):
    y = position[1]
    for key, value in metrics.items():
        if value is None:  # If the value is None, print N/A
            text = f"{key}: N/A"
        else:
            text = f"{key}: {value:.2f}"
        cv2.putText(img, text, (position[0], y), font, font_scale, color, thickness)
        y += 30  # Move down for next line

 

# Create a directory for the output images
output_folder = "./Output_Images"
os.makedirs(output_folder, exist_ok=True)

 

# Extract ROIs from all masks in a folder
mask_folder = "./Test_Masks"
os.makedirs(mask_folder, exist_ok=True)  # Ensure mask folder exists

 

# Iterate over all mask files in the directory
for mask_file in os.listdir(mask_folder):
    mask_file_path = os.path.join(mask_folder, mask_file)
    mask = cv2.imread(mask_file_path, cv2.IMREAD_GRAYSCALE)  # Load mask as grayscale

 

    # Find contours (i.e., bubbles) in the mask
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

 

    # Iterate over each contour/bubble
    for i, contour in enumerate(contours):
        # Create a blank image to draw the current contour/bubble on
        roi = np.zeros_like(mask)
        cv2.drawContours(roi, [contour], -1, (255), thickness=cv2.FILLED)

 

        # Calculate metrics for the current ROI
        area, diameter = calculate_area_and_diameter(roi)
        aspect_ratio = calculate_aspect_ratio(roi)
        solidity = calculate_solidity(roi)
        orientation = calculate_orientation(roi)
        extent = calculate_extent(roi)
        perimeter = calculate_perimeter(roi)
        roundness = calculate_roundness(roi)

 

        # Store metrics in a dictionary
        metrics = {
            'Area': area,
            'Diameter': diameter,
            'Aspect Ratio': aspect_ratio,
            'Solidity': solidity,
            'Orientation': orientation,
            'Extent': extent,
            'Perimeter': perimeter,
            'Roundness': roundness
        }

 

        # Draw metrics on the original color image
        roi_color = cv2.cvtColor(roi, cv2.COLOR_GRAY2BGR)  # Convert to color
        draw_metrics(roi_color, metrics)

 

        # Save the ROI image with overlaid metrics
        filename = os.path.splitext(os.path.basename(mask_file))[0]  # Base filename without extension
        roi_file = os.path.join(output_folder, f"{filename}_roi_{i}.png")
        cv2.imwrite(roi_file, roi_color)