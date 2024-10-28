import cv2
import numpy as np

# Initialize the camera
laser_camera = cv2.VideoCapture(0)
laser_camera.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
laser_camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

# Focal length (adjust based on the camera's actual parameters)
FOCAL_LENGTH = 700  # Assumed focal length in pixels
LASER_DIAMETER = 0.006  # Diameter of the laser dot (in meters, assumed to be 6mm)
DISTANCE_TOLERANCE = 0.1  # Distance tolerance in meters

def count_white_pixels(matrix):
    return np.count_nonzero(matrix == 255)  # Use numpy to speed up the count

def calculate_distance(laser_area):
    # Calculate distance
    if laser_area > 0:
        distance = (FOCAL_LENGTH * LASER_DIAMETER) / (laser_area ** 0.5)  # Calculate distance based on laser dot area
        return distance
    return None

while True:
    ret, frame = laser_camera.read()  # Capture a frame from the camera
    if not ret:
        print("Failed to grab frame")
        break

    hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)  # Convert the frame to HSV color space
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  # Convert the frame to grayscale
    _, binary_frame = cv2.threshold(gray_frame, 220, 255, cv2.THRESH_BINARY)  # Apply binary thresholding

    # Define the HSV range for the red laser dot
    lower_red = np.array([160, 100, 100])
    upper_red = np.array([179, 255, 255])
    mask = cv2.inRange(hsv_frame, lower_red, upper_red)  # Create a mask for the red color

    # Use morphological operations to remove noise
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)  # Morphological opening
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)  # Morphological closing

    # Find contours in the mask
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Track the bounding box with the maximum white pixel count
    max_frame_corners = []
    max_white_pixel_count = 0
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)  # Get the bounding rectangle of the contour
        sub_matrix = binary_frame[y:y+h, x:x+w]  # Extract the sub-matrix
        white_pixel_count = count_white_pixels(sub_matrix)  # Count the white pixels
        if white_pixel_count >= max_white_pixel_count:
            max_frame_corners = [x, y, x+w, y+h]  # Update the bounding box with the maximum count
            max_white_pixel_count = white_pixel_count

    # Calculate the laser dot area and estimate the distance
    laser_area = max_white_pixel_count
    distance = calculate_distance(laser_area)

    if distance is not None:
        print(distance)  # Print the estimated distance
        if 1.9 <= distance <= 2.1:  # Check if the distance is within the specified range
            print(f"Estimated distance: {distance:.2f} meters")  # Output the estimated distance
            print("Distance within tolerance (2 meters ± 10 cm)")

    if cv2.waitKey(1) & 0xFF == ord('q'):  # Break the loop if 'q' is pressed
        break

    if max_frame_corners:
        cv2.rectangle(frame, (max_frame_corners[0], max_frame_corners[1]),
                       (max_frame_corners[2], max_frame_corners[3]),
                       (60, 255, 100), 2)  # Draw a rectangle around the detected laser dot

    cv2.imshow('Laser Tracking', frame)  # Show the laser tracking result
    cv2.imshow('Laser Mask', mask)  # Show the mask image

laser_camera.release()  # Release the camera resource
cv2.destroyAllWindows()  # Close all OpenCV windows
