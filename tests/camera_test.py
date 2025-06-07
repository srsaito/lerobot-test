from lerobot.common.robot_devices.cameras.configs import OpenCVCameraConfig
from lerobot.common.robot_devices.cameras.opencv import OpenCVCamera
import cv2
import os

for camera_index in [0, 2]:
	# config = OpenCVCameraConfig(camera_index=0, fps=30, width=640, height=480)
	# index = 1 is back camera, index = 0 is selfie camera
	config = OpenCVCameraConfig(camera_index=camera_index)
	camera = OpenCVCamera(config)
	camera.connect()
	
	color_image = camera.read()
	print(f"Image shape: {color_image.shape}")
	print(f"Image dtype: {color_image.dtype}")
	
	# Save the image to a file
	image_filename = f"captured_image_camera_{camera_index}.png"  
	cv2.imwrite(image_filename, color_image)
	print(f"Image saved as: {image_filename}")
	
	# Open the image using a system command (macOS 'open' command)
	# This will open the image in the default image viewer
	os.system(f"open {image_filename}")
	camera.disconnect()
