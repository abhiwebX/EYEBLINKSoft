
import cv2 as cv
import mediapipe as mp
import time
import utils, math
import numpy as np
import pygame 
from pygame import mixer 

# variables 
frame_counter =0
CEF_COUNTER =0
TOTAL_BLINKS =0
start_voice= False
counter_right=0
counter_left =0
counter_center =0 
# constants
CLOSED_EYES_FRAME =3
FONTS =cv.FONT_HERSHEY_COMPLEX

# initialize mixer 
mixer.init()
# loading in the voices/sounds 
voice_left = mixer.Sound('Voice/left.wav')
voice_right = mixer.Sound('Voice/Right.wav')
voice_center = mixer.Sound('Voice/center.wav')

# face bounder indices 
FACE_OVAL=[ 10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288, 397, 365, 379, 378, 400, 377, 152, 148, 176, 149, 150, 136, 172, 58, 132, 93, 234, 127, 162, 21, 54, 103,67, 109]

# lips indices for Landmarks
LIPS=[ 61, 146, 91, 181, 84, 17, 314, 405, 321, 375,291, 308, 324, 318, 402, 317, 14, 87, 178, 88, 95,185, 40, 39, 37,0 ,267 ,269 ,270 ,409, 415, 310, 311, 312, 13, 82, 81, 42, 183, 78 ]
LOWER_LIPS =[61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291, 308, 324, 318, 402, 317, 14, 87, 178, 88, 95]
UPPER_LIPS=[ 185, 40, 39, 37,0 ,267 ,269 ,270 ,409, 415, 310, 311, 312, 13, 82, 81, 42, 183, 78] 
# Left eyes indices 
LEFT_EYE =[ 362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385,384, 398 ]
LEFT_EYEBROW =[ 336, 296, 334, 293, 300, 276, 283, 282, 295, 285 ]

# right eyes indices
RIGHT_EYE=[ 33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161 , 246 ]  
RIGHT_EYEBROW=[ 70, 63, 105, 66, 107, 55, 65, 52, 53, 46 ]

map_face_mesh = mp.solutions.face_mesh

# camera object 
camera = cv.VideoCapture(0)
_, frame = camera.read()
img = cv.resize(frame, None, fx=1.0, fy=1.0, interpolation=cv.INTER_CUBIC)
img_hieght, img_width = img.shape[:2]
print(img_hieght, img_width)




### gui...............................................................................................................................................


# Create a blank image to represent the keyboard with a white background
keyboard = np.full((1000, 1500, 3), (240, 240, 240), dtype=np.uint8)
border_color = (192, 192, 192)  # Pinkish color
border_thickness = 8  # Thickness of the border

# Draw the border around the keyboard
cv.rectangle(keyboard, (0, 0), (keyboard.shape[1]-1, keyboard.shape[0]-1), border_color, border_thickness)



# Function to add heading to the center (without background color)
def add_heading(text):
    font = cv.FONT_HERSHEY_SIMPLEX
    font_scale = 1.5  # Reduced font size for smaller text
    font_thickness = 3  # Reduced thickness for smaller text
    text_color = (0, 255, 0)  # Green text

    # Get text size
    text_size = cv.getTextSize(text, font, font_scale, font_thickness)[0]
    text_width = text_size[0]
    text_height = text_size[1]

    # Calculate the x and y coordinates to center the text
    text_x = (keyboard.shape[1] - text_width) // 2
    text_y = text_height + 20  # Some padding from the top (e.g., 20 pixels)

    # Put the text on the image (no background color this time)
    cv.putText(keyboard, text, (text_x, text_y), font, font_scale, text_color, font_thickness)

# Define the function to create keys and add an image with a green border
def letter(x, y, icon_path, text_label,keyid):
    width = 160  # Key width
    height = 160  # Key height
    border = 1
    text_color = (255,0,0)  # Blue text for visibility

  
    cv.rectangle(keyboard, (x + border, y + border), (x + width - border, y + height - border), (255, 0, 0), border)
        
    # Draw the key (rectangle) with a blue border

    # Load the image (icon) and resize it to fit the key
    icon = cv.imread(icon_path)
    if icon is None:
        print(f"Error: Image {icon_path} not found.")
        return

    icon_resized = cv.resize(icon, (width, height)) 


    # Convert icon to grayscale and create a mask for blending
    icon_gray = cv.cvtColor(icon_resized, cv.COLOR_BGR2GRAY)
    _, mask = cv.threshold(icon_gray, 1, 255, cv.THRESH_BINARY)
    mask_inv = cv.bitwise_not(mask)

    # Define the region of interest (ROI) on the keyboard where the icon will be placed
    roi = keyboard[y:y + height, x:x + width]

    # Create the background and foreground for blending
    keyboard_bg = cv.bitwise_and(roi, roi, mask=mask_inv)
    icon_fg = cv.bitwise_and(icon_resized, icon_resized, mask=mask)

    # Add the icon to the region of interest
    dst = cv.add(keyboard_bg, icon_fg)
    keyboard[y:y + height, x:x + width] = dst

   
    cv.rectangle(keyboard, (x, y), (x + width, y + height), (0, 255, 0), 1)  # Green border

    # Add the text label above the key
    font = cv.FONT_HERSHEY_SIMPLEX
    font_scale = 0.8
    font_thickness = 2
    text_size = cv.getTextSize(text_label, font, font_scale, font_thickness)[0]
    text_width = text_size[0]
    text_height = text_size[1]

    text_x = x + (width - text_width) // 2
    text_y = y - 10  # Position text above the key with some padding

    cv.putText(keyboard, text_label, (text_x, text_y), font, font_scale, text_color, font_thickness)

# Add smaller heading to the center of the GUI (without background)
add_heading("Eye Blink System")

# Example of calling the letter function with text labels
keys_data = {
    1: {'x': 100, 'y': 220, 'icon': 'images/eat.jpg', 'label': 'Eat'},
    2: {'x': 300, 'y': 220, 'icon': 'images/water.jpg', 'label': 'Water'},
    3: {'x': 500, 'y': 220, 'icon': 'images/Toilet.jpg', 'label': 'Toilet'},
    4: {'x': 700, 'y': 220, 'icon': 'images/write.jpg', 'label': 'Message'},
    5: {'x': 900, 'y': 220, 'icon': 'images/Mytv.jpg', 'label': 'Watch TV'},
    6: {'x': 1100, 'y': 220, 'icon': 'images/music.jpg', 'label': 'Music'},
    7: {'x': 100, 'y': 450, 'icon': 'images/call.jpg', 'label': 'Call'},
    8: {'x': 300, 'y': 450, 'icon': 'images/dress.jpg', 'label': 'Dressing'},
    9: {'x': 500, 'y': 450, 'icon': 'images/Gson.jpg', 'label': 'Grandchild'},
    10: {'x': 700, 'y': 450, 'icon': 'images/book.jpg', 'label': 'Reading Book'},
    11: {'x': 900, 'y': 450, 'icon': 'images/emer.jpg', 'label': 'Emergency'},
    12: {'x': 1100, 'y': 450, 'icon': 'images/exit.png', 'label': 'Close App'}
}
for key_id, key_data in keys_data.items():
    letter(key_data['x'], key_data['y'], key_data['icon'], key_data['label'],key_id)

# Integrating video capture into top-right corner with green border
def integrate_videoframe_on_keyboard(frame, x=1150, y=13, width=200, height=150, border_thickness=3, 
                                     border_color=(255, 192, 203)):
    # Resize the frame to fit in the desired area
    resized_frame = cv.resize(frame, (width, height))
    # Place the resized video frame onto the keyboard at the specified location
    keyboard[y:y + height, x:x + width] = resized_frame
    # Draw a green border around the webcam feed
    cv.rectangle(keyboard, (x, y), (x + width, y + height), border_color, border_thickness)


#.......................................... gui.................................................................................................................



# landmark detection function 


def landmarksDetection(img, results, draw=False):
    img_height, img_width= img.shape[:2]
    # list[(x,y), (x,y)....]
    mesh_coord = [(int(point.x * img_width), int(point.y * img_height)) for point in results.multi_face_landmarks[0].landmark]
    if draw :
        [cv.circle(img, p, 2, (0,255,0), -1) for p in mesh_coord]

    # returning the list of tuples for each landmarks 
    return mesh_coord

# Euclaidean distance 
def euclaideanDistance(point, point1):
    x, y = point
    x1, y1 = point1
    distance = math.sqrt((x1 - x)**2 + (y1 - y)**2)
    return distance

# Blinking Ratio
def blinkRatio(img, landmarks, right_indices, left_indices):
    # Right eyes 
    # horizontal line 
    rh_right = landmarks[right_indices[0]]
    rh_left = landmarks[right_indices[8]]
    # vertical line 
    rv_top = landmarks[right_indices[12]]
    rv_bottom = landmarks[right_indices[4]]
    # draw lines on right eyes 
    # cv.line(img, rh_right, rh_left, utils.GREEN, 2)
    # cv.line(img, rv_top, rv_bottom, utils.WHITE, 2)

    # LEFT_EYE 
    # horizontal line 
    lh_right = landmarks[left_indices[0]]
    lh_left = landmarks[left_indices[8]]

    # vertical line 
    lv_top = landmarks[left_indices[12]]
    lv_bottom = landmarks[left_indices[4]]

    rhDistance = euclaideanDistance(rh_right, rh_left)
    rvDistance = euclaideanDistance(rv_top, rv_bottom)

    lvDistance = euclaideanDistance(lv_top, lv_bottom)
    lhDistance = euclaideanDistance(lh_right, lh_left)

    reRatio = rhDistance/rvDistance
    leRatio = lhDistance/lvDistance

    ratio = (reRatio+leRatio)/2
    return ratio 

# Eyes Extrctor function,
def eyesExtractor(img, right_eye_coords, left_eye_coords):
    # converting color image to  scale image 
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    
    # getting the dimension of image 
    dim = gray.shape

    # creating mask from gray scale dim
    mask = np.zeros(dim, dtype=np.uint8)

    # drawing Eyes Shape on mask with white color 
    cv.fillPoly(mask, [np.array(right_eye_coords, dtype=np.int32)], 255)
    cv.fillPoly(mask, [np.array(left_eye_coords, dtype=np.int32)], 255)

    # showing the mask 
    # cv.imshow('mask', mask)
    
    # draw eyes image on mask, where white shape is 
    eyes = cv.bitwise_and(gray, gray, mask=mask)
    # change black color to gray other than eys 
     # cv.imshow('eyes draw', eyes)
    eyes[mask==0]=155
    
    # getting minium and maximum x and y  for right and left eyes 
    # For Right Eye 
    r_max_x = (max(right_eye_coords, key=lambda item: item[0]))[0]
    r_min_x = (min(right_eye_coords, key=lambda item: item[0]))[0]
    r_max_y = (max(right_eye_coords, key=lambda item : item[1]))[1]
    r_min_y = (min(right_eye_coords, key=lambda item: item[1]))[1]

    # For LEFT Eye
    l_max_x = (max(left_eye_coords, key=lambda item: item[0]))[0]
    l_min_x = (min(left_eye_coords, key=lambda item: item[0]))[0]
    l_max_y = (max(left_eye_coords, key=lambda item : item[1]))[1]
    l_min_y = (min(left_eye_coords, key=lambda item: item[1]))[1]

    # croping the eyes from mask 
    cropped_right = eyes[r_min_y: r_max_y, r_min_x: r_max_x]
    cropped_left = eyes[l_min_y: l_max_y, l_min_x: l_max_x]

    # returning the cropped eyes 
    return cropped_right, cropped_left

# Eyes Postion Estimator 
def positionEstimator(cropped_eye):
    # getting height and width of eye 
    h, w =cropped_eye.shape
    
    # remove the noise from images
    gaussain_blur = cv.GaussianBlur(cropped_eye, (9,9),0)
    median_blur = cv.medianBlur(gaussain_blur, 3)

    # applying thrsholding to convert binary_image
    ret, threshed_eye = cv.threshold(median_blur, 130, 255, cv.THRESH_BINARY)

    # create fixd part for eye with 
    piece = int(w/3) 

    # slicing the eyes into three parts 
    right_piece = threshed_eye[0:h, 0:piece]
    center_piece = threshed_eye[0:h, piece: piece+piece]
    left_piece = threshed_eye[0:h, piece +piece:w]
    
    # calling pixel counter function
    eye_position, color = pixelCounter(right_piece, center_piece, left_piece)

    return eye_position, color 

# creating pixel counter function 
current_box = 1
def pixelCounter(first_piece, second_piece, third_piece):
    # counting black pixel in each part 
    right_part = np.sum(first_piece==0)
    center_part = np.sum(second_piece==0)
    left_part = np.sum(third_piece==0)
    # creating list of these values
    eye_parts = [right_part, center_part, left_part]

    # getting the index of max values in the list 
    max_index = eye_parts.index(max(eye_parts))
    pos_eye ='' 
    if max_index==0:
        pos_eye="RIGHT"
        print(f"Current BOx : {current_box}")
        
        color=[utils.BLACK, utils.GREEN]
    elif max_index==1:
        pos_eye = 'CENTER'
        color = [utils.YELLOW, utils.PINK]
    elif max_index ==2:
        pos_eye = 'LEFT'
       
        color = [utils.GRAY, utils.YELLOW]
    else:
        pos_eye="Closed"
        color = [utils.GRAY, utils.YELLOW]
    return pos_eye, color

def highlight():
    global current_box
    for key_id, key_data in keys_data.items():
        if key_id == current_box:
            # Highlight the current box with red border
            letter(key_data['x'], key_data['y'], key_data['icon'], key_data['label'], key_id)
            cv.rectangle(keyboard, (key_data['x'], key_data['y']), 
                         (key_data['x'] + 160, key_data['y'] + 160), (0, 0, 255), 2)  # Red border
        else:
            # Reset other boxes with green border
            letter(key_data['x'], key_data['y'], key_data['icon'], key_data['label'], key_id)
            cv.rectangle(keyboard, (key_data['x'], key_data['y']), 
                         (key_data['x'] + 160, key_data['y'] + 160), (0, 255, 0), 2)  # Green border




with map_face_mesh.FaceMesh(min_detection_confidence =0.5, min_tracking_confidence=0.5) as face_mesh:

    # starting time here 
    start_time = time.time()
    # starting Video loop here.
    while True:
        frame_counter +=1 # frame counter
        ret, frame = camera.read() # getting frame from camera 
        if not ret: 
            break # no more frames break
        #  resizing frame
        
        frame = cv.resize(frame, None, fx=1.5, fy=1.5, interpolation=cv.INTER_CUBIC)
        frame_height, frame_width= frame.shape[:2]
        rgb_frame = cv.cvtColor(frame, cv.COLOR_RGB2BGR)
        results  = face_mesh.process(rgb_frame)
        if results.multi_face_landmarks:
            mesh_coords = landmarksDetection(frame, results, False)
            ratio = blinkRatio(frame, mesh_coords, RIGHT_EYE, LEFT_EYE)
            cv.putText(frame, f'ratio {ratio}', (100, 100), FONTS, 1.0, utils.GREEN, 2)
            utils.colorBackgroundText(frame,  f'Ratio : {round(ratio,2)}', FONTS, 0.7, (30,100),2, utils.PINK, utils.YELLOW)

            if ratio >5.5:
                CEF_COUNTER +=1
             #   cv.putText(keyboard, 'Blink', (200, 50), FONTS, 1.3, utils.PINK, 2)                 
              #  utils.colorBackgroundText(frame,  f'Blink', FONTS, 1.7, (int(frame_height/2), 100), 2, utils.YELLOW, pad_x=6, pad_y=6, )
                if TOTAL_BLINKS == 9:
                    TOTAL_BLINKS = 0
                time.sleep(1)
            else:
                if CEF_COUNTER>CLOSED_EYES_FRAME:
                    TOTAL_BLINKS +=1
                    CEF_COUNTER =0
                    highlight()
                    current_box+=1
                    if current_box > 12:
                        current_box = 1
           # cv.putText(keyboard, f'Total Blinks: {TOTAL_BLINKS}', (100, 20), FONTS, 0.6, utils.GREEN, 1)
            utils.colorBackgroundText(keyboard,  f'Total Blinks: {TOTAL_BLINKS}', FONTS, 0.7, (30,30),1)
            
            cv.polylines(frame,  [np.array([mesh_coords[p] for p in LEFT_EYE ], dtype=np.int32)], True, utils.GREEN, 1, cv.LINE_AA)
            cv.polylines(frame,  [np.array([mesh_coords[p] for p in RIGHT_EYE ], dtype=np.int32)], True, utils.GREEN, 1, cv.LINE_AA)

            # Blink Detector Counter Completed
            right_coords = [mesh_coords[p] for p in RIGHT_EYE]
            left_coords = [mesh_coords[p] for p in LEFT_EYE]
            crop_right, crop_left = eyesExtractor(frame, right_coords, left_coords)
            # cv.imshow('right', crop_right)
            # cv.imshow('left', crop_left)
            eye_position_right, color = positionEstimator(crop_right)
           # utils.colorBackgroundText(keyboard, f'R: {eye_position_right}', FONTS, 1.0, (60, 35), 2, color[0], color[1], 3, 3)
            utils.colorBackgroundText(keyboard,  f'R: {eye_position_right}', FONTS, 0.5, (30,55),1)

            eye_position_left, color = positionEstimator(crop_left)
        #    utils.colorBackgroundText(keyboard, f'L: {eye_position_left}', FONTS, 1.0, (50, 40), 2, color[0], color[1], 3, 3)
        #    utils.colorBackgroundText(keyboard,  f'L: {eye_position_left}', FONTS, 0.5, (150,55),1)
            # Starting Voice Indicator 
            if eye_position_right=="RIGHT" and pygame.mixer.get_busy()==0 and counter_right<2:
                # starting counter 
                counter_right+=1
                # resetting counters 
                counter_center=0
                counter_left=0
                # playing voice 
              #  voice_right.play()


            
            
            if eye_position_right=="LEFT" and pygame.mixer.get_busy()==0 and counter_left<2: 
                counter_left +=1
                # resetting counters 
                counter_center=0
                counter_right=0
                # playing Voice 
              #  voice_left.play()



        # calculating  frame per seconds FPS
        end_time = time.time()-start_time
        fps = frame_counter/end_time
        

        frame =utils.textWithBackground(frame,f'FPS: {round(fps,1)}',FONTS, 1.0, (30, 50), bgOpacity=0.9, textThickness=2)
        # writing image for thumbnail drawing shape
        # cv.imwrite(f'img/frame_{frame_counter}.png', frame)
        # wirting the video for demo purpose 
        integrate_videoframe_on_keyboard(frame)
        cv.imshow("EyeBlinkSoFtware", keyboard)
#        cv.imshow('frame', frame)
        key = cv.waitKey(2)
        if key==ord('q') or key ==ord('Q'):
            break

 
    
    
cv.destroyAllWindows()
camera.release()
