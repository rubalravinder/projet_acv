# import des librairies nécessaires
import cv2
import numpy as np 
import math
import mediapipe as mp 

# on stocke des fonctions mediapipe dans des variables
mp_face_mesh = mp.solutions.face_mesh
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_drawing_utils = mp.solutions.drawing_utils
Holistic = mp.solutions.holistic.Holistic

def blend_img_with_overlay(img, overlay_img, blending_pos_x, blending_pos_y):
    img_h, img_w = img.shape[:2]
    over_h, over_w = overlay_img.shape[:2]

    crop_left = 0
    crop_right = 0
    crop_top = 0
    crop_bottom = 0

    if blending_pos_y < 0 : # si on dépasse sur la gauche de l'img
        crop_left = - blending_pos_y
    if blending_pos_y + over_w > img_w: # si on dépasse sur la droite de l'img
        crop_right = blending_pos_y + over_w - img_w

    if blending_pos_x < 0 : # si on dépasse par le dessous de l'img
        crop_top = - blending_pos_x
    if blending_pos_x + over_h > img_h: # si on dépasse par le haut de l'img
        crop_bottom = blending_pos_x + over_h - img_h
    

    new_img = img.copy()

    pos_x2 = blending_pos_x + over_h
    pos_y2 = blending_pos_y + over_w


    if (crop_left < over_w) and (crop_right < over_w) and (crop_top < over_h) and (crop_bottom < over_h):
        extOverlay = np.zeros(img.shape, np.uint8) # on crée un array de 0. Elle peut que prendre 256 valeurs différentes
        extOverlay[(blending_pos_x + crop_top):(pos_x2 - crop_bottom), (blending_pos_y + crop_left):(pos_y2 - crop_right)] = overlay_img[crop_top:(over_h - crop_bottom),crop_left:(over_w - crop_right),:3]

        new_img[extOverlay > 0] = extOverlay[extOverlay > 0] # on met dans new img seulement les valeurs différentes de 0 (où y'as de l'info)

    return new_img

def run_filter_with_mediapipe_model(mediapipe_model, mediapipe_based_filter):
    cap = cv2.VideoCapture(-1)
    
    with mediapipe_model as model:
        while cap.isOpened():
            success, image = cap.read()

            if not success:
                print("Ignoring empty camera frame.")
                continue     # If loading a video, use 'break' instead of 'continue'.

            # Flip the image horizontally for a later selfie-view display, and convert
            # the BGR image to RGB.
            image = cv2.cvtColor(cv2.flip(image, 1), cv2.COLOR_BGR2RGB)

            results = model.process(image)

            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

            result_image = mediapipe_based_filter(image, results)
            
            cv2.imshow('MediaPipe', result_image)

            if cv2.waitKey(5) & 0xFF == ord('q'):
                break

    cap.release()
    cv2.destroyAllWindows()


def draw_blob_hand(image, results):
    if (results.right_hand_landmarks != None):
        cv2.circle(image,
            (int(results.right_hand_landmarks.landmark[4].x*image.shape[1]), int(results.right_hand_landmarks.landmark[4].y*image.shape[0])),
            20,
            (255, 0, 0),
            2)
        
    if (results.left_hand_landmarks != None):    
        cv2.circle(image,
            (int(results.left_hand_landmarks.landmark[4].x*image.shape[1]), int(results.left_hand_landmarks.landmark[4].y*image.shape[0])),
            20,
            (50, 168, 82),
            2)
    
    return image

def draw_Ok_hand(image, results):
    if (results.right_hand_landmarks != None):
        x4 = int(results.right_hand_landmarks.landmark[4].x*image.shape[1])
        y4 = int(results.right_hand_landmarks.landmark[4].y*image.shape[0])
        x8 = int(results.right_hand_landmarks.landmark[8].x*image.shape[1])
        y8 = int(results.right_hand_landmarks.landmark[8].y*image.shape[0])
        
        if (abs(x8-x4) + abs(y8-y4)) < 20:
            cv2.circle(image,
                (x4, y4),
                20,
                (255, 0, 0),
                2)
    
    return image

def draw_object_Ok_hand(img, results):
    png_fname = './xmas_hat2.png'
    object = cv2.imread(png_fname, cv2.IMREAD_UNCHANGED)
    new_img = img.copy()

    if results.right_hand_landmarks:

        x4 = int(results.right_hand_landmarks.landmark[4].x*img.shape[1])
        y4 = int(results.right_hand_landmarks.landmark[4].y*img.shape[0])
        x8 = int(results.right_hand_landmarks.landmark[8].x*img.shape[1])
        y8 = int(results.right_hand_landmarks.landmark[8].y*img.shape[0])
        
        obj_h, obj_w = object.shape[:2]
        img_h, img_w = img.shape[:2]
        ratio_w = 0.2 # voir si ca marche
        object = cv2.resize(object, (int(ratio_w * obj_w), int(obj_h * ratio_w))) # resize
        
        if (abs(x8-x4) + abs(y8-y4)) < 20:
            obj_h, obj_w = object.shape[:2]
            pos_x = int(y4 - obj_h)
            pos_y = int(x4)
            new_img = blend_img_with_overlay(new_img, object, pos_x, pos_y)
    return new_img


# run_filter_with_mediapipe_model(
#     mediapipe_model=Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5),
#     mediapipe_based_filter=draw_blob_hand
# )

# run_filter_with_mediapipe_model(
#     mediapipe_model=Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5),
#     mediapipe_based_filter=draw_Ok_hand
# )

run_filter_with_mediapipe_model(
    mediapipe_model=Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5),
    mediapipe_based_filter=draw_object_Ok_hand
)


# cam = cv2.VideoCapture(0) # allumer la caméra
    
# with Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as model:
#     while cam.isOpened():
#         _, frame = cam.read()
#         frame = cv2.flip(frame, 1)
#         results = model.process(frame)
#         cv2.imshow('OK xmas_hat', draw_object_Ok_hand(frame, './xmas_hat3.png', results))

#         if cv2.waitKey(1) == 27 :
#             break

# cam.release()
# cv2.destroyAllWindows()