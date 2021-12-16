# import des librairies nécessaires
import cv2
import numpy as np 
import math
import mediapipe as mp 

# on stocke des fonctions mediapipe dans des variables
mp_face_mesh = mp.solutions.face_mesh
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles 

# on définit des fonctions de modèle
def get_face_landmarks(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) # on change les couleurs de BGR à RGB
    results = face_mesh.process(img)
    return results

cam = cv2.VideoCapture(0) # allumer la caméra

# on applique la fonction modèle
with mp_face_mesh.FaceMesh(max_num_faces=1, # détecte 1 visage max
                            min_detection_confidence=0.5,
                            min_tracking_confidence=0.5) as face_mesh:
    while cam.isOpened():
        _, frame = cam.read() # frame is the image that we receive
        frame = cv2.flip(frame, 1) # flip the image received because it's a selfie

        cv2.imshow('Webcam', frame)

        results = get_face_landmarks(frame) # on récupère les results : dict different for each face seen 
        # results.multi_face_landmarks # récup les coordonnées des points de tous les visages différents vus quand la cam était ouverte
        # results.multi_face_landmarks[0] : donne les infos pour le premier visage vu
        # results.multi_face_landmarks[0].landmark : donne les coordonnées pour le premier visage vu
        
        if results.multi_face_landmarks :


        if cv2.waitKey(1) == 27 : # 27 veut dire escap : si on fait Esc, la boucle s'arrête
            break

cam.release() # on release la camera parce qu'on en a plus besoin
cv2.destroyAllWindows() # on détruit toutes les fenêtres de caméra pour pouvoir relancer la caméra une autre fois


