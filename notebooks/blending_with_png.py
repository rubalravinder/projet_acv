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

# results = get_face_landmarks(frame) # on récupère les results : dict different for each face seen 
# results.multi_face_landmarks # récup les coordonnées des points de tous les visages différents vus quand la cam était ouverte
# results.multi_face_landmarks[0] : donne les infos pour le premier visage vu
# results.multi_face_landmarks[0].landmark : donne les coordonnées pour le premier visage vu

# on définit un filtre
def draw_face_landmarks(img):
    results = get_face_landmarks(img)
    new_img = img.copy() # on ne remplace pas l'image, il faut créer une copie qu'on renverra
    if results.multi_face_landmarks : 
        for face_landmarks in results.multi_face_landmarks: # pour chaque visage détecté sur la caméra
            mp_drawing.draw_landmarks(image=new_img,
                                        landmark_list=face_landmarks, # on affiche les points du visage
                                        connections=mp_face_mesh.FACEMESH_TESSELATION, # on affiche les connexions entre les points du visage
                                        landmark_drawing_spec = None,
                                        connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_tesselation_style())
    return new_img

def sharpening(img): # accentue les couleurs, ou permet de repérer les contours en fonction du kernel utilisé
    kernel = np.array([[-1,-1,-1], [-1,10,-1],[-1,-1,-1]])
    img_sharpen = cv2.filter2D(img, -1, kernel)
    return img_sharpen

def compute_angle(point1, point2):
    x1, y1, x2, y2 = point1[0], point1[1], point2[0], point2[1]
    angle = -180/math.pi * math.atan(float(y2-y1)/float(x2-x1)) # on fait la tangente qu'on multiplie par -180/pi
    return angle

def blend_img_with_overlay(img, overlay_img, blending_pos_x, blending_pos_y):
    img_h, img_w = img.shape[:2]
    over_h, over_w = overlay_img.shape[:2]

    new_img = img.copy()

    # les 2 imgs sont pas à la même taille.
    pos_x2 = blending_pos_x + over_h
    pos_y2 = blending_pos_y + over_w

    # on met en gray scale pour avoir un channel
    overlayMask = cv2.cvtColor(overlay_img, cv2.COLOR_BGR2GRAY)
    _, overlayMask = cv2.threshold(overlayMask, 10, 1, cv2.THRESH_BINARY_INV) # on crée une matrice de bool pour savoir s'il y a
    # du contenu ou si c'est du background
    # pixel au dessus de 10 sont True et ceux en dessous de 10 seront False

    # on a une matrice de bool mais que de la taille de l'overlay, on en crée une de la taille de l'img pour pouvoir l'appliquer partout
    extOverlayMask = np.ones(img.shape[:2], np.bool) # on crée un array de bool. Pour l'instant tout est True
    extOverlayMask[blending_pos_x:pos_x2, blending_pos_y:pos_y2] = overlayMask[:,:,:3] # on change certaines des valeurs en False


    # on écrase les valeurs dans img avec celles de l'overlay
    new_img[blending_pos_x:pos_x2, blending_pos_y:pos_y2] = overlay_img[:,:,:3] # on change juste les 3 premiers channels

    return new_img

# récuperer l'image utilisée par la prof pour le filtre depuis le google drive -> day3
def lens_filter(img, png_fname): #png_fname pour récupérer le path de l'image
    results = get_face_landmarks(img)
    doggy_ears = cv2.imread(png_fname, cv2.IMREAD_UNCHANGED) # read the image with opencv in another window than img
    print(doggy_ears.shape) # on vérifie qu'il y a 4 channels (le dernier pour alpha) pour vérifier que c'est bien un png
    # si c'est pas le cas et qu'on a que 3 channels, on met le param IMREAD UNCHANGED pour que opencv change pas les channels en lisant
    # l'image
    new_img = img.copy()

    if results.multi_face_landmarks:
        # on veut l'index de 2 points sur le crane à gauche et à droite où seraient les oreilles de l'img png
        # on ouvre la carte des landmark du visage et on regarde : 332, 103
        face_landmarks = results.multi_face_landmarks[0].landmark # pour le 1er visage

        dog_h, dog_w = doggy_ears.shape[:2] # on récup les dimensions de l'img pour plus bas
        face_pin_1 = face_landmarks[332]
        face_pin_2 = face_landmarks[103]

        # on calcule l'angle entre ces deux pts grâce à une fonction définit plus haut
        angle = compute_angle((face_pin_1.x, face_pin_1.y), (face_pin_2.x, face_pin_2.y))

        # on veut rotationner l'img en fonction de l'angle calculé qui correspond à l'angle du visage
        # voir le notebook day2 ACV
        M = cv2.getRotationMatrix2D((dog_w/2, dog_h/2), angle, 1) # on compute la matrix pour faire la rotation
        # centre de rotation, angle, échelle
        doggy_ears = cv2.warpAffine(doggy_ears, # img
                        M, # matrice de transformation
                        (dog_w, dog_h)) # size of img
    
        # resize image of doggy_ears for them to match the scale of face
        # on va regarder les points landmarks du visage à utiliser pour avoir l'échelle du visage
        face_right = face_landmarks[454] # pts le plus à droite du visage
        face_left = face_landmarks[234] # pts le plus à gauche
        
        face_top = face_landmarks[10] # pts le plus haut du visage
        face_bottom = face_landmarks[152] # pts le plus bas du visage

        # on calcule la largeur du visage
        face_w = math.sqrt((face_right.x - face_left.x)**2 + (face_right.y - face_left.y)**2)
        # on calcule la longueur du visage
        face_h = math.sqrt((face_top.x - face_bottom.x)**2 + (face_top.y - face_bottom.y)**2)

        # on veut changer les dimensions des doggy ears avec un ratio pour la largeur et un pour la hauteur
        img_h, img_w = img.shape[:2] # dimensions de l'img de base affichée sur la caméra

        ratio_w = (face_w * img_w) / dog_w
        ratio_h = (img_h * face_h) / dog_h

        # on resize les doggy ears pour qu'elles soient aux même dimensions que le visage
        doggy_ears = cv2.resize(doggy_ears, # img à resize
                    (int(ratio_w * dog_w), int(dog_h*ratio_h))) # nvelles dimensions de l'img

        # on veut blend l'img avec les doggy ears. on cherche la position des ears sur l'image
        # /!\ dans opencv, x et y sont inversés mais pas dans mediapipe
        dog_h, dog_w = doggy_ears.shape[:2] # les dim ont changé vu qu'on resize, on récup les nvelles valeurs

        pos_x = int(img_h * face_top.y - dog_h/2)
        pos_y = int(img_w * face_top.x - dog_w/2)

        # on utile une fonction pour blend qu'on a définit plus haut
        doggy_ears = blend_img_with_overlay(img, doggy_ears, pos_x, pos_y)

        # on veut juste récup les oreilles et pas toute l'img, pour pas avoir le fond noir.
        # pour ca on va créer un masque qui est une matrice de booleans. Voir fonction pour blend

    return doggy_ears






cam = cv2.VideoCapture(0) # allumer la caméra

# on applique la fonction modèle
with mp_face_mesh.FaceMesh(max_num_faces=1, # détecte 1 visage max
                            min_detection_confidence=0.5,
                            min_tracking_confidence=0.5) as face_mesh:
    while cam.isOpened():
        _, frame = cam.read() # frame is the image that we receive
        frame = cv2.flip(frame, 1) # flip the image received because it's a selfie

        cv2.imshow('Webcam', frame) # ouvre une page avec la caméra "brute"
        # cv2.imshow('Face landmarks', draw_face_landmarks(frame)) # ouvre une 2e fenêtre avec les landmarks des visages si y'en a
        # cv2.imshow('Sharpened', sharpening(frame)) # 3e fenêtre avec le filtre qui accentue les bords
        cv2.imshow('Doggy Ears', lens_filter(frame,"./doggy_ears.png")) # 4e fenêtre avec juste les oreilles





        if cv2.waitKey(1) == 27 : # 27 veut dire escap : si on fait Esc, la boucle s'arrête
            break

cam.release() # on release la camera parce qu'on en a plus besoin
cv2.destroyAllWindows() # on détruit toutes les fenêtres de caméra pour pouvoir relancer la caméra une autre fois


