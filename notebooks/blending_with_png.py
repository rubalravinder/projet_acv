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

    # Maintenant, on aimerait que la caméra crash pas à chaque fois qu'on sort du cadre
    # on va faire en sorte de croper le filtre (on pourrait sinon lui dire de pas mettre du filtre du tout)
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

    # on a x et y pour l'image, on récupère les 2 autres extrémités du rectangle correspondant à l'overlay
    pos_x2 = blending_pos_x + over_h
    pos_y2 = blending_pos_y + over_w

    # on veut juste récup les oreilles et pas toute l'img, pour pas avoir le fond noir.
    # pour ca on va créer un masque qui est une matrice de booleans

    if (crop_left < over_w) and (crop_right < over_w) and (crop_top < over_h) and (crop_bottom < over_h):
        # on a une matrice mais que de la taille de l'overlay, on en crée une de la taille de l'img pour pouvoir l'appliquer partout
        extOverlay = np.zeros(img.shape, np.uint8) # on crée un array de 0. Elle peut que prendre 256 valeurs différentes
        extOverlay[(blending_pos_x + crop_top):(pos_x2 - crop_bottom), (blending_pos_y + crop_left):(pos_y2 - crop_right)] = overlay_img[crop_top:(over_h - crop_bottom),crop_left:(over_w - crop_right),:3]
        # on met les valeurs des pixels de l'overlay, le reste reste à 0 (background + reste de l'img originale)

        # on écrase les valeurs dans img avec celles de l'overlay
        new_img[extOverlay > 0] = extOverlay[extOverlay > 0] # on met dans new img seulement les valeurs différentes de 0 (où y'as de l'info)

    return new_img


def lens_filter_ears(img, png_fname):
    """
    png_fname pour récupérer le path de l'image
    """

    results = get_face_landmarks(img)
    doggy_ears = cv2.imread(png_fname, cv2.IMREAD_UNCHANGED) # read the image with opencv in another window than img
    # on vérifie qu'il y a 4 channels (le dernier pour alpha) pour vérifier que c'est bien un png
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
                    (int(ratio_w * dog_w), int(dog_h*ratio_w))) # nvelles dimensions de l'img

        # on veut blend l'img avec les doggy ears. on cherche la position des ears sur l'image
        # /!\ dans opencv, x et y sont inversés mais pas dans mediapipe
        dog_h, dog_w = doggy_ears.shape[:2] # les dim ont changé vu qu'on resize, on récup les nvelles valeurs

        pos_x = int(img_h * face_top.y - dog_h)
        pos_y = int(img_w * face_top.x - dog_w/2)

        # on utile une fonction qu'on a définit plus haut pour blend et crop
        doggy_ears = blend_img_with_overlay(img, doggy_ears, pos_x, pos_y)

    return doggy_ears

def lens_filter_hat(img, png_fname, shift_x, decimal_taille, decalage_vers_droite):
    """
    png_fname pour récupérer le path de l'image
    shift_x : valeur de shift en hauteur pour le filtre qu'on met sur le haut de la tête
    """

    results = get_face_landmarks(img)
    xmas_hat = cv2.imread(png_fname, cv2.IMREAD_UNCHANGED)
    new_img = img.copy()

    if results.multi_face_landmarks:
        face_landmarks = results.multi_face_landmarks[0].landmark

        hat_h, hat_w = xmas_hat.shape[:2]
        face_pin_1 = face_landmarks[21] # un pts à gauche du crane
        face_pin_2 = face_landmarks[251] # un pts à droite du crane

        angle = compute_angle((face_pin_1.x, face_pin_1.y), (face_pin_2.x, face_pin_2.y))

        M = cv2.getRotationMatrix2D((hat_w/2, hat_h/2), angle, 1) # rotation du hat en fonction du visage
        xmas_hat = cv2.warpAffine(xmas_hat, M, (hat_w, hat_h))

        face_right = face_landmarks[454] # pts le plus à droite du visage
        face_left = face_landmarks[234] # pts le plus à gauche
        
        face_top = face_landmarks[10] # pts le plus haut du visage
        face_bottom = face_landmarks[152] # pts le plus bas du visage

        face_w = math.sqrt((face_right.x - face_left.x)**2 + (face_right.y - face_left.y)**2) # largeur visage
        face_h = math.sqrt((face_top.x - face_bottom.x)**2 + (face_top.y - face_bottom.y)**2) # hauteur visage

        img_h, img_w = img.shape[:2]

        ratio_w = (face_w * img_w) / hat_w + decimal_taille
        ratio_h = (img_h * face_h) / hat_h + decimal_taille

        xmas_hat = cv2.resize(xmas_hat, (int(ratio_w * hat_w), int(hat_h * ratio_w))) # resize

        hat_h, hat_w = xmas_hat.shape[:2]

        pos_x = int(img_h * face_top.y - (hat_h/2 * shift_x))
        pos_y = int(img_w * face_top.x - hat_w/decalage_vers_droite)

        xmas_hat = blend_img_with_overlay(img, xmas_hat, pos_x, pos_y)

    return xmas_hat




cam = cv2.VideoCapture(0) # allumer la caméra

# on applique la fonction modèle
with mp_face_mesh.FaceMesh(max_num_faces=1, # détecte 1 visage max
                            min_detection_confidence=0.5,
                            min_tracking_confidence=0.5) as face_mesh:
    while cam.isOpened():
        _, frame = cam.read() # frame is the image that we receive
        frame = cv2.flip(frame, 1) # flip the image received because it's a selfie

        # cv2.imshow('Webcam', frame) # ouvre une page avec la caméra "brute"
        # cv2.imshow('Face landmarks', draw_face_landmarks(frame)) # ouvre une 2e fenêtre avec les landmarks des visages si y'en a
        # cv2.imshow('Sharpened', sharpening(frame)) # 3e fenêtre avec le filtre qui accentue les bords
        # cv2.imshow('Doggy Ears', lens_filter_ears(frame,"./doggy_ears.png"))
        cv2.imshow('Xmas hat2', lens_filter_hat(frame, "./xmas_hat2.png", 1.2, 0.2, 3))
        cv2.imshow('Xmas hat3', lens_filter_hat(frame, "./xmas_hat3.png", 1, 0.15, 3))


        if cv2.waitKey(1) == 27 : # 27 veut dire escap : si on fait Esc, la boucle s'arrête
            break

cam.release() # on release la camera parce qu'on en a plus besoin
cv2.destroyAllWindows() # on détruit toutes les fenêtres de caméra pour pouvoir relancer la caméra une autre fois