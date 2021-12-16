from matplotlib import pyplot as plt
plt.rcParams['figure.figsize'] = [15, 10]

import numpy as np
import cv2

def to_RGB(image):
    """convertit l'image de BGR en RGB"""
    img = cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
    return img

def to_HSV(image):
    """convertit l'image de BGR en HSV"""
    img = cv2.cvtColor(image,cv2.COLOR_BGR2HSV)
    return img

def filtre_miroir_vertic(image):
    '''Applique un filtre miroir à la verticale de l'image entrée en input'''
    
    height, width = image.shape[:2]
    image_miroir = np.zeros((height, width, 3), 'uint8')
    if width %2 != 0:
        demi_image = image[:,:(width//2)+1]
        demi_image_miroir = np.flip(image[:,:(width//2)], axis = 1)
        image_miroir[:,:(width//2)+1] = demi_image
        image_miroir[:,(width//2)+1:] = demi_image_miroir
    else:
        demi_image = image[:,:(width//2)]
        demi_image_miroir = np.flip(demi_image, axis = 1)
        image_miroir[:,:(width//2)] = demi_image
        image_miroir[:,(width//2):] = demi_image_miroir

    return image_miroir

def filtre_miroir_horiz(image):
    '''Applique un filtre miroir à l'horizontale de l'image entrée en input'''
    height, width = image.shape[:2]
    image_miroir = np.zeros((height, width, 3), 'uint8')

    if height %2 != 0:
        demi_image = image[0:(height//2)+1,:]
        demi_image_miroir = np.flip(image[:height//2,:], axis = 0)
        image_miroir[:(height//2)+1,:] = demi_image
        image_miroir[(height//2)+1:,:] = demi_image_miroir        
    else:
        demi_image = image[0:height//2,:]
        demi_image_miroir = np.flip(demi_image, axis = 0)
        image_miroir[0:height//2,:] = demi_image
        image_miroir[height//2:,:] = demi_image_miroir

    return image_miroir

def filtre_flou(image):
    '''Applique un filtre flou à l'image entrée en input par flou gaussien'''
    blurred = cv2.GaussianBlur(image,(19,19),0)
    return blurred

def filtre_gray(image):
    '''Transforme l'image entrée en input en échelle de gris'''
    grayscale = np.zeros((image.shape[0], image.shape[1], 3), 'uint8') #création d'une image en 3D vide
    grayscale[:,:,0] = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    grayscale[:,:,1] = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    grayscale[:,:,2] = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    return grayscale 

def filtre_pixel(image):
    """Pixelise l'image"""
    height,width = image.shape[:2]
    img = image[0:height:5, 0:width:5] # le 5 correspond au pas (nb de pixels à sauter)
    img = cv2.resize(img, dsize=(width, height))
    return img

def filtre_cartoon(image):
    """Renvoie l'image avec un effet cartoon"""
    height,width = image.shape[:2]
    tublur = cv2.medianBlur(image, 29)

    edge = cv2.Canny(tublur, 10, 300)
    kernel = np.ones((2,2), np.uint8)
    edge = cv2.dilate(edge, kernel, iterations = 1)
    tublur[edge==255] = 0
    return tublur

def quadrillage_filtre(image):
    '''Pour l'image donnée en entrée, la fonction affiche un quadrillage en 3 par 3 de l'image transformée par les filtres.
    Les filtres sont fournis en tant que liste de variables.'''
    
    list_filtres = [filtre_miroir_horiz,filtre_miroir_vertic,to_HSV,to_RGB,filtre_gray,filtre_cartoon,filtre_pixel,filtre_flou]

    newheight = image.shape[0]//3
    newwidth = image.shape[1]//3
    img_resized = cv2.resize(image, dsize=(newwidth,newheight)) # on resize l'image de départ en la divisant par 3. attention : width avant height
    quadrillage = np.zeros((img_resized.shape[0]*3, img_resized.shape[1]*3, 3), 'uint8') #création d'un quadrillage vide


    def no_filter(image): # on crée un filtre pour afficher l'image originale
        return image  

    filtres = list_filtres[:4] + [no_filter] + list_filtres[4:] # on veut que l'image originale soit au milieu du quadrillage
    filter_names = ['Miroir horizontal', 'Miroir vertical', 'HSV', 'RGB', 'Original', 'Nuances de gris', 'Cartoon', 'Pixelisé', 'Flou']
    for y in range(3): # pour chaque colonne
        for x in range(3): # pour chaque ligne
            filtre = filtres[3*y+x] # sélection du filtre à appliquer parmi la liste
            img_filtree = filtre(img_resized)
            quadrillage[y*newheight:(y+1)*newheight,x*newwidth:(x+1)*newwidth] = img_filtree

    return quadrillage


# TEST FONCTION
tree = plt.imread('xmas_tree.jpg') # import de l'image à tester
quadrillage = quadrillage_filtre(tree) # application de la fonction pour créer le quadrillage

# POUR AFFICHER LE RESULTAT : pip install Pillow --> probablement déjà installé dans acv env
from PIL import Image                                                                            
img = Image.fromarray(quadrillage, 'RGB') # conversion du numpy array en image RGB
img.show() # affichage de l'img sur l'appareil
