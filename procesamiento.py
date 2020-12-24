import cv2
import time
import numpy as np
import math
import matplotlib.pyplot as plt
# import the necessary packages
from skimage.exposure import rescale_intensity
from skimage.segmentation import slic, felzenszwalb, watershed
from skimage.color import rgb2gray
from skimage.filters import sobel
from skimage.util import img_as_float
from skimage import io
from skimage import data, segmentation, measure, color, img_as_float
from skimage.measure import regionprops
from sklearn import metrics #ROC graphics, metrics
import itertools


dic_colors = { "lower_color_blue"   : np.array([95, 100, 40], dtype=np.uint8), #Azul
               "upper_color_blue"   : np.array([135, 255, 255], dtype=np.uint8), #Azul
               "lower_color_green"  : np.array([30, 0, 0], dtype=np.uint8), #35,100,20,255/25,52,72 ideal->25, 52, 20, Colorverde 30,0,0     
               "upper_color_green"  : np.array([90, 255, 255], dtype=np.uint8), #102, 255, 255 Color verde 90,255,255
               "lower_color_white"  : np.array([0, 0, 168], dtype=np.uint8), #blanco 0, 0, 212
               "upper_color_white"  : np.array([172,111,255], dtype=np.uint8), #blanco 131, 255, 255
               "lower_color_yellow" : np.array([20, 70,   70], dtype=np.uint8), #amarillo
               "upper_color_yellow" : np.array([35, 255, 255], dtype=np.uint8), #amarillo
               "lower_color_red"    : np.array([0,20,20], dtype=np.uint8), #rojo (175,50,20)
               "upper_color_red"    : np.array([7,255,255], dtype=np.uint8), #rojo (180,255,255)
}

class Aprocesamiento:
    def __init__(self, obj_mapeo, obj_robot, obj_vision):
        self.obj_mapeo  = obj_mapeo
        self.obj_robot  = obj_robot
        self.obj_vision = obj_vision

        self.color_intrin = 0 #Obtener ancho de objeto
        self.width_obj = 0 #Ancho de un objeto
        self.color_image = 0 #Imagen a color
        self.bg_removed = 0 #Imagen a color limitado a distancia
        self.var_limits_inside_object = 5 #Pixeles dentro de objeto detectado

        self.real_array_roc = [] #Lista de real 
        self.pred_array_roc = [] #Lista pred

        self.numbers_array_obj = list(range(0,192))
        self.distance_array_obj = [0]
        self.width_array_obj = [0]

    #Funciones------------------------------------------------------------------------------------------------------------------------------------------
    def cen_moments(self, countours_found):
        M = cv2.moments(countours_found) # Encontrar el centroide del mejor contorno y marcarlo
        if M["m00"] != 0:
            cx, cy = int(M['m10']/M['m00']), int(M['m01']/M['m00'])
        else:
            cx, cy = 0, 0  # set values as what you need in the situation
        return cx, cy

    def paint_region_with_avg_intensity(self, img, rp, mi, channel): #Marcar regiones de segmentacion
        for i in range(rp.shape[0]):
            img[rp[i][0]][rp[i][1]][channel] = mi
        return img

    def seg_superpix(self, img): #Felzenszwalb es mas estable, realizar pruebas con los 3 (slic, watershed & fel..)
        #segments = slic(img, n_segments=200, compactness=10, multichannel=True, enforce_connectivity=True, convert2lab=True)
        segments = felzenszwalb(img, scale=100, sigma=0.5, min_size=60)
        #gradient = sobel(rgb2gray(img))
        #segments = watershed(gradient, markers=250, compactness=0.001)
        for i in range(3):
            regions = regionprops(segments, intensity_image=img[:,:,i])
            for r in regions:
                img = self.paint_region_with_avg_intensity(img, r.coords, int(r.mean_intensity), i)
        return img 

    def filter_color(self, image, color): #Funcion para filtrar color
        hsv_frame = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        color_mask = cv2.inRange(hsv_frame, dic_colors.get("lower_color_"+color), dic_colors.get("upper_color_"+color))
        color_total = cv2.bitwise_and(image, image, mask=color_mask)
        diff_total = cv2.absdiff(image, color_total)
        cv2.imshow('Diferencias detectadas', diff_total)
        #cv2.imwrite("framegreen.png", diff_total) #diff_total imagen sin verde
        return diff_total

    def filter_background(self): #Funcion para filtrar la imagen en distancia
        return None
         
    def get_distance_objs(self, cz_blue): #Obtener distancia z dado un valor dado de z, considerando al robot, analisis en 4.5.1
        print("Centroid in {} cm.".format(cz_blue)) #Distancia de objeto
        A_dis_total = cz_blue * math.sin(math.radians(self.obj_robot.angle_robot_camera))  
        print("Centroid real in {} cm.".format(A_dis_total)) #Distancia de objeto
        return A_dis_total

    def get_coordenates_map(self, distance_robot_obj): #Obtener las coordenadas para mapear objeto con respecto a robot
        xb = math.sin(math.radians(self.obj_robot.get_angle_robot_z())) * distance_robot_obj
        yb = math.cos(math.radians(self.obj_robot.get_angle_robot_z())) * distance_robot_obj
        xb_object = self.obj_mapeo.sx + xb #Sx posicion robot en mapa
        yb_object = self.obj_mapeo.sy + yb #Sy posicion robot en mapa
        return xb_object, yb_object

    #Poner objetos en mapa-----------------------------------------------------------------------------------------------------------------------------
    #obj 0 Put blue obstacles
    #obj 1 Put ball 
    def put_obj_in_map(self, cx_blue, cy_blue, cz_blue, image_blue, cnt, var_limits_inside, obj):
        self.image_blue = image_blue
        x, y, w, h = cv2.boundingRect(cnt) #Dibujar un rectángulo alrededor del objeto
        cv2.rectangle(self.image_blue, (x, y), (x+w, y+h), (0, 255, 0), 2) #(image, starrpoint, endpoint,color,thickness(-1 fill))
        cv2.circle(self.image_blue, (cx_blue, cy_blue), 5, 255, -1)
        cv2.line(self.image_blue, (cx_blue, cy_blue), (round(self.obj_vision.width/2), round(self.obj_vision.height/2)), 255, 2) #Línea centro del frame al centroide
        cv2.circle(self.image_blue, (x+var_limits_inside, y+round(h/2)), 10, (0, 255, 0), -1)
        cv2.circle(self.image_blue, (x+w-var_limits_inside, y+round(h/2)), 10, (0, 255, 0), -1)
        cv2.line(self.image_blue, (x+var_limits_inside, y+round(h/2)), (x+w-var_limits_inside, y+round(h/2)), (0, 0, 255), 2) #Línea centro del frame al centroide

        width_obj = self.obj_vision.get_width_objs(x, y, w, h, var_limits_inside) #Obtener ancho de objeto
        #print('Result width: '+str(round(width_obj,3)))
        #texted_image =cv2.putText(img=self.image_blue, text="Width: "+str(round(width_obj,3)), org=(100,200),fontFace=2, fontScale=1, color=(0,0,255), thickness=2)
        self.width_obj = round(width_obj,2)

        cz_blue_real = round(self.get_distance_objs(cz_blue),2) #Obtener distacia real
        
        #texted_image_z =cv2.putText(img=self.image_blue, text="Distance: "+str(cz_blue_real), org=(100,50),fontFace=2, fontScale=1, color=(0,255,255), thickness=2)
        self.width_obj = round(width_obj,2)
        x_obs, y_obs = self.get_coordenates_map(cz_blue_real)
        #Afield_obj.Aplot_ball_robot(self.pos_robot_x, self.pos_robot_y, self.pos_ball_x, self.pos_ball_y) #Posiciones del robot, pelota y ruta (rx, ry)
        if obj == 0: #print('Obj blue')
            #pass
            #COMMENT PLOTpass
            self.obj_mapeo.Aplot_obstacle(x_obs, y_obs, self.width_obj) #Ubicar obj, obstacle
        elif obj == 1: #print('Obj ball')
            #pass
            #COMMENT PLOTpass
            self.obj_mapeo.Aplot_ball(x_obs, y_obs)
        return self.image_blue, cz_blue_real, self.width_obj

    def put_ball(self):
        return None

    def put_lines(self):
        return None

    def put_line_goal(self):
        return None

    def put_goal(self):
        return None

    def search_preprocesamiento(self):
        pass

    #Buscar objetos-------------------------------------------------------------------------------------------------------------------------------------
    def search_blue(self, image_blue, color):
        print("search_blue")
        self.x_z_object = 0
        self.x_width_object = 0
        cx, cy = 0, 0
        self.image_blue = image_blue
        frame = cv2.blur(self.image_blue, (15, 15))  # Aplicar desenfoque para eliminar ruido
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV) # Convertir frame de BRG a HSV
        thresh = cv2.inRange(hsv, dic_colors.get("lower_color_"+color), dic_colors.get("upper_color_"+color)) #Aplicar umbral a img y extraer los pixeles en el rango de colores
        cnts, h = cv2.findContours(thresh, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE) # Encontrar los contornos en la imagen extraída
        for cnt in cnts:
            area = cv2.contourArea(cnt)
            if area > 300:
                cx, cy = self.cen_moments(cnt)
                self.image_blue, self.x_z_object, self.x_width_object = self.put_obj_in_map(cx,cy,round(self.obj_vision.depth_image[cy,cx]/10,2), self.image_blue, cnt, self.var_limits_inside_object, 0)
        
        #cv2.imshow('blue object', self.image_blue)
        cv2.imwrite("frameblue_xxx.png", self.image_blue) 
        cv2.imshow('blue object', self.image_blue)
        return self.x_z_object, self.x_width_object

    def search_ball(self, color_image):
        pass
        return None

    def search_lines(self, image_line, color):
        self.image_color = image_line
        filered = cv2.GaussianBlur(self.image_color, (5, 5), 0)  # (7,7),2
        hsv = cv2.cvtColor(filered, cv2.COLOR_BGR2HSV) # Convertir frame de BRG a HSV
        thresh = cv2.inRange(hsv, dic_colors.get("lower_color_"+color), dic_colors.get("upper_color_"+color)) #Aplicar umbral a img y extraer los pixeles en el rango de colores        
        
        ret, thresh_img = cv2.threshold(thresh, 100, 255, cv2.THRESH_BINARY)
        contours, hierachy = cv2.findContours(thresh_img, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area > 100:
                cv2.drawContours(self.image_color, cnt, -1, (255, 255, 0), 5)
        cv2.imwrite('contours.png', self.image_color)
        #cv2.imshow("contours", filered)
        
    # Line ends filter
    def lineEnds(P):
        """Central pixel and just one other must be set to be a line end"""
        return 255 * ((P[4]==255) and np.sum(P)==510)
        
    def seach_lines(self):
        return None

    def search_line_goal(self):
        return None

    def search_goal(self):
        pass

    #Main-----------------------------------------------------------------------------------------------------------------------------------------------
    def main(self):
        x_graphics = []
        y_graphics = []
        num_frames_limit = 60
        num_frames_count = 0

        true_positive_input = 0
        false_positive_input = 0
        true_negative_input = 0
        false_negative_input = 0

        try: #Streaming loop
            while True:

                num_frames_count += 1
                print(num_frames_count)
                if num_frames_count < 10:
                    pass
                else:
                    self.bg_removed = self.obj_vision.get_image_depth() #Eliminar pixeles mayores a 3 metros
                    #self.bg_removed = self.color_image
                    bg_removed_green = self.filter_color(self.bg_removed, "green") #Filtro color verde
                    bg_removed_green_blue = self.filter_color(bg_removed_green, "blue") #Filtro color azul 

                    #new_image = self.seg_superpix(bg_removed_green)

                    print("Search blue main")
                    cz_blue_real, width_object = self.search_blue(bg_removed_green.copy(), "blue") #Search blue obstacles
                    
                    self.search_lines(bg_removed_green_blue.copy(), "white")

                    #cv2.imshow('xxx blue object', self.bg_removed)
                    #cv2.imwrite("framegreen_xxx.png", bg_removed_green_blue) 

                    #cv2.imshow('blue object', img_blue)

                    #self.distance_array_obj.append(cz_blue_real)
                    #self.width_array_obj.append(width_object)

                    #self.graphic_roc()
                    #COMMENT PLOT
                    self.obj_mapeo.Aplot_ball_robot()

                # Salir del bucle si se presiona ESC
                k = cv2.waitKey(5) & 0xFF
                if k == 27:
                    break
                if k == ord("q"):
                    print("That's all folks :)")
                    break

        finally:
            pass

"""if __name__ == '__main__':
    obj_procesamiento = Aprocesamiento()
    #print(str(obja.clipping_distance_in_meters)) #Obtener un valor de la clase
    obj_procesamiento.main()"""