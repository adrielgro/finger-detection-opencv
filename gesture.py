import cv2
import numpy as np
import math


cap = cv2.VideoCapture(1)
while(cap.isOpened()):
    # leemos el frame
    ret, img = cap.read()

    # obtiene datos de mano desde la ventana secundaria del rectangulo en la pantalla
    cv2.rectangle(img, (300,300), (100,100), (0,255,0),0)
    crop_img = img[100:300, 100:300]

    # convertir a escala de grisis
    grey = cv2.cvtColor(crop_img, cv2.COLOR_BGR2GRAY)

    # aplica efecto de desenfoque gussiano 
    value = (35, 35)
    blurred = cv2.GaussianBlur(grey, value, 0)

    # aplicamos el efecto umbral, con el metodo de Otsu
    _, thresh1 = cv2.threshold(blurred, 127, 255,
                               cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)

    # mostrar imagen con umbral
    cv2.imshow('Umbral', thresh1)

    # compruebe la version de OpenCV para evitar desempaquetar el error
    (version, _, _) = cv2.__version__.split('.')

    if version == '3':
        image, contours, hierarchy = cv2.findContours(thresh1.copy(), \
               cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    elif version == '2':
        contours, hierarchy = cv2.findContours(thresh1.copy(),cv2.RETR_TREE, \
               cv2.CHAIN_APPROX_NONE)

    # encontrar el contorno con el area maxima
    cnt = max(contours, key = lambda x: cv2.contourArea(x))

    # crea un rectangulo delimitador alrededor del contorno (puede omitir las dos lineas debajo)
    #x, y, w, h = cv2.boundingRect(cnt)
    #cv2.rectangle(crop_img, (x, y), (x+w, y+h), (0, 0, 255), 0)

    # encontrar un hull convexo
    hull = cv2.convexHull(cnt)

    # dibuja los contornos
    drawing = np.zeros(crop_img.shape,np.uint8)
    cv2.drawContours(drawing, [cnt], 0, (0, 255, 0), 0)
    cv2.drawContours(drawing, [hull], 0,(0, 0, 255), 0)

    # encontrando el casco convexo
    hull = cv2.convexHull(cnt, returnPoints=False)

    # encontrando los defectos convexos
    defects = cv2.convexityDefects(cnt, hull)
    count_defects = 0
    cv2.drawContours(thresh1, contours, -1, (0, 255, 0), 3)

    # aplicando la regla de coseno para encontrar el angulo de todos los defectos (entre los dedos)
    # con un angulo > 90 grados e ignorar defectos
    for i in range(defects.shape[0]):
        s,e,f,d = defects[i,0]

        start = tuple(cnt[s][0])
        end = tuple(cnt[e][0])
        far = tuple(cnt[f][0])

        # encontrar la longitud de todos los lados del triangulo
        a = math.sqrt((end[0] - start[0])**2 + (end[1] - start[1])**2)
        b = math.sqrt((far[0] - start[0])**2 + (far[1] - start[1])**2)
        c = math.sqrt((end[0] - far[0])**2 + (end[1] - far[1])**2)

        # aplicar la regla del coseno aqui
        angle = math.acos((b**2 + c**2 - a**2)/(2*b*c)) * 57

        # ignorar angulos > 90 y resalta el resto con puntos rojos
        if angle <= 90:
            count_defects += 1
            cv2.circle(crop_img, far, 1, [0,0,255], -1)
        #dist = cv2.pointPolygonTest(cnt,far,True)

        # dibuja una linea de principio a fin, es decir, los puntos convexos (puntas de los dedos)
        # (se puede saltar esta parte)
        cv2.line(crop_img,start, end, [0,255,0], 2)
        #cv2.circle(crop_img,far,5,[0,0,255],-1)

    # definir las acciones requeridas
    if count_defects == 0:
        cv2.putText(img,"1 dedo", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 2, 2)
    elif count_defects == 1:
        cv2.putText(img,"2 dedos", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 2, 2)
    elif count_defects == 2:
        cv2.putText(img, "3 dedos", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 2, 2)
    elif count_defects == 3:
        cv2.putText(img,"4 dedos", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 2, 2)
    elif count_defects == 4:
        cv2.putText(img,"5 dedos", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 2, 2)
    else:
        cv2.putText(img,"Cero vertices", (22, 50), cv2.FONT_HERSHEY_SIMPLEX, 2, 2)

    # mostrar imagenes apropiadas en ventanas
    cv2.imshow('Gestos', img)
    all_img = np.hstack((drawing, crop_img))
    cv2.imshow('Contorno', all_img)
    
    key = cv2.waitKey(1) & 0xFF
    if key == ord("q"):
        break
    
cv2.destroyAllWindows()
