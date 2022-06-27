import cv2
import numpy as np
import time

def Removeglare(img):
  clahefilter = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(16,16))




  ## crop if required 
  #FACE
  x,y,h,w = 550,250,400,300
  # img = img[y:y+h, x:x+w]

  #NORMAL
  # convert to gray
  gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
  grayimg = gray


  GLARE_MIN = np.array([0, 0, 50],np.uint8)
  GLARE_MAX = np.array([0, 0, 225],np.uint8)

  hsv_img = cv2.cvtColor(img,cv2.COLOR_BGR2HSV)

  #HSV
  frame_threshed = cv2.inRange(hsv_img, GLARE_MIN, GLARE_MAX)


  #INPAINT
  mask1 = cv2.threshold(grayimg , 220, 255, cv2.THRESH_BINARY)[1]
  result1 = cv2.inpaint(img, mask1, 0.1, cv2.INPAINT_TELEA) 



  #CLAHE
  claheCorrecttedFrame = clahefilter.apply(grayimg)

  #COLOR 
  lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
  lab_planes = cv2.split(lab)
  clahe = cv2.createCLAHE(clipLimit=2.0,tileGridSize=(8,8))
  lab_planes[0] = clahe.apply(lab_planes[0])
  lab = cv2.merge(lab_planes)
  clahe_bgr = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)


  #INPAINT + HSV
  result = cv2.inpaint(img, frame_threshed, 0.1, cv2.INPAINT_TELEA) 


  #INPAINT + CLAHE
  grayimg1 = cv2.cvtColor(clahe_bgr, cv2.COLOR_BGR2GRAY)
  mask2 = cv2.threshold(grayimg1 , 220, 255, cv2.THRESH_BINARY)[1]
  result2 = cv2.inpaint(img, mask2, 0.1, cv2.INPAINT_TELEA) 



  #HSV+ INPAINT + CLAHE
  lab1 = cv2.cvtColor(result, cv2.COLOR_BGR2LAB)
  lab_planes1 = cv2.split(lab1)
  clahe1 = cv2.createCLAHE(clipLimit=2.0,tileGridSize=(8,8))
  lab_planes1[0] = clahe1.apply(lab_planes1[0])
  lab1 = cv2.merge(lab_planes1)
  clahe_bgr1 = cv2.cvtColor(lab1, cv2.COLOR_LAB2BGR)




  # fps = 1./(time.time()-t1)
  # cv2.putText(clahe_bgr1    , "FPS: {:.2f}".format(fps), (10, 180), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 0, 255))    

  
  cv2_imshow(result)
  cv2_imshow(result1)
  cv2_imshow(result2)
  return result2

def get_grayscale(image):
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
def blur(image):
    return cv2.medianBlur(image,3)
def thresholding(image):
    return cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]

from typing import Tuple
import numpy as np
import cv2
from PIL import Image, ImageDraw, ImageFont
 
def cv2_img_add_text(img, text, left_corner: Tuple[int, int],
                     text_rgb_color=(255, 0, 0), text_size=24, font='Roboto-Regular.ttf', **option):
    pil_img = img
    if isinstance(pil_img, np.ndarray):
        pil_img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(pil_img)
    font_text = ImageFont.truetype(font=font, size=text_size, encoding=option.get('encoding', 'utf-8'))
    draw.text(left_corner, text, text_rgb_color, font=font_text)
    cv2_img = cv2.cvtColor(np.asarray(pil_img), cv2.COLOR_RGB2BGR)
    if option.get('replace'):
        img[:] = cv2_img[:]
        return None
    return cv2_img