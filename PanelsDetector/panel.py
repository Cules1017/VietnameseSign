import torch
import cv2
import matplotlib.pyplot as plt
from google.colab.patches import cv2_imshow
import requests
from pathlib import Path
import matplotlib as mpl
import matplotlib.font_manager as fm
import numpy as np
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
class panel():
  def __init__(self,image,x1,y1,x2,y2,text,color):
    self.image=image
    self.x1_sta=x1
    self.y1_sta=y1
    self.x2_end=x2
    self.y2_end=y2
    self.text=text
    self.color=color
   
  def drawResult(self):
    if(self.text!=''):
      sta_pan=(self.x1_sta,self.y1_sta)
      end_pan=(self.x2_end,self.y2_end)
      if self.y1_sta<20:
        sta_lab=(self.x1_sta,self.y1_sta)
        end_lab=(self.x1_sta+len(self.text)*8,self.y1_sta+20)
      else:
        sta_lab=(self.x1_sta,self.y1_sta-20)
        end_lab=(self.x1_sta+len(self.text)*8,self.y1_sta)
      self.image = cv2.rectangle(self.image,sta_pan, end_pan, self.color, 2)
      self.image = cv2.rectangle(self.image, sta_lab, end_lab, self.color, -1)
      self.image =cv2_img_add_text(self.image, self.text, sta_lab , text_rgb_color=(255, 255, 255), text_size=14)
      
    return self.image