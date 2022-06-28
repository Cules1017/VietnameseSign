from sympy import symbols, Eq, solve
from vietocr.tool.predictor import Predictor
from vietocr.tool.config import Cfg
import glob
from PIL import Image
from google.colab.patches import cv2_imshow
import cv2
import torch
from PIL import Image
from sympy import symbols, Eq, solve
import torch
import torchvision as tv
from pyvi import ViTokenizer, ViPosTagger,ViUtils

def FindLinearEquation(label1,label2):
  ass,bss = symbols('asd,bsd')
  eq1= Eq((label1[0]*ass+bss),label1[1])
  eq2= Eq((label2[0]*ass+bss),label2[1])
  dictt=solve((eq1, eq2), (ass, bss),set=True)
  return dictt
def convertToXYXY(label):
  xyx=torch.tensor([label[0]-label[2]/2,label[1]-label[3]/2,label[0]+label[2]/2,label[1]+label[3]/2,label[4],label[5]],dtype=torch.int32)
  return xyx
import numpy as np
from vietocr.tool.predictor import Predictor
from vietocr.tool.config import Cfg
import glob
from PIL import Image
import cv2
from google.colab.patches import cv2_imshow
import torch
from PIL import Image
def distanceToTopLeft(x,y):
  dis=( x**2 + y**2 )**(1/2)
  return dis
def distanceToTopRight(x,y,imgWidth):
  dis=( (imgWidth - x)**2 + y**2 )**(1/2)
  return dis
def distanceBetween(x1,y1,x2,y2):
  dis=((x1 - x2)**2 + (y1 - y2)**2)**(1/2)
  return dis
def onSameRow(label,firstWord,endWord):
  bisa=FindLinearEquation(firstWord,endWord)
  # print(bisa)
  for i in bisa[1]:
    coef=i[0]
    biasB=i[1]
  
  yOnLine=label[0]*coef + biasB

  if (label[1] - yOnLine)**2 < (label[3]/2)**2:
    return True
  # print(coef,biasB)
  return False
def onSameRow2(firstWord,nearestWord):
  # print(val1,val2,firstWord[0],nearestWord[0])
  bisa=FindLinearEquation(firstWord,nearestWord)
  # print(bisa)
  for i in bisa[1]:
    coef=i[0]
    biasB=i[1]
  angle=np.arctan(float(coef))/np.pi*180
  if (angle <20) & (angle >-20) and (nearestWord[0]>firstWord[0]) and (firstWord[3]*1.2>nearestWord[3]) and (firstWord[3]*0.8<nearestWord[3]):
    return True
  return False

class predictedImage():
  def __init__(self,vietOcrConfig):
    super()
    self.vietOcrConfig=vietOcrConfig
    self.detector = Predictor(self.vietOcrConfig)
  def predictWord(self,label,img):
    left=label[0]
    top=label[1]
    right=label[2]
    bottom=label[3]
    cropped=img[top:bottom,left:right]
    # cv2_imshow(cropped)
    cImg=Image.fromarray(cropped)
    textPredict=self.detector.predict(cImg)
    # print(textPredict)
    return textPredict
  def detect(self,results,imgs):
    listCenter=results.xywh[0]
    fullSentences=[]
    listTemp=results.xywh[0]
    listRemoved=[]
    while len(listTemp) > 0:
      if len(listTemp) > 1:
        temp=[]
        minDisTL=10000
        minDisTR=10000
        minTop=10000
        for i in range(len(listTemp)):
          if convertToXYXY(listTemp[i])[1]<minTop:
            minTop=convertToXYXY(listTemp[i])[1]
            minTopc=convertToXYXY(listTemp[i])
            minTopIndex=i
        listFirstLine=[]
        for i in range(len(listTemp)):
          if listTemp[i][1]<minTopc[3]:
            listFirstLine.append(listTemp[i])
        for i in range(len(listFirstLine)):
          if(minDisTL>distanceToTopLeft(listFirstLine[i][0],listFirstLine[i][1])):
            minDisTL=distanceToTopLeft(listFirstLine[i][0],listFirstLine[i][1])
            firstWord=convertToXYXY(listFirstLine[i])
            firstWordc=listFirstLine[i]
            firstWordIndex=i
        closestDis=10000
        curWord=firstWordc
        curWIndex=firstWordIndex
        rowsw=[]
        for i in range(len(listTemp)):
          trys=0
          flag=True
          for _ in range(20):
            reAssign=False
            if flag:
              flag=False
              # print('word ready',self.predictWord(convertToXYXY(curWord),imgs))
              # print('word ready to del',self.predictWord(convertToXYXY(listTemp[curWIndex]),imgs))
              listTemp=torch.cat([listTemp[0:curWIndex],listTemp[curWIndex+1:]])
            for j in range(len(listTemp)):
              if(closestDis>distanceBetween(curWord[0],curWord[1],listTemp[j][0] , listTemp[j][1])):
                if distanceBetween(curWord[0],curWord[1],listTemp[j][0],listTemp[j][1])>0:
                  closestDis=distanceBetween(curWord[0],curWord[1],listTemp[j][0] , listTemp[j][1])
                  closestPoint=listTemp[j]
                  closestPointIndex=j
                  reAssign=True
            # print('attemp:',trys)
            if (distanceBetween(curWord[0],curWord[1],closestPoint[0],closestPoint[1])>0) & (reAssign==True):
              # if onSameRow2(curWord,closestPoint):
              if onSameRow2(curWord,closestPoint,self.predictWord(convertToXYXY(curWord),imgs),self.predictWord(convertToXYXY(closestPoint),imgs)):
                rowsw.append(curWord)
                curWord=closestPoint
                curWIndex=closestPointIndex
                # print('ready',self.predictWord(convertToXYXY(curWord),imgs))
                # print('to del',self.predictWord(convertToXYXY(listTemp[curWIndex]),imgs))
                closestDis=10000
                break
              else:
                listTemp=torch.cat([listTemp[0:closestPointIndex],listTemp[closestPointIndex+1:]])
                closestDis=10000
                trys+=1
            else:
              listTemp=torch.cat([listTemp[0:closestPointIndex],listTemp[closestPointIndex+1:]])
              closestDis=10000
              trys+=1
          # print('final try',trys)
          if trys==20:
            rowsw.append(curWord)
            break
        for i in rowsw:
          listRemoved.append(i)
        listTemp=[]
        for i in results.xywh[0]:
          flag=True
          for j in listRemoved:
            if(sum(i) == sum(j)):
              flag=False
          if flag:
            listTemp.append(i)
            
        if len(listTemp) != 0 :
          listTemp = torch.stack(listTemp)
        
        arow=[]
        for i in range(len(rowsw)):
          arow.append(self.predictWord(convertToXYXY(rowsw[i]),imgs))
        print('row',arow)
        fullSentences.append(arow)
      else: 
        fullSentences.append(self.predictWord(convertToXYXY(listTemp[0]),imgs))
        listTemp = []
    cv2_imshow(results.render()[0])
    return fullSentences
      



class wordDetect():
  def __init__(self,modelPath):

    self.DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"
    self.model = torch.hub.load('ultralytics/yolov5', 'custom',path=modelPath,device=self.DEVICE)
    
    self.config = Cfg.load_config_from_name('vgg_transformer')
    # config['weights'] = './weights/transformerocr.pth'
    self.config['weights'] = 'https://drive.google.com/uc?id=13327Y1tz1ohsm5YZMyXVMPIOjoOA0OaA'
    self.config['cnn']['pretrained']=True
    self.config['device'] = self.DEVICE
    self.config['predictor']['beamsearch']=True
    self.imageP=predictedImage(self.config)

  def detect(self,img):
    results = self.model(img, size=640)
    sentence=self.imageP.detect(results,img)
    newSen=''
    for sen in sentence:
      newSen=newSen+' '.join(sen)+' '
    sepSentence=ViTokenizer.tokenize(newSen)
    return sepSentence
