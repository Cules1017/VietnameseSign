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

def distanceToTopLeft(x,y):
  dis=( x**2 + y**2 )**(1/2)
  return dis


def distanceToTopRight(x,y,imgWidth):
  dis=( (imgWidth - x)**2 + y**2 )**(1/2)
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

def convertToXYXY(label):
  xyx=torch.tensor([label[0]-label[2]/2,label[1]-label[3]/2,label[0]+label[2]/2,label[1]+label[3]/2,label[4],label[5]],dtype=torch.int32)
  return xyx



class predictedImage():
  def __init__(self,vietOcrConfig):
    super()
    self.vietOcrConfig=vietOcrConfig
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
    self.detector = Predictor(self.vietOcrConfig)
    fullSentences=[]
    while len(listCenter) > 0:
      if len(listCenter) > 1:
        # print(listCenter)
        temp=[]
        minDisTL=10000
        minDisTR=10000
        for i in range(len(listCenter)):
          if(minDisTL>distanceToTopLeft(listCenter[i][0],listCenter[i][1])):
            minDisTL=distanceToTopLeft(listCenter[i][0],listCenter[i][1])
            firstWord=convertToXYXY(listCenter[i])
            firstWordc=listCenter[i]
            firstWordIndex=i
        for i in range(len(listCenter)):
          if ( minDisTR > distanceToTopRight( listCenter[i][0] , listCenter[i][1] , 416)):
            minDisTR=distanceToTopRight(listCenter[i][0] , listCenter[i][1] , 416)
            endWord=convertToXYXY(listCenter[i])
            endWordc = listCenter[i]
            endWordIndex=i
        if(firstWordIndex!=endWordIndex):
          rowtes=[]
          for i in range(len(listCenter)):
            if onSameRow(listCenter[i],firstWordc,endWordc):
              rowtes.append(convertToXYXY(listCenter[i]))
              temp.append(i)
          def sortWord(label):
            return (label[0]+label[2])/2
          # for i in range(len(rowtes)):
          #   self.predictWord(rowtes[i],imgs)
          # print('row1')
          rowtes.sort(key=sortWord)
          for i in range(len(rowtes)):
            fullSentences.append(self.predictWord(rowtes[i],imgs))
          # print('row2')
          tempRv=[]
          for i in range(len(listCenter)):
            if i not in temp:
              tempRv.append(i)
          listCenter=listCenter[tempRv]
        else:
          fullSentences.append(self.predictWord(convertToXYXY(listCenter[firstWordIndex]),imgs))
          tempRv=[]
          for i in range(len(listCenter)):
            if i not in [firstWordIndex]:
              tempRv.append(i)
          listCenter=listCenter[tempRv]
      else: 
        fullSentences.append(self.predictWord(convertToXYXY(listCenter[0]),imgs))
        listCenter = []
    # print(convertToXYXY(listCenter)
    # print(imgs.shape)
    # for i in range(len(convertToXYXY(listCenter)):
    #   self.predictWord(convertToXYXY(listCenter[i]),imgs)
    cv2_imshow(results.render()[0])
    return fullSentences


def wordDetect(modelPath,img):
    DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"
    # DEVICE='cpu'
    model = torch.hub.load('ultralytics/yolov5', 'custom',path=modelPath,device=DEVICE)
    results = model(img, size=640)
    config = Cfg.load_config_from_name('vgg_transformer')
    # config['weights'] = './weights/transformerocr.pth'
    config['weights'] = 'https://drive.google.com/uc?id=13327Y1tz1ohsm5YZMyXVMPIOjoOA0OaA'
    config['cnn']['pretrained']=True
    config['device'] = DEVICE
    config['predictor']['beamsearch']=True
    imageP=predictedImage(config)
    sentence=imageP.detect(results,img)
    sepSentence=ViTokenizer.tokenize(' '.join(sentence))
    return sepSentence