import torch
import torch.nn.functional as F
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import numpy as np
from vietocr.tool.predictor import Predictor
from vietocr.tool.config import Cfg
from google.colab.patches import cv2_imshow
import torch
from PIL import Image
from pyvi import ViTokenizer, ViPosTagger,ViUtils

from detectorv2 import predictedImage as pI

def padding(data,type):
  if type=='data':
    pad=[0,0,0,0]
  elif type=='label':
    pad=0
  
  k=20
  for i in range(len(data)):
    if len(data[i])<k:
      for n in range(0,k-len(data[i])):
        data[i].append(pad)
  return data
def convertToXYXY(label):
  xyx=torch.tensor([label[0]-label[2]/2,label[1]-label[3]/2,label[0]+label[2]/2,label[1]+label[3]/2,label[4],label[5]],dtype=torch.int32)
  return xyx

# Luong attention layer
class Attn(torch.nn.Module):
    def __init__(self, method, hidden_size):
        super(Attn, self).__init__()
        self.method = method
        if self.method not in ['dot', 'general', 'concat']:
            raise ValueError(self.method, "is not an appropriate attention method.")
        self.hidden_size = hidden_size
        if self.method == 'general':
            self.attn = torch.nn.Linear(self.hidden_size, hidden_size)
        elif self.method == 'concat':
            self.attn = torch.nn.Linear(self.hidden_size * 2, hidden_size)
            self.v = torch.nn.Parameter(torch.FloatTensor(hidden_size))

    def dot_score(self, hidden, encoder_output):
        return torch.sum(hidden * encoder_output, dim=2)

    def general_score(self, hidden, encoder_output):
        energy = self.attn(encoder_output)
        return torch.sum(hidden * energy, dim=2)

    def concat_score(self, hidden, encoder_output):
        energy = self.attn(torch.cat((hidden.expand(encoder_output.size(0), -1, -1), encoder_output), 2)).tanh()
        return torch.sum(self.v * energy, dim=2)

    def forward(self, hidden, encoder_outputs):
        # Calculate the attention weights (energies) based on the given method
        if self.method == 'general':
            attn_energies = self.general_score(hidden, encoder_outputs)
        elif self.method == 'concat':
            attn_energies = self.concat_score(hidden, encoder_outputs)
        elif self.method == 'dot':
            attn_energies = self.dot_score(hidden, encoder_outputs)

        # Transpose max_length and batch_size dimensions
        attn_energies = attn_energies.t()

        # Return the softmax normalized probability scores (with added dimension)
        return F.softmax(attn_energies, dim=1).unsqueeze(1)



class LSTM(torch.nn.Module):
    def __init__(self, seqLen ,DEVICE):
        super(LSTM, self).__init__()
        self.DEVICE = DEVICE
        self.seqLen=seqLen
        self.dropout = torch.nn.Dropout(0.3)
        self.expandEnc = torch.nn.Linear(4, 256)
        
        self.enclstm = torch.nn.LSTM(256,516,2,dropout=0.5,bidirectional=True)

        self.declstm = torch.nn.LSTM(20,516,2,dropout=0.5,bidirectional=True)
        self.decout = torch.nn.Linear(1032, 20)
    



        self.concat = torch.nn.Linear(1032 * 2, 1032)
        # self.out = torch.nn.Linear(256, len(vocabidx_y))

        self.attn = Attn('concat', 1032)


    def forward(self,x,y,label):
        x = self.expandEnc(x)
        outenc,(hidden,cell) = self.enclstm(x)
        # print(outenc.shape,hidden.shape,cell.shape)
        n_y=y.shape[0]
        outputs = torch.zeros(n_y,x.shape[1],self.seqLen).to(self.DEVICE)
        loss = torch.tensor(0.,dtype=torch.float32).to(self.DEVICE)
        for i in range(n_y):
            input = y[i]
            input = input.unsqueeze(0)
            outdec, (hidden,cell) = self.declstm(input,(hidden,cell))

            attn_weights = self.attn(outdec, outenc)
            # Multiply attention weights to encoder outputs to get new "weighted sum" context vector
            context = attn_weights.bmm(outenc.transpose(0, 1))
            # Concatenate weighted context vector and GRU output using Luong eq. 5
            outdec = outdec.squeeze(0)
            context = context.squeeze(1)
            concat_input = torch.cat((outdec, context), 1)
            concat_output = torch.tanh(self.concat(concat_input))
            # Predict next word using Luong eq. 6



            output = self.decout(concat_output)
            loss += F.cross_entropy(output, label[i])

        return loss

    def evaluate(self,x):

        x = self.expandEnc(x)
        # print(x.shape)

        outenc,(hidden,cell)=self.enclstm(x)
        
        y = torch.zeros((1,1,self.seqLen)).to(self.DEVICE)
        pred=[]
        for i in range(self.seqLen):

            outdec,(hidden,cell)= self.declstm(y,(hidden,cell))
            output = self.decout(outdec.squeeze(0))  

            attn_weights = self.attn(outdec, outenc)
            # Multiply attention weights to encoder outputs to get new "weighted sum" context vector
            context = attn_weights.bmm(outenc.transpose(0, 1))
            # Concatenate weighted context vector and GRU output using Luong eq. 5
            outdec = outdec.squeeze(0)
            context = context.squeeze(1)
            concat_input = torch.cat((outdec, context), 1)
            concat_output = torch.tanh(self.concat(concat_input))
            # Predict next word using Luong eq. 6
            output = self.decout(concat_output)

            pred_id = output.squeeze().argmax().item()
            pred.append(pred_id)
            y[0,0,i]=pred_id
        return pred

    def evaluate2(self,x,y):
        outenc,(hidden,cell)=self.enclstm(x)
        pred=[]
        for i in range(self.seqLen):
            input=y[i].view(1,1,-1)
            outdec,(hidden,cell)= self.declstm(input,(hidden,cell))
            attn_weights = self.attn(outdec, outenc)
            # Multiply attention weights to encoder outputs to get new "weighted sum" context vector
            context = attn_weights.bmm(outenc.transpose(0, 1))
            # Concatenate weighted context vector and GRU output using Luong eq. 5
            outdec = outdec.squeeze(0)
            context = context.squeeze(1)
            concat_input = torch.cat((outdec, context), 1)
            concat_output = torch.tanh(self.concat(concat_input))
            # Predict next word using Luong eq. 6
            output = self.decout(concat_output)

            pred_id = output.squeeze().argmax().item()
            pred.append(pred_id)
        return pred


class predictedImage():
  def __init__(self,vietOcrConfig,sortModelPath,DEVICE):
    super()
    self.DEVICE = DEVICE

    self.vietOcrConfig=vietOcrConfig
    self.detector = Predictor(self.vietOcrConfig)
    self.arrangeModel = LSTM(20,self.DEVICE).to(self.DEVICE)
    self.arrangeModel.load_state_dict(torch.load(sortModelPath))
    self.arrangeModel.eval()
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
  def detect(self,results,imgs, display = False):
    listCenter=results.xywh[0]
    if len(listCenter > 20):
      listCenter = listCenter[0:20]
    senLen=listCenter.shape[0]
    listNP=[np.delete(listCenter.cpu().numpy(),[4,5],1).tolist()]
    scaledData=[]
    for i in range(len(listNP)):
      scaler=MinMaxScaler((1,2))
      scaler.fit(listNP[i])
      scaledData.append(scaler.transform(listNP[i]).tolist())

    scaledData=np.array(padding(scaledData,'data'))
    # print(scaledData)
    x=torch.tensor(scaledData,dtype=torch.float32).transpose(1,0).to(self.DEVICE)
    # print(x.shape)
    predRes=self.arrangeModel.evaluate(x)
    sorted=predRes
    # print(sorted)
    SENTENCES=[]
    for i in range(senLen):
      if sorted[i] <senLen :
        SENTENCES.append(self.predictWord(convertToXYXY(listCenter[sorted[i]-1]),imgs))
    # print(SENTENCES)
    if display :
      cv2_imshow(results.render()[0])
    return SENTENCES
      
class wordDetect():
  def __init__(self,modelPath,sortModelPath):

    self.DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"
    self.model = torch.hub.load('ultralytics/yolov5', 'custom',path=modelPath,device=self.DEVICE)
    
    self.config = Cfg.load_config_from_name('vgg_transformer')
    # config['weights'] = './weights/transformerocr.pth'
    self.config['weights'] = 'https://drive.google.com/uc?id=13327Y1tz1ohsm5YZMyXVMPIOjoOA0OaA'
    self.config['cnn']['pretrained']=True
    self.config['device'] = self.DEVICE
    self.config['predictor']['beamsearch']=True
    self.imageP=predictedImage(self.config,sortModelPath,self.DEVICE)
    self.imageP2=pI(self.config)

  def detect(self,img ,display = False):
    results = self.model(img, size=640)
    # print('len word :',len(results.xywh[0]))

    if len(results.xywh[0]) > 0 :
      if len(results.xywh[0]) > 10 :
        sentence=self.imageP2.detect(results,img)
        newSen=''
        for sen in sentence:
          newSen=newSen+' '.join(sen)+' '
        sepSentence=ViTokenizer.tokenize(newSen)
        return sepSentence
      else:
        sentence = self.imageP.detect(results,img,display = display)
        newSen=''
        for sen in sentence:
          newSen=newSen+sen+' '
        sepSentence=ViTokenizer.tokenize(newSen)
        return sepSentence
    return ''

 
