from imageai.Detection import ObjectDetection
import os
import dwave_networkx as dnx
import matplotlib.pyplot as plt
execution_path = os.getcwd()
detector = ObjectDetection()
detector.setModelTypeAsRetinaNet()
detector.setModelPath( os.path.join(execution_path , "resnet50_coco_best_v2.1.0.h5"))
detector.loadModel()
detections = detector.detectObjectsFromImage(input_image=os.path.join(execution_path , "image.jpeg"), output_image_path=os.path.join(execution_path , "imagenew.jpg"))
prob=[]
for eachObject in detections:
    print(eachObject["name"] , " : " , eachObject["percentage_probability"] )
    prob.append(eachObject["percentage_probability"])
import sys
from random import *
import numpy as np
from math import *
import time
from matplotlib import pyplot as plt
class  Neural_Network(object):
    def __init__(self,n):
        self.inputlayers=n
        self.outputlayers=1
        self.hiddenlayers=3
        self.w1=np.random.randn(self.inputlayers,self.hiddenlayers)
        self.w2=np.random.randn(self.hiddenlayers,self.outputlayers)
    def forward(self,X):
        self.z2=np.dot(X,self.w1)
        self.a2=self.myactfun(self.z2,2)
        self.z3=np.dot(self.a2,self.w2)
        yvect=self.myactfun(self.z3,3)
        return yvect
    def sigmoid(self,Z):
        return 1/(1+np.exp(-Z))
    def myactfun(self,Z,n):
        sm=0.0
        for i in range(0,n+1):
            sm+=pow(pi,i+1)*np.exp(-i*Z)
        return 1/sm
X=np.array((prob),dtype=int)
nn=Neural_Network(len(prob))
yvect=nn.forward(X)
print("\nThe final precised vector is:\t"+str(yvect))
weights=np.linspace(-10,10,1000)
weight=np.linspace(-5,5,1000)
c1=np.linspace(-0.1,1.9,5)
times=np.linspace(0.0,0.9,5)
costs=np.zeros(1000)
starttime=time.clock()
y=np.array([1])
res={0:"Not detected . We need further training for the same.",1:"Cross safely . There is no crowd.",2:"Wait ! Its clumpsy . I will tell you.",3: "Its crowdy . Don't go now . I will tell you ."}
print("Our actul assumption is:\t"+str(y))
accur=0.0
for i in range(1000):
    nn.w1[0,0]=weights[i]
    yvect=nn.forward(X)
    costs[i]=0.5*sum((y-yvect)**2)
    accur+=sum(y-yvect)/sum(y)
print(costs)
endtime=time.clock()
timeelapsed=endtime-starttime
print(timeelapsed)
x=np.array([[0.5],[1],[1.5]])
print("\n The minimum cost is:\t"+str(min(costs)))
print("\n The accuracy is "+str(accur/10))
print(str(res[(int(yvect[0]*1000)%4)]))
import pyttsx3
engine = pyttsx3.init()
print(int(yvect[0]*1000))
engine.say(str(res[(int(yvect[0]*1000)%4)]))
engine.runAndWait()
import dwave_networkx as dnx
import matplotlib.pyplot as plt
graph = dnx.chimera_graph(1, 1, 4)
dnx.draw_chimera(graph)
plt.title('Road network')
plt.show()
plt.subplot(2,2,1)
plt.plot(weights,costs)
plt.xlabel('Weight')
plt.ylabel('Cost')
#plt.show()
plt.subplot(2,2,3)
plt.scatter(weight,costs)
plt.xlabel('Accuracy')
plt.ylabel('Cost')
#plt.show()
plt.subplot(2,2,2)
plt.bar(c1,times,width=0.8)
plt.xlabel('Cost')
plt.ylabel('Timeelapsed')
plt.show()
