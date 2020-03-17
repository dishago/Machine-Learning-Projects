#!/usr/bin/env python
# coding: utf-8

# In[2]:


import cv2
import numpy as np

face_cascade = cv2.CascadeClassifier(r"C:/Users/insp/Documents/College/ML/Face Recognition/haarcascade_frontalface_alt.xml")
cap = cv2.VideoCapture(0) #0 tells which webam, 0 for default
skip = 0
face_data = []
dataset_path = 'C:/Users/insp/Documents/College/ML/Face Recognition/'
file_name = input("Enter name: ")
while(True):
    ret, frame = cap.read() #ret tells if webcam working
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    if(ret==False):
        continue
    faces = face_cascade.detectMultiScale(gray_frame,1.3,5) #frame,scaling factor,number of neighbors
    faces = sorted(faces, key = lambda f:f[2]*f[3]) #sort according to face area w*h
    
    for (x,y,w,h) in faces[-1:]: #consider largest face first
        cv2.rectangle(frame,(x,y),(x+w,y+h),(255,0,0),2)
        offset = 10
        face_section = frame[y-offset:y+h+offset, x-offset:x+w+offset]
        face_section = cv2.resize(face_section,(100,100))
        skip = skip+1
        if skip%10==0:
            face_data.append(face_section)
            print(len(face_data))
    
    cv2.imshow("Video", frame)
    #cv2.imshow("Video Gray", gray_frame)
    key_pressed = cv2.waitKey(1) & 0xFF #first 8 bits considered out of 32
    if key_pressed == ord('q'): #ord tells ASCII value 
        break;

face_data = np.asarray(face_data)
face_data = face_data.reshape((face_data.shape[0],-1))
print(face_data.shape)
np.save(dataset_path+file_name+'.npy',face_data)
print("Data successfully saved")

cap.release()
cv2.destroyAllWindows()


# In[3]:


def distance(x1,x2):
    return np.sqrt(((x1-x2)**2).sum())

def knn(train, test,k=5):
    vals = []
    m = train.shape[0]
    for i in range(m):
        ix = train[i, :-1]
        iy = train[i, -1]
        d = distance(test,ix)
        vals.append([d,iy])
    dk = sorted(vals, key = lambda x:x[0])[:k]
    labels = np.array(dk)[:,-1]
    new_vals = np.unique(labels, return_counts=True)
    index = np.argmax(new_vals[1])
    pred = new_vals[0][index]
    return pred


# In[4]:


import os

face_cascade = cv2.CascadeClassifier(r"C:/Users/insp/Documents/College/ML/Face Recognition/haarcascade_frontalface_alt.xml")
cap = cv2.VideoCapture(0) #0 tells which webam, 0 for default
skip = 0

face_data = []
label = []

class_id = 0      #labels for given file
names = {}        #mapping between id-name

dataset_path = 'C:/Users/insp/Documents/College/ML/Face Recognition/'

#Data Preparation
for fx in os.listdir(dataset_path):
    if fx.endswith('.npy'):
        names[class_id] = fx[:-4] #Mapping between class label and output
        print("Loaded "+fx)
        data_item = np.load(dataset_path+fx)
        face_data.append(data_item)
        target = class_id*np.ones((data_item.shape[0],))
        class_id = class_id+1
        label.append(target)

face_dataset = np.concatenate(face_data,axis=0)
face_labels = np.concatenate(label, axis=0).reshape((-1,1))
print(face_dataset.shape)
print(face_labels.shape)

trainset = np.concatenate((face_dataset, face_labels), axis=1)
print(trainset.shape)

while True:
    ret, frame = cap.read() #ret tells if webcam working
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    if(ret==False):
        continue
    faces = face_cascade.detectMultiScale(gray_frame,1.3,5) #frame,scaling factor,number of neighbors
    for face in faces:
        x,y,w,h = face
        offset = 10
        face_section = frame[y-offset:y+h+offset, x-offset:x+w+offset]
        face_section = cv2.resize(face_section,(100,100))
        out = knn(trainset, face_section.flatten())
        pred_name = names[int(out)]
        cv2.putText(frame, pred_name, (x,y-10), cv2.FONT_HERSHEY_SIMPLEX, 1,(255,0,0),2,cv2.LINE_AA)
        cv2.rectangle(frame, (x,y), (x+w,y+h), (0,255,255),2)
    cv2.imshow("Faces",frame)
    key_pressed = cv2.waitKey(1) & 0xFF #first 8 bits considered out of 32
    if key_pressed == ord('q'): #ord tells ASCII value 
        break;
cap.release()
cv2.destroyAllWindows()


# In[ ]:




