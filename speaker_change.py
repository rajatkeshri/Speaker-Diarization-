import librosa
import numpy
import librosa
import math

import matplotlib.pyplot as plt  
import numpy as np  
from sklearn import cluster
from sklearn import cluster
from scipy.spatial import distance
import sklearn.datasets
from sklearn.preprocessing import StandardScaler
import numpy as np
import librosa.display
import pandas as pd


def speaker_change_detect(filename):
    #filename= "1"

    y, sr = librosa.load(r"C:/Anaconda codes/speaker diarization/hack/dataset/train/" + filename+".wav")
    mfccs = librosa.feature.mfcc(y=y, sr=sr,n_mfcc=20,n_fft=256,hop_length=511)

    mfccs1=mfccs.transpose()
    data=mfccs1

    print("\nfile name = "+filename+".wav")
    #-------------------------------------------------------------------------------
    #BIC FUNCTION
    def compute_bic(kmeans,X):
       
        # assign centers and labels
        centers = [kmeans.cluster_centers_]
        labels  = kmeans.labels_
        #print(labels)
        #number of clusters
        m = kmeans.n_clusters
        # size of the clusters
        n = np.bincount(labels)
        #size of data set
        N, d = X.shape

        #compute variance for all clusters beforehand
        cl_var = (1.0 / (N - m) / d) * sum([sum(distance.cdist(X[np.where(labels == i)], [centers[0][i]], 
                 'euclidean')**2) for i in range(m)])
        #print(cl_var)

        const_term = 0.5 * m * np.log(N) * (d+1)
        #print(const_term)
        
        BIC = np.sum([n[i] * np.log(n[i]) -
                   n[i] * np.log(N) -
                 ((n[i] * d) / 2) * np.log(2*np.pi*cl_var) -
                 ((n[i] - 1) * d/ 2) for i in range(m)]) - const_term

        return(BIC)

    #----------------------------------------------------------------------------------------------------------------
    def calcyBIC(data,kmeans):
        
        a=0
        t=a+20
        b=a+40
        X_axis=[]
        Y_axis=[]
        Z_axis=[]

        threshold= 200 #flexible as per penalty added and our knowledge too
        count=0
        for i in range(200,len(data)): 


            X1= data[a:t]
            X2= data[t:b]
            X=  data[a:b]

            #print(X1.shape)
            if(X2.shape[0]<20):
                break        
            #compute_bic(X1) compute BIC(X2)
            bic1= BIC = compute_bic(Kmeans,X1) 
            bic2= BIC = compute_bic(kmeans,X2)
            bic= BIC =  compute_bic(kmeans,X) 
            diff= abs((bic1[0]+bic2[0])-bic[0])

            print(bic1,bic2,bic,diff)
            #print bic1[0]
            #print bic2[0]

            if diff>threshold:
                        #print("speaker change detected at ",(t)," frame")
                        X_axis.append(t)
                        Y_axis.append(threshold)
                        Z_axis.append(diff)

                        count=count+1
                        a=b
                        t=a+20
                        b=a+40
                        #print a,t,b
            else:
                        t=t+1
                        a=t-20
                        b=t+20
            
            return(X_axis)

    #-------------------------------------------------------------------------------

    #FINDING TIME STAMPS

    lll=[]
    for i in range(0,mfccs1.shape[0]):
        if(sum(list(mfccs1[i][1:]))==0):
            lll.append((librosa.core.frames_to_time(i, sr=sr, hop_length=512, n_fft=None)))
    #print(lll)
        
    import statistics 
    a=[]
    llll=[]
    for i in range(0,len(lll)-1):
        if (abs(lll[i+1]-lll[i]<1)):
            a.append(lll[i])
        else:
            if a!=[]:
                if statistics.mean(a)>1:
                    llll.append(statistics.mean(a))
                a=[]
    #print("timestamps \n",llll)

    #---------------------------------------------------------------------------------
    #CONVERTING TO FRAMES
    b=[]
    for i in range(0,len(llll)):
        v=librosa.core.time_to_frames(llll[i], sr=sr, hop_length=512, n_fft=None)
        print("voice changed at frame ",v)
        b.append(v)
    print("total voice changes ",len(b))

    print("\n")

    #LOG FILE GEN
    with open('C:/Anaconda codes/speaker diarization/hack/LOG files/'+filename+'.txt', 'w') as f:
        for item in llll:
            f.write("%s\n" % item)

    print("\n*LOG FILE MADE*\n")
    
    data1 = pd.read_csv('C:/Anaconda codes/speaker diarization/hack/LOG files/'+filename+'.txt', sep=" ", header=None)
    data1.columns = ["OUR TS"]
    Q1=data1["OUR TS"].values.tolist()

    data = pd.read_csv('C:/Anaconda codes/speaker diarization/hack/dataset/train_script/'+filename+'.txt', sep=" ", header=None)
    data.columns = ["given TS"]
    Q2=data['given TS'].values.tolist()

    print("TIMESTAMP DATA")
    print("----------\n",data)
    print("----------\n",data1)

    #-----------------------------------------------------------------------
    #METRICS
    h=[]
    j=0
    for i in Q2:
        for j in Q1:
            if(math.floor(i)==math.floor(j)):
                h.append(j)
            #print(i,j)

    #print(h)

    ss=[]
    for i in range(0,len(h)):
        ss.append((1-abs((h[i]-Q2[i])/Q2[i] ))*100)
        
    #print(h)
    #print(Q2)
    print("\nPERCENTAGE ACCURACY\n")
    print(ss)

#---------------------------------------------------------------------------------

filename= "20"
speaker_change_detect(filename)




