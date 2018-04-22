
# coding: utf-8

# In[1]:

import matplotlib.pyplot as plt
import numpy as np
import scipy.io as sp
from scipy.fftpack import fft
from scipy.io import wavfile # get the api
import sunau;
import librosa
import librosa.display
import pandas
import os
np.set_printoptions(suppress=True)


# In[18]:

def writeMatrixToFile(self,filename,matrix):
        np.save(filename, matrix);
        os.rename(filename+".npy", filename+".txt");
        print("Written successfully..");
        pass;
    
def loadMatrixFromFile(self,filename):        
    matrix=None;
    if(os.path.isfile(filename)):
        matrix=np.load(filename);       
    return matrix;

def readCSVFile(file):
    data=pandas.read_csv(file,",",header=0, na_values='?', skipinitialspace=True);
    return data;
    pass;


# In[62]:

class PreProcessing:
    filename=None;
    y=None;
    sr=None;
    log_enabled=True;
    centroid=None;
    spectro=None;
    spectro_phase=None;
    max_sample_vector_size=660000; 
    duration=30;
    def __init__(self,filename,duration=30):
        #self.log(filename);
        self.filename=filename;
        self.reloadAudioFile(duration);
        self.duration=duration;
        pass;
    
    def reloadAudioFile(self,duration=30):
        self.y, self.sr = librosa.load(self.filename,duration=duration);
        self.y=self.y[:self.max_sample_vector_size];
        pass;
    
    #Short-Term-Fourier trasform
    def getSTFT(self):
        self.stft=librosa.stft(y=self.y);
        return self.stft;
        pass;
    
    #spectro graph
    def getSpectrogram(self):
        stft=self.getSTFT();
        self.spectro, self.spectro_phase = librosa.magphase(stft);        
        return self.spectro, self.spectro_phase;
        pass;
    
    def getCentroid(self):
        self.centroid=librosa.feature.spectral_centroid(y=self.y,sr=self.sr);
        return self.centroid;    

    def getSpectralRolloff(self):
        self.rolloff=librosa.feature.spectral_rolloff(y=self.y, sr=self.sr);
        return self.rolloff;
    
    def getZeroCrossing(self):
        self.zero_crossing_rate=librosa.feature.zero_crossing_rate(self.y);
        return self.zero_crossing_rate;
    
    def getSpectralContrast(self):
        #Jiang, Dan-Ning, Lie Lu, Hong-Jiang Zhang, Jian-Hua Tao, and Lian-Hong Cai. “Music type classification by spectral contrast feature.” In Multimedia and Expo, 2002. ICME‘02. Proceedings. 2002 IEEE International Conference on, vol. 1, pp. 113-116. IEEE, 2002.
        S = np.abs(self.getSTFT());
        self.contrast = librosa.feature.spectral_contrast(S=S, sr=self.sr);
        return self.contrast;
    
    def getMFCC(self):
        self.mfcc = librosa.feature.mfcc(y=self.y, sr=self.sr, hop_length=512, n_mfcc=13);
        return self.mfcc;
    
    def getMelSpec(self):
        self.mel=librosa.feature.melspectrogram(y=pp.y, sr=pp.sr,n_mels=10);
        return self.mel;
    
    def getRMS(self):
        self.rms=librosa.feature.rmse(y=self.y);
        return self.rms;

    def drawRMS(self):
        rms=self.getRMS();
        S,phase=self.getSpectrogram();
        plt.figure()
        plt.subplot(2, 1, 1)
        plt.semilogy(rms.T, label='RMS Energy')
        plt.xticks([])
        plt.xlim([0, rms.shape[-1]])
        plt.legend(loc='best')
        plt.subplot(2, 1, 2)
        librosa.display.specshow(librosa.amplitude_to_db(S, ref=np.max),y_axis='log', x_axis='time')
        plt.title('log Power spectrogram')
        plt.tight_layout()
        plt.show();
        pass;
    
    def drawSpectrogramWithCentroid(self):
        centroid=self.getCentroid();
        S,phase=self.getSpectrogram();
        plt.figure()
        plt.subplot(2, 1, 1)
        plt.semilogy(centroid.T, label='Spectral centroid')
        plt.ylabel('Hz')
        plt.xticks([])
        plt.xlim([0, centroid.shape[-1]])
        plt.legend()
        plt.subplot(2, 1, 2)
        librosa.display.specshow(librosa.amplitude_to_db(S, ref=np.max),y_axis='log', x_axis='time')
        plt.title('log Power spectrogram')
        plt.tight_layout();
        plt.show();
        pass;
    
    def drawSpectralRolloff(self):
        rolloff=self.getSpectralRolloff();
        S,phase=self.getSpectrogram();
        plt.figure()
        plt.subplot(2, 1, 1)
        plt.semilogy(rolloff.T, label='Roll-off frequency')
        plt.ylabel('Hz')
        plt.xticks([])
        plt.xlim([0, rolloff.shape[-1]])
        plt.legend()
        plt.subplot(2, 1, 2)
        librosa.display.specshow(librosa.amplitude_to_db(S, ref=np.max),y_axis='log', x_axis='time')
        plt.title('log Power spectrogram')
        plt.tight_layout();
        plt.show();
        pass;
    
    def drawSpectralContrast(self):
        contrast=self.getSpectralContrast();
        S,phase=self.getSpectrogram();
        S = np.abs(self.getSTFT());
        plt.figure()
        plt.subplot(2, 1, 1)
        librosa.display.specshow(librosa.amplitude_to_db(S,ref=np.max),y_axis='log')
        plt.colorbar(format='%+2.0f dB')
        plt.title('Power spectrogram')
        plt.subplot(2, 1, 2)
        librosa.display.specshow(contrast, x_axis='time')
        plt.colorbar()
        plt.ylabel('Frequency bands')
        plt.title('Spectral contrast')
        plt.tight_layout();
        plt.show();
        pass;
    
    def drawMFCC(self):
        mfccs=self.getMFCC();
        plt.figure(figsize=(6, 4))
        librosa.display.specshow(mfccs, x_axis='time')
        #plt.colorbar()
        plt.title('MFCC')
        plt.tight_layout()
        plt.show();
        pass;
    
    
    def log(self,a,b=None):
        if(self.log_enabled):
            if(b!=None):
                print(a,b);
            else:
                print(a);
        pass;  


# In[3]:




# In[65]:

class ProcessDataset:
    columns = ["id","type","y_index","y","centroid mean","centroid var","rolloff mean","rolloff var","zero mean",
               "zero var","rms mean","rms var","contrast mean","contrast var","mfcc1 mean","mfcc1 var",
               "mfcc2 mean","mfcc2 var","mfcc3 mean","mfcc3 var","mfcc4 mean","mfcc4 var","mfcc5 mean","mfcc5 var",
              "mel1 mean","mel1 var","mel2 mean","mel2 var","mel3 mean","mel3 var","mel4 mean","mel4 var","mel5 mean","mel5 var"]               
    #columns = ["id","type","y_index","y","centroid mean","centroid var","rolloff mean","rolloff var","zero mean",
    #           "zero var","rms mean","rms var","contrast mean","contrast var","mfcc1 mean","mfcc1 var",
    #           "mfcc2 mean","mfcc2 var","mfcc3 mean","mfcc3 var","mfcc4 mean","mfcc4 var","mfcc5 mean","mfcc5 var",
    #           "mfcc6 mean","mfcc6 var","mfcc7 mean","mfcc7 var","mfcc8 mean","mfcc8 var","mfcc9 mean","mfcc9 var",
    #            "mfcc10 mean","mfcc10 var","mfcc11 mean","mfcc11 var","mfcc12 mean","mfcc12 var","mfcc13 mean","mfcc13 var"];
    genre_out={"blues":[1,0,0,0,0,0,0,0,0,0],"classical":[0,1,0,0,0,0,0,0,0,0],"country":[0,0,1,0,0,0,0,0,0,0],"disco":[0,0,0,1,0,0,0,0,0,0],"hiphop":[0,0,0,0,1,0,0,0,0,0],"jazz":[0,0,0,0,0,1,0,0,0,0],"metal":[0,0,0,0,0,0,1,0,0,0],"pop":[0,0,0,0,0,0,0,1,0,0],"reggae":[0,0,0,0,0,0,0,0,1,0],"rock":[0,0,0,0,0,0,0,0,0,1]};    
    genre_out_index={"blues":0,"classical":1,"country":2,"disco":3,"hiphop":4,"jazz":5,"metal":6,"pop":7,"reggae":8,"rock":9};    
    dataframe=None;
    mfcc_features=5;
    mel_features=5;
    dir="../genres";
    genre_dir={"blues":"blues","classical":"classical","country":"country","disco":"disco","hiphop":"hiphop","jazz":"jazz","metal":"metal","pop":"pop","reggae":"reggae","rock":"rock"};
    def __init__(self):
        self.dataframe = pandas.DataFrame(columns=self.columns);
        pass;

    def extractTimberalFeatures(self,genre,audio_number,filename):
        features=[];
        pp=PreProcessing(filename);            
        centroid=pp.getCentroid()[0];
        rolloff=pp.getSpectralRolloff()[0];        
        zero=pp.getZeroCrossing()[0];        
        contrast=pp.getSpectralContrast()[0];        
        rms=pp.getRMS()[0];
        mfcc=pp.getMFCC();
        mel=pp.getMelSpec();
        
        features.append(audio_number);
        features.append(genre);        
        features.append(self.genre_out_index[genre]);        
        features.append(self.genre_out[genre]);
        features.append(centroid.mean());
        features.append(centroid.var()); 
        features.append(rolloff.mean());
        features.append(rolloff.var()); 
        features.append(zero.mean());
        features.append(zero.var()); 
        features.append(rms.mean());
        features.append(rms.var()); 
        features.append(contrast.mean());
        features.append(contrast.var()); 
        for i in range(self.mfcc_features):
            features.append(mfcc[i].mean());
            features.append(mfcc[i].var());  
        for i in range(self.mel_features):
            features.append(mel[i].mean());
            features.append(mel[i].var());  
        self.dataframe.loc[self.dataframe.size]=features;
    
    def saveDataFrame(self):
        filename="audiofeatures_numpy_matrix";
        arr=self.dataframe.as_matrix(columns=None);
        np.save(filename,arr);       
        self.dataframe.to_csv("audiofeatures.csv",sep=",")
        print("Written successfully..");
        pass;
    
    def extractFeatures(self):
        print("Extracting Features...");
        percent_completed=0;        
        for k in self.genre_dir:    
            genre=k;
            print("-------------------["+genre+"]-----------------------")
            for i in range(100):
                audio_number="%0.5d"%i;
                filename=self.dir+"/"+self.genre_dir[genre]+"/"+self.genre_dir[genre]+"."+audio_number+".au";              
                self.extractTimberalFeatures(genre,audio_number,filename);  
                percent_completed=i;
                if(percent_completed%10==0):
                    print("Percent completed:",percent_completed);            
        self.saveDataFrame();
        print("Extraction done");
        pass;
        


# In[66]:

p1=ProcessDataset();
p1.extractFeatures();
p1.dataframe


# In[39]:

dir="../genres";
genre_dir={"blues":"blues","classical":"classical","country":"country","disco":"disco","hiphop":"hiphop","jazz":"jazz","metal":"metal","pop":"pop","reggae":"reggae","rock":"rock"};


# In[48]:

genre="disco"

audio_number="%0.5d"%0;
filename=dir+"/"+genre_dir[genre]+"/"+genre_dir[genre]+"."+audio_number+".au";  

pp=PreProcessing(filename); 
s,p=pp.getSpectrogram();
len(s)


# In[56]:

genre="disco"
audio_number="%0.5d"%0;
filename=dir+"/"+genre_dir[genre]+"/"+genre_dir[genre]+"."+audio_number+".au";  

pp=PreProcessing(filename); 
a=librosa.feature.melspectrogram(y=pp.y, sr=pp.sr,n_mels=10)
len(a[0])


# In[40]:

genre="disco"
pds=ProcessDataset();

audio_number="%0.5d"%98;
filename=dir+"/"+genre_dir[genre]+"/"+genre_dir[genre]+"."+audio_number+".au";  
pds.extractTimberalFeatures(genre,audio_number,filename);

audio_number="%0.5d"%99;
filename=dir+"/"+genre_dir[genre]+"/"+genre_dir[genre]+"."+audio_number+".au";  
pds.extractTimberalFeatures(genre,audio_number,filename);

audio_number="%0.5d"%0;
filename=dir+"/"+genre_dir[genre]+"/"+genre_dir[genre]+"."+audio_number+".au";  
pds.extractTimberalFeatures(genre,audio_number,filename);

#pds.extractFeatures();
pds.dataframe


# In[41]:

#RMS
for k in genre_dir:    
    genre=k;
    print("-------------------["+genre+"]-----------------------")
    ta=[];
    for i in range(1):
        audio_number="%0.5d"%i;
        filename=dir+"/"+genre_dir[genre]+"/"+genre_dir[genre]+"."+audio_number+".au"; 
        pp=PreProcessing(filename);        
        pp.drawRMS();
        print(pp.getRMS()); 
        ta.append(len(pp.y));
        print(i,":",len(pp.getRMS()[0])," ylen:",len(pp.y)," sr:",pp.sr); 
    #break;


# In[24]:

#MFCC
for k in genre_dir:    
    genre=k;
    print("-------------------["+genre+"]-----------------------")
    ta=[];
    for i in range(3):
        audio_number="%0.5d"%i;
        filename=dir+"/"+genre_dir[genre]+"/"+genre_dir[genre]+"."+audio_number+".au"; 
        pp=PreProcessing(filename);        
        pp.drawMFCC();
        print(pp.getMFCC()); 
        ta.append(len(pp.y));
        print(i,":",len(pp.getMFCC()[0])," ylen:",len(pp.y)," sr:",pp.sr); 
    break;


# In[15]:

#Zero crossing
for k in genre_dir:    
    genre=k;
    print("-------------------["+genre+"]-----------------------")
    ta=[];
    for i in range(10):
        audio_number="%0.5d"%i;
        filename=dir+"/"+genre_dir[genre]+"/"+genre_dir[genre]+"."+audio_number+".au"; 
        pp=PreProcessing(filename,10);
        print(pp.getZeroCrossing()); 
        ta.append(len(pp.y));
        print(i,":",len(pp.getZeroCrossing()[0])," ylen:",len(pp.y)," sr:",pp.sr); 
    break;


# In[155]:

#Spectral Contrast
for k in genre_dir:    
    genre=k;
    print("-------------------["+genre+"]-----------------------")
    ta=[];
    for i in range(3):
        audio_number="%0.5d"%i;
        filename=dir+"/"+genre_dir[genre]+"/"+genre_dir[genre]+"."+audio_number+".au"; 
        pp=PreProcessing(filename);
        pp.drawSpectralContrast();
        print(pp.getSpectralContrast());
        ta.append(len(pp.y));
        print(i,":",len(pp.getSpectralContrast()[0])," ylen:",len(pp.y)," sr:",pp.sr); 
    break;
    


# In[145]:

#SpectralRolloff
for k in genre_dir:    
    genre=k;
    print("-------------------["+genre+"]-----------------------")
    ta=[];
    for i in range(10):
        audio_number="%0.5d"%i;
        filename=dir+"/"+genre_dir[genre]+"/"+genre_dir[genre]+"."+audio_number+".au"; 
        pp=PreProcessing(filename);
        pp.drawSpectralRolloff();
        print(pp.getSpectralRolloff()); 
        ta.append(len(pp.y));
        print(i,":",len(pp.getSpectralRolloff()[0])," ylen:",len(pp.y)," sr:",pp.sr);    
    print(genre," ymin:",np.min(ta),' max:',np.max(ta));
    break;


# In[131]:




# In[ ]:




# In[ ]:




# In[67]:

def partitionDataFrame(df,ratio):
    df = df.sample(frac=1).reset_index(drop=True)#shuffling rows
    df = df.sample(frac=1).reset_index(drop=True)#again shuffling
    size=df["id"].count();
    limit=int(ratio*size);
    train_ds=df.loc[0:limit];    
    test_ds=df.loc[limit:size];
    train_ds.to_csv("train.csv",sep=",");
    test_ds.to_csv("test.csv",sep=",");
    print("Partitioning done");


# In[68]:

df=readCSVFile("audiofeatures.csv");
partitionDataFrame(df,0.8);


# In[ ]:



