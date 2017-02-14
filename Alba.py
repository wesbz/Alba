import matplotlib
matplotlib.use('Agg')
from scipy.io.wavfile import read
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import numpy as np
import os

if (os.getcwd() != '___WHERE YOU ARE SUPPOSED TO WORK WITH ALBA___'):
    os.chdir("___WHERE YOU WANT TO GO___")


notes1 = np.array([[32.70,65.41,130.81,261.63,523.25,1046.50,2093.00,4186.01],
[34.65,69.30,138.59,277.18,554.37,1108.73,2217.46,4434.92],
[36.71,73.42,146.83,293.66,587.33,1174.66,2349.32,4698.64],
[38.89,77.78,155.56,311.13,622.25,1244.51,2489.02,4978.03],
[41.20,82.41,164.81,329.63,659.26,1318.51,2637.02,5274.04],
[43.65,87.31,174.61,349.23,698.46,1396.91,2793.83,5587.65],
[46.25,92.50,185.00,369.99,739.99,1479.98,2959.96,5919.91],
[49.00,98.00,196.00,392.00,783.99,1567.98,3135.96,6271.93],
[51.91,103.83,207.65,415.30,830.61,1661.22,3322.44,6644.88],
[55.00,110.00,220.00,440.00,880.00,1760.00,3520.00,7040.00],
[58.27,116.54,233.08,466.16,932.33,1864.66,3729.31,7458.62],
[61.74,123.47,246.94,493.88,987.77,1975.53,3951.07,7902.13]])
#[note,octave]: 0:do 1:do# 2:ré etc

notes2 = np.zeros((8000,3), dtype=int)

for i in range(notes1.shape[0]):
    for k in range(notes1.shape[1]):
        notes2[round(notes1[i,k]),:]=np.array([i+1,k+1,notes1[i,k]])

fichier = #THE FILE YOU WANT TO ANALYZE

def spectre():
    Gson = read(fichier)[1]
    NFFT = 16384 #34000
    Fs = 44100
    global Pxx
    global freqs
    global t                                                        #32768
    Pxx, freqs, t, im = plt.specgram(Gson, NFFT=NFFT, Fs=Fs, noverlap=14000, cmap=plt.cm.gist_heat)
    return

#plt.show()


def LAnote(): #Recherche du pic max, retourne la note et octave
    L = np.zeros((Pxx.shape[1],3))
    #lissage(Pxx)
    for k in range(Pxx.shape[1]):
        d = 0
        while (notes2[  round(    freqs[  np.argmax(    Pxx[:,k]    )   ]  )-d,0] == 0) and (notes2[    round(    freqs[    np.argmax(  Pxx[:,k]    )   ]   )+d,0] == 0):
            d+=1
        if notes2[round(freqs[np.argmax(Pxx[:,k])])-d,0] != 0:
            L[k,:2] = notes2[round(freqs[np.argmax(Pxx[:,k])])-d,:2] 
        elif notes2[round(freqs[np.argmax(Pxx[:,k])])+d,0] != 0:
            L[k,:2] = notes2[round(freqs[np.argmax(Pxx[:,k])])+d,:2]
        L[k,2] = max(Pxx[:,k])
    return L


def lissage(Pxx): #Lissage pour éliminer le bruit
    for k in range(Pxx.shape[0]):
        for i in range(Pxx.shape[1]):
            if Pxx[k,i] < 20000:
                Pxx[k,i] = 0
    return Pxx


def peak(): #Recherche des pics principaux, retourne les [fréquences, amplitude]
    L = np.zeros((Pxx.shape[1],8,2))
    lissage(Pxx)
    for k in range(Pxx.shape[1]):
        M=max(Pxx[:,k])
        for i in range(1,Pxx.shape[0]-1):
            if (Pxx[i,k]-Pxx[i-1,k] > 0) and (Pxx[i+1,k]-Pxx[i,k] < 0) and (Pxx[i,k]>=M*0.03):
                if Pxx[i,k] > L[k,:,1].min():
                    L[k,np.argmin(L[k,:,1]),:] = [freqs[i],Pxx[i,k]]
    return L


def ChNote(L): #Transcription fréquence --> (note, octave)
    M = np.zeros((L.shape[0],L.shape[1],3))
    for k in range(L.shape[0]):
        for i in range(L.shape[1]):
            if L[k,i,0] != 0:
                d = 0
                while (notes2[round(L[k,i,0])+d,0] == 0) and (notes2[round(L[k,i,0])-d,0] == 0) :
                    d+=1
                if (notes2[round(L[k,i,0])+d,:2] != [0,0]).all():
                    M[k,i,:] = [notes2[round(L[k,i,0])+d,0],notes2[round(L[k,i,0])+d,1],L[k,i,1]]
                    L[k,i,0] = notes2[round(L[k,i,0])+d,2]
                elif (notes2[round(L[k,i,0])-d,:2] != [0,0]).all():
                    L[k,i,0] = notes2[round(L[k,i,0])-d,2]
                    M[k,i,:] = [notes2[round(L[k,i,0])-d,0],notes2[round(L[k,i,0])-d,1],L[k,i,1]]
    return M
    
    """Pour sauvegarder
    np.savetxt('WAV/Piano/gamme chromatique bruit notes.txt',LAnote(Pxx,freqs),fmt='%.2i',delimiter=',', newline='\r\n')
    """
def lily(L):
    nom = input("nom du fichier?")
    f=open("Partitions/"+str(nom)+".ly", 'a')
    f.write('\version "2.16.0"\n\r \header{ \n\r title = "'+str(nom)+'" \n\r subtitle = "" \n\r } \n\r { \n\r')
    ton=['r','c','cis','d','dis','e','f','fis','g','gis','a','ais','b']
    octave=['',',,',',','',"'","''","'''","''''","'''''","''''''"]
    for note in L:
        f.write(str(ton[int(note[0])])+str(octave[int(note[1])])+" ")
    f.write('\n\r }')
    f.close()
    return("Partition crée !")

def lily2(L): #Faut lui donner du ChNote()
    nom = input("nom du fichier? ")
    f=open("Partitions/"+str(nom)+".ly", 'a')
    f.write('\version "2.16.0"\n\r \header{ \n\r title = "'+str(nom)+'" \n\r subtitle = "" \n\r } \n\r { \n\r r')
    ton=['r','c','cis','d','dis','e','f','fis','g','gis','a','ais','b']
    octave=['',',,',',','',"'","''","'''","''''","'''''","''''''"]
    for moment in L:
        f.write('<<')
        if (moment != np.zeros(moment.shape)).any():
            for note in moment:
                if (note != np.zeros(note.shape)).any():
                    f.write(str(ton[int(note[0])])+str(octave[int(note[1])])+" ")
        else:
            f.write('r')
        f.write('>>\n\r')
    f.write('\n\r }')
    f.close()
    return("Partition crée !")

def recondMult(L):
    """reconditionnement de la liste des notes
     _____________________________________________________
    | (1;0) |                                             |
    |_______|_____________________________________________|
    | (1;1) |                                             |
    |_______|_____________________________________________|
    ...
     _____________________________________________________
    | (12;7)|                                             |
    |_______|_____________________________________________|
    """
    T = np.zeros((12,8,len(t)))
    for k in range(len(t)):
        for note in L[k]:
            if (note != np.array([0,0,0])).all():
                T[note[0]-1,note[1]-1,k] = note[2]
    return T

def recond(L):
    """reconditionnement de la liste de la note de max amplitude"""
    T = np.zeros((12,8,len(t)))
    for k in range(len(t)):
        if (L[k,:] != np.array([0,0,0])).all():
            T[L[k,0]-1,L[k,1]-1,k] = L[k,2]
    return T

"""def baseRythme(T):
    """

def recordSpec():
    """fonction pour enregistrer les spectres tracés par spectre() dans /TIPE/Courbes/<nom>"""
    nom = input("nom du dossier? ")
    try:
        os.chdir('Courbes/'+str(nom))
    except FileNotFoundError:
        os.mkdir('Courbes/'+str(nom))
        os.chdir('Courbes/'+str(nom))
    for k in range(Pxx.shape[1]):
        plt.clf()
        plt.ylim(ymax=Pxx.max())
        plt.plot(freqs,Pxx[:,k])
        plt.savefig('t='+str(k))
    os.chdir('../..')


def fondamentaliseur(L):
    M = np.copy(L)
    for k in M:
        for f in range(len(k[:,0])-1):
            fonda = k[:,0][f]
            if fonda != 0:
                for h in range(len(k[:,0])):
                    harmo = k[:,0][h]
                    if ((harmo/fonda) > 1) and (abs(round(harmo/fonda)-(harmo/fonda))%1 < 0.1):
                        k[:][h] = 0,0
    return(M)

def Optimus():
    spectre()
    X = peak()
    Y = ChNote(X)
    Y = ChNote(X)
    Z = recondMult(Y)
    A = fondamentaliseur(X)
    B = ChNote(A)
    C = recondMult(B)
    D = correcteur_rythme(C)
    E = placeur(D)
    F = timer(E)
    lily_3(E)
    return(X,Y,Z,A,B,C,D,E,F)


def correcteur_rythme(L):
    tmin = 0.15
    E = np.copy(L)
    pics = []
    for note in E:
        for octave in note:
            pics = []
            for i in range(1,len(octave)-1):
                if ( (octave[i-1]-octave[i]) < 0) and ( (octave[i]-octave[i+1]) > 0):
                    pics.append(i)
            for p in pics:
                d, g = 0, 0
                while (p+d < len(octave)) and (octave[p+d] - octave[p+d+1] > 0):
                    d += 1
                while (p-g > 1) and (octave[p-g-1] - octave[p-g] < 0):
                    g += 1
                if (abs(t[p+d] - t[p-g]) < tmin):
                    for j in range(p-g,p+d):
                        octave[j] = octave[p-g-1]
    return(E)


def placeur(L):
    M = np.zeros(L.shape)
    for k in range(L.shape[0]):
        for j in range(L.shape[1]):
            for i in range(1,len(L[k,j])-1):
                if (L[k,j,i-1] - L[k,j,i] >= 0) and (L[k,j,i] - L[k,j,i+1] < 0) or (L[k,j,i-1] - L[k,j,i] > 0) and (L[k,j,i] == 0):
                    M[k,j,i] = int(L[k,j,i] - L[k,j,i+1] < 0)
                    M[k,j,i-1] = (-1)*int(L[k,j,i-1] - L[k,j,i] > 0)
    """for i in range(L.shape[2]):
        if (M[:,:,i] == np.zeros(M[:,:,i].shape)).all():
            M[:,:,i] = 2*np.ones(M[:,:,i].shape)"""
    return(M)

def timer(M):
    L = [[[] for k in range(8)] for i in range(12)]
    for k in range(12):
        for i in range(8):
            for j in range(M.shape[2]):
                if (M[k,i,j] == 1):
                    l = j
                    while (M[k,i,l] != -1):
                        l += 1
                    L[k][i].append([t[l]-t[j],j])
                    M[k,i,j+1:l]= (-0.5) * np.ones((l-j-1))
    unit = np.mean(np.sum(L), axis = 0)[0]
    for k in range(12):
        for i in range(8):
            for note in L[k][i]:
                M[k,i,note[1]] = 4/(round(2*note[0]/unit)/2)
    return L



def lily_recond(L):
    nom = input("nom du fichier? ")
    f=open("Partitions/"+str(nom)+".ly", 'a')
    f.write('\version "2.16.0"\n\r \header{ \n\r title = "'+str(nom)+'" \n\r subtitle = "" \n\r } \n\r { \n\r r')
    ton=['r','c','cis','d','dis','e','f','fis','g','gis','a','ais','b']
    octave=['',',,',',','',"'","''","'''","''''","'''''","''''''"]
    for i in range(L.shape[2]):
        f.write('<<')
        for j in range(len(L)):
            for k in range(len(L[0])):
                if L[j,k,i] == 1:
                    f.write(str(ton[j+1])+str(octave[k+1])+" ")
        f.write('>>\n\r')
    f.write('\n\r }')
    f.close()
    return("Partition crée !")

def lily_3(L):
    nom = input("nom du fichier? ")
    f=open("Partitions/"+str(nom)+".ly", 'a')
    f.write('\version "2.16.0"\n\r'+'\header{ \n\r'+'title = "'+str(nom)+'" \n\r'+'subtitle = "" \n\r'+'}\n\r'+'{ \n\r'+'r')
    ton=['r','c','cis','d','dis','e','f','fis','g','gis','a','ais','b']
    octave=['',',,',',','',"'","''","'''","''''","'''''","''''''"]
    for i in range(L.shape[2]):
        f.write('<<')
        if (L[:,:,i] != np.zeros(L[:,:,i].shape)).any():
            for j in range(len(L)):
                for k in range(len(L[0])):
                    if L[j,k,i] > 0:
                        if L[j,k,i] == 4/1.5:
                            f.write(str(ton[j+1])+str(octave[k+1])+"4."+" ")
                        else:
                            f.write(str(ton[j+1])+str(octave[k+1])+str(int(L[j,k,i]))+" ")
        """else :
            f.write('r4')"""
        f.write('>>\n\r')
    f.write('\n\r }')
    f.close()
    return("Partition crée !")
