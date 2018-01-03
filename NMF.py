import matplotlib
import numpy as np
import matplotlib.pyplot as plt
import random as rd
import copy as copy
from scipy.io.wavfile import read
from mpl_toolkits.mplot3d import Axes3D
import os

def iterationEuc(W,H,V):
    """Fonction qui prend des array en entrée et calcule l'itération suivante dans l'algorithme multiplicatif de NMF selon la distance euclidienne"""
    (n,r),m = W.shape, H.shape[1]
    H , W = H*(np.dot(W.transpose(),V)/np.dot(W.transpose(),np.dot(W,H))),W*(np.dot(V,H.transpose())/np.dot(np.dot(W,H),H.transpose()))
    return(W,H)

def iterationKL(W,H,V):
    """Fonction qui prend des array en entrée et calcule l'itération suivante dans l'algorithme multiplicatif de NMF selon la divergence généralisée de Kullback-Leibler"""
    H , W = H*(np.dot(W.transpose(),V/np.dot(W,H)))/np.dot(W.transpose(),np.ones(V.shape)), W*(np.dot(V/np.dot(W,H),H.transpose()))/np.dot(np.ones(V.shape),H.transpose())
    return(W,H)

def norme(x):
    """Prend en entrée x un array de shape (n,) où n est un entier naturel."""
    return (np.dot(x,np.transpose(x))**0.5)

def init_K_moy_spher(V,K):
    repr_index = []
    epsilon = 0.1
    while (len(repr_index) < K): #On initialise aléatoirement la liste des centroïdes
        a = rd.randint(0,V.shape[1]-1)
        if (a not in repr_index):
            repr_index += [a]
    C_old = [] #les Clusters
    C_new = []
    R_old = [] #les centroïdes
    R_new = []
    for i in range(K):
        C_old += [[]]
        C_new += [[]]
        R_old += [V[:,repr_index[i]]/norme(V[:,repr_index[i]])]
    for i in range(V.shape[1]):
        r = 0
        d = (-1) * np.inf
        for j in range(K):
            if (np.dot(R_old[j]/norme(R_old[j]),V[:,i]/norme(V[:,i])) > d):
                r = j
                d = np.dot(R_old[j]/norme(R_old[j]),V[:,i]/norme(V[:,i]))
        C_new[r] += [V[:,i]]
    for i in range(K):
        nouveau_representant = np.zeros((V.shape[0],))
        for objet in C_new[i]:
            nouveau_representant += objet
        nouveau_representant /= norme(nouveau_representant)
        R_new += [nouveau_representant]
    while (np.array([norme(np.array(R_new) - np.array(R_old))[i,i] for i in range(len(R_new))]) > epsilon).any() :
        C_old , R_old = copy.deepcopy(C_new), copy.deepcopy(R_new)
        C_new = []
        R_new = []
        for i in range(K):
            C_new += [[]]
        for i in range(V.shape[1]):
            r = 0
            d = -1*np.inf
            for j in range(K):
                if (np.dot(R_old[j],V[:,i]) > d):
                    r = j
                    d = np.dot(R_old[j],V[:,i])
            C_new[r] += [V[:,i]]
        for i in range(K):
            nouveau_representant = np.zeros((V.shape[0],))
            for objet in C_new[i]:
                nouveau_representant += objet
            nouveau_representant /= norme(nouveau_representant)
            R_new += [nouveau_representant]
    return (np.transpose(np.array(R_new)),C_old,C_new,R_old,R_new)




def distEuc(A,B):
    """Prend en entrée A et B deux array et renvoie la distance euclidienne entre les 2 matrices"""
    return (sum(sum((A-B)**2)))

# X,Y = np.array([[1,0],[2,0],[3,1],[6,3]]), np.array([[1,2,0,0],[0,0,1,2]])
# V = np.dot(X,Y)
# #d1, d2 = np.inf, np.inf
# for i in range(1000):
#     W , H = init_K_moy_spher(V,2)[0] , np.random.rand(2,4)
#     Wp, Hp = np.random.rand(4,2), np.random.rand(2,4)
#     for k in range(1000):
#         W,H = iterationEuc(W,H,V)
#         Wp,Hp = iterationEuc(Wp,Hp,V)
#     if distEuc(V,np.dot(W,H)) < d1:
#         d1 = distEuc(V,np.dot(W,H))
#         print("d1=",d1)
#         Wsave , Hsave = copy.deepcopy(W), copy.deepcopy(H)
#     if distEuc(V,np.dot(Wp,Hp)) < d2:
#         print("d2=",d2)
#         d2 = distEuc(V,np.dot(Wp,Hp))
#         Wpsave , Hpsave = copy.deepcopy(Wp), copy.deepcopy(Hp)
# 
# print(V,'\n',np.dot(Wsave,Hsave),'\n',np.dot(Wpsave,Hpsave))





if (os.getcwd() != 'C:\\Users\\Wassim\\Google Drive\\Projet'):
    os.chdir("C:/Users/Wassim/Google Drive/Projet/")

fichier = "Piano scale/3 accords do fa si.wav"

def spectre():
    Gson = read(fichier)[1]
    NFFT = 8192
    Fs = 22050
    Pxx, freqs, t, im = plt.specgram(Gson, NFFT=NFFT, Fs=Fs, noverlap=NFFT/2, cmap=plt.cm.gist_heat)
    return Pxx,freqs,t,im

Pxx,freqs,t,im = spectre()

    
R = 3
Oxx = Pxx[:800,:]
W,H = np.random.rand(Oxx.shape[0],R), np.random.rand(R,Oxx.shape[1])
A,B = np.random.rand(Oxx.shape[0],R), np.random.rand(R,Oxx.shape[1])

for i in range(1000):
    W,H = iterationEuc(W,H,Oxx)
    A,B = iterationKL(A,B,Oxx)

Wp = np.zeros(W.shape)
Ap = np.zeros(A.shape)
for i in range(W.shape[1]):
    Wp.transpose()[i] = W.transpose()[i]/norme(W.transpose()[i])
    Ap.transpose()[i] = A.transpose()[i]/norme(A.transpose()[i])

Wpm, Apm =np.dot(Wp.transpose(),Wp)*(np.dot(Wp.transpose(),Wp)>0.45), np.dot(Ap.transpose(),Ap)*(np.dot(Ap.transpose(),Ap)>0.45)

elements = list(range(R))
clusters = []
while not(not(elements)):
    e = elements[0]
    clusters += [[elements[elements.index(i)] for i in range(R) if (Wpm[e,i]!=0) and (i in elements)]]
    for i in clusters[-1]:
        elements.pop(elements.index(i))

plt.figure()    
for i in range(len(clusters)):
    plt.plot(t,list(map(sum,H[clusters[i]].transpose())))
    
plt.figure()
for i in range(len(clusters)):
    plt.plot(freqs[:800],list(map(sum,W[:,clusters[i]])))
    
    
    
    
    
    
    
    