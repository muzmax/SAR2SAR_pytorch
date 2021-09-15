import numpy as np
from numpy.core.defchararray import add
import seaborn as sns
import matplotlib.pyplot as plt

class add_speckle():
    def __init__(self,L=1) -> None:
        self.L_=L
    
    def __call__(self,im):
              
        dim = im.shape
        s = np.zeros(dim)
        
        for k in range(0,self.L_):
            
            real = np.random.normal(size=dim)
            imag = np.random.normal(size=dim)
            gamma = (np.abs(real + 1j*imag)**2)/2 
            s+=gamma
        s = s/self.L_
        speck_im = im**2 * s
        return speck_im

L=10
sp = add_speckle(L=L)
im = np.ones(10000)
im = sp(im)

sns.histplot(im,stat='density')
plt.xlabel('values of speckle in intensity')
plt.title('mean:{}, var:{}'.format(round(np.mean(im),4),round(np.var(im),4)))
plt.show()

