import cv2
import glob
import matplotlib.pyplot as plt
import os
from pylab import *
from scipy.optimize import curve_fit
import pandas as pd

def generate_NDVI(image):
    rChannel = image[:,:,0]
    bChannel = image[:,:,2]

    NDVI = (1.706*bChannel - 0.706*rChannel) / (0.294*bChannel + 0.706*rChannel)

    return(NDVI)

def save_NDVI(ndvi,name,path):
    plt.imshow(ndvi)
    plt.axis('off')
    plt.imsave(path+name,ndvi)

def show_hist(ndvi):
    plt.hist(ndvi.ravel(),100,[-1,1])
    plt.show()

def gauss(x,mu,sigma,A):
    return A*exp(-(x-mu)**2/2/sigma**2)

def bimodal(x,mu1,sigma1,A1,mu2,sigma2,A2):
    return gauss(x,mu1,sigma1,A1)+gauss(x,mu2,sigma2,A2)

def generate_analysis_img(image,name="sample",show=False,save=True):
    ndvi = generate_NDVI(image)
    try:
        os.mkdir("NDVI_analysis")
    except:
        pass

    ax1 = plt.subplot(212)
    ax1.margins(0.0)           # Default margin is 0.05, value 0 means fit
    y,x,_ = ax1.hist(ndvi.ravel(),100,[-1,1],label='data')
    
    x=(x[1:]+x[:-1])/2

    
    params,cov=curve_fit(gauss,x,y, maxfev=5000)

    if(params[1]<0.11):
        sigma=sqrt(diag(cov))

        ax1.plot(x,gauss(x,*params),color='red',lw=3,label='model')
        #print(sigma)
        mean = params[0]
        sd = params[1]
        ax1.set_title("Histogram, mean="+str(round(params[0],5))+",sigma="+str(round(params[1],5)))
        ax1.legend()
    else:
        expected=(-0.1,.07,50000,0.25,.05,125000)
        params,cov=curve_fit(bimodal,x,y,expected, maxfev=100000,bounds=([-1,0.00001,0,-1,0.00001,0],[1,10,np.inf,1,10,np.inf]))
        sigma=sqrt(diag(cov))
        #print(sigma)
        plot(x,bimodal(x,*params),color='red',lw=3,label='model')
        mean = params[3]
        sd = params[4]
        ax1.set_title("Histogram, mean="+str(round(params[3],5))+",sigma="+str(round(params[4],5)))
        #print(list(zip(bimodal.__code__.co_varnames[1:],params)))
        legend()
    # print(params)

    ax2 = plt.subplot(221)
    ax2.imshow(image)
    ax2.axis('off')
    ax2.set_title('Original')

    ax3 = plt.subplot(222)
    ax3.imshow(ndvi)
    ax3.axis('off')
    ax3.set_title('NDVI')

    if save:
        plt.savefig("NDVI_analysis/"+name+".jpg")

    if show:
        plt.show()
    else:
        plt.clf()
        plt.close()
    
    return(mean,abs(sd))

def main():
    files = glob.glob("*Images/*.jpg")

    df = pd.read_csv("UWYieldData.csv")

    mu_ndvi = np.zeros(len(df))
    sigma_ndvi = np.zeros(len(df))

    for f in files:
        print(f)
        d = f.split(" ")[0].replace(".","/")+"/2019"
        #print(d)
        g = f.split("/")[1].split(" ")[0]
        #print(g)

        df.loc[df['Date']==d].loc[df['GCP']==g]['Mean_NDVI']=1
        i = df.loc[df['Date']==d].loc[df['GCP']==g].index[0]

        #print(df.loc[df['Date']==d].loc[df['GCP']==g].index[0])
        name = f.split("/")[1]
        name = name[:-4]+"_NDVI.jpg"
        img = plt.imread(f)
        m,s = generate_analysis_img(img,name=name)
        mu_ndvi[i] = m
        sigma_ndvi[i] = s

    df['Mean_NDVI']=mu_ndvi
    df['SD_NDVI']=sigma_ndvi

    df.to_csv("UWYieldDataNDVI.csv",index=False)

if __name__ == "__main__":
    main()
