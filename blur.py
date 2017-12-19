# coding=utf-8

import numpy as np                          # use ndarray and calc
from scipy import misc, signal              # imsave, convolve2d
from matplotlib import pylab as plt         # plot
from PIL import Image                       # open Image
from tqdm import tqdm

###############################
########## CONSTANTS ##########
###############################
FILENAME = "paper.gif"



###############################
########## FUNCTIONS ##########
###############################
def open_gray_img(path):
    return np.array(Image.open(path).convert('L'))

def blur(img, d):
    fil = np.zeros((d, d))
    fil[d//2, :] = 1/d
    return signal.convolve2d(img, fil, mode='same', boundary='wrap')

def power_spectrum(img):
    return np.log1p( np.abs( np.fft.fftshift( np.fft.fft2(img) ) ) )



####################################
########## MAIN FUNCTIONS ##########
####################################
def main():
    #img = open_gray_img(FILENAME)
    img = misc.ascent()
    #misc.imsave("img.png", img)

    H, W = img.shape[:2]

    # motion blur size
    d = 31
    # blur
    bimg = blur(img, d=d)
    # add noise
    noise = np.random.rand(H, W) * np.average(bimg)*0.05
    noise -= noise.max()/2
    nbimg = bimg + noise

    #misc.imsave("bimg.png", bimg)
    #misc.imsave("nbimg.png", nbimg)

    # algebra
    A = np.zeros((W,W))
    for i in range(W):
        for j in range(d):
            k = (i - d//2 + j) % W
            A[i, k] = 1/d
    A = np.matrix(A)
    A_inv = A.I

    # only N row
    """
    N = 0
    b = np.matrix(nbimg[N, :]).T
    #b = np.matrix(bimg[N, :]).T
    x_hat = A_inv * b

    plt.figure()
    plt.plot(img[N, :], label="origin")
    plt.plot(x_hat, label="deblurred")
    plt.legend()
    plt.savefig("N{}".format(N))
    return
    """

    # all rows
    """
    dbimg = np.zeros((H, W))
    for N in range(H):
        #b = np.matrix(bimg[N, :]).T
        b = np.matrix(nbimg[N, :]).T
        x_hat = A_inv * b
        dbimg[N, :] = x_hat.T
    #misc.imsave("dbimg.png", dbimg)
    #misc.imsave("dnbimg.png", dbimg)
    return
    """

    # deblur ~ use SVD ~
    U, S, V = np.linalg.svd(A, full_matrices=True)
    A_svd = U * np.diag(S) * V

    #misc.imsave("A.png", A)
    #misc.imsave("A-SVD.png", A_svd)

    x_hat = np.matrix(np.zeros(len(S))).T

    phi_tsvd = lambda i, k: int(i < k)
    phi_tik = lambda s, a: s**2 / (s**2 + a**2)

    # deblur
    dnbimg = np.zeros(img.shape)
    #alpha = np.average(S)
    for j in range(H):
        b = np.matrix(nbimg[j, :]).T
        x_hat = np.matrix(np.zeros(len(S))).T

        for i, s in enumerate(S):
            tmp = (U[:, i].T * b)[0, 0] * V.T[:, i] / s
            x_hat += tmp * phi_tsvd(i, 133)
            #x_hat += tmp * phi_tik(S[i], alpha)
        dnbimg[j, :] = x_hat.T

    misc.imsave("dnbimg.png", dnbimg)

    N = 100
    plt.figure()
    plt.plot(img[N, :])
    plt.plot(dnbimg[N, :])
    plt.savefig("N{}".format(N))


    # search best Truncated SVD parameters
    """
    errors = []
    for k in tqdm(range(130, 140, 1)):
        dnbimg = np.zeros(img.shape[:2])
        for j in range(H):
            b = np.matrix(nbimg[j, :]).T
            x_hat = np.matrix(np.zeros(len(S))).T

            for i, s in enumerate(S):
                tmp = (U[:, i].T * b)[0, 0] * V.T[:, i] / s
                x_hat += tmp * phi_tsvd(i, k)
            dnbimg[j, :] = x_hat.T

        errors.append(np.average((img - dnbimg)**2))
    
    plt.figure()
    plt.plot(errors)
    plt.show()
    """

    """
    plt.figure()
    plt.xscale("log")
    plt.plot(S, [phi_tsvd(i, W//2) for i, _ in enumerate(S)])
    plt.xlabel("singular values")
    plt.ylabel("filter factor")
    plt.savefig("tsvd")

    plt.figure()
    plt.xscale("log")
    plt.plot(S, [ phi_tik(s, 0.03) for s in S ])
    plt.xlabel("singular values")
    plt.ylabel("filter factor")
    plt.savefig("tik")
    """

    return


    misc.imsave("blur.png", b)
    misc.imsave("blur_fft.png", power_spectrum(b))
    #misc.imsave("deblur.png", db)
    #misc.imsave("deblur_fft.png", power_spectrum(db))



##########################
########## MAIN ##########
##########################
if __name__ == '__main__':
    main()

