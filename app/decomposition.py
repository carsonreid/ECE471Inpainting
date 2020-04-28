import numpy as np
import math
import config
import cv2
iterations = 10
lambd = 0.1
mu = 0.1


def decompose(im):
    # Note: this uses border reflection
    u = np.copy(im)
    newu = np.zeros(im.shape)

    kernX = np.array([[-1, 0, 1],
                      [-1, 0, 1],
                      [-1, 0, 1]])
    kernY = np.array([[-1, -1, -1],
                        [0, 0, 0],
                        [1, 1, 1]])
    fx = cv2.filter2D(im, -1, kernX)
    fy = cv2.filter2D(im, -1, kernY)

    g1 = np.zeros(im.shape)
    g2 = np.zeros(im.shape)
    """apply initialization of -1/2lambda * fx/|∇f|"""
    for i in range(len(im)):
        for j in range(len(im[0])):
            for channel in range(3):
                g1[i][j][channel] = -fx[i][j][channel] / (np.linalg.norm([fx[i][j][channel], fy[i][j][channel]]) * 2 * lambd)
                g2[i][j][channel] = -fy[i][j][channel] / (np.linalg.norm([fx[i][j][channel], fy[i][j][channel]]) * 2 * lambd)

    newg1 = np.zeros(im.shape)
    newg2 = np.zeros(im.shape)

    for z in range(iterations):
        if config.verbose:
            print("Decomposition iteration "+str(z))
        for channel in range(3):
            # Update u
            for i in range(len(u)):
                """precalculate reflection padding indices for the image"""
                ip1 = i + 1 if i + 1 < len(im) else len(im) - 1
                im1 = i - 1 if i - 1 >= 0 else 0
                for j in range(len(u[0])):
                    """precalculation again"""
                    jp1 = j+1 if j+1 < len(im[0]) else len(im[0])-1
                    jm1 = j-1 if j-1 >= 0 else 0

                    """c values from paper, pp.10"""
                    c1 = 1/(math.sqrt(
                        ((u[ip1][j][channel]-u[i][j][channel]))**2
                        +
                        ((u[i][jp1][channel]-u[i][jm1][channel])/2)**2
                    )+1)  # TODO: check what to do about div by 0
                    c2 = 1/(math.sqrt(
                        (u[i][j][channel]-u[im1][j][channel])**2
                        +
                        ((u[im1][jp1][channel]-u[im1][jm1][channel])/2)**2
                    )+1)  # TODO: check what to do about div by 0
                    c3 = 1/(math.sqrt(
                        ((u[ip1][j][channel]-u[im1][j][channel])/2)**2
                        +
                        (u[i][jp1][channel]-u[i][j][channel])**2
                    )+1)  # TODO: check what to do about div by 0
                    c4 = 1/(math.sqrt(
                        ((u[ip1][jm1][channel]-u[im1][jm1][channel])/2)**2
                        +
                        (u[i][j][channel]-u[i][jm1][channel])**2
                    )+1)  # TODO: check what to do about div by 0

                    """calculate next iteration values for u(n+1), g1(n+1), g2(n+1)"""
                    newu[i][j][channel] = (im[i][j][channel]
                                           -
                                           ((g1[ip1][j][channel]-g1[im1][j][channel]) + (g2[i][jp1][channel]-g2[i][jm1][channel]))/2
                                           +
                                           (c1*u[ip1][j][channel] + c2*u[im1][j][channel]
                                            + c3*u[i][jp1][channel] + c4*u[i][jm1][channel])/2*lambd
                                           )\
                                          * \
                                          1/(1 + (c1+c2+c3+c4)/(2*lambd))

                    newgFirstTerm = (2*lambd)/(mu*H(g1[i][j][channel], g2[i][j][channel]) + (4*lambd))
                    newg1[i][j][channel] = newgFirstTerm \
                                           * \
                                           (
                                                   (u[ip1][j][channel]-u[im1][j][channel])/2
                                                   -
                                                   (im[i][j][channel]-im[im1][j][channel])/2
                                                   +
                                                   (g1[ip1][j][channel]+g1[im1][j][channel])/2
                                                   +
                                                   (2*g2[i][j][channel] + g2[im1][jm1][channel] + g2[ip1][jp1][channel]
                                                    - g2[i][jm1][channel] - g2[im1][j][channel] - g2[ip1][j][channel]
                                                    - g2[i][jp1][channel])/2
                                           )
                    newg2[i][j][channel] = newgFirstTerm \
                                           * \
                                           (
                                                   (u[ip1][j][channel] - u[im1][j][channel]) / 2
                                                   -
                                                   (im[i][j][channel] - im[im1][j][channel]) / 2
                                                   +
                                                   (g2[ip1][j][channel] + g2[im1][j][channel]) / 2
                                                   +
                                                   (2 * g1[i][j][channel] + g1[im1][jm1][channel] + g1[ip1][jp1][
                                                       channel]
                                                    - g1[i][jm1][channel] - g1[im1][j][channel] - g1[ip1][j][channel]
                                                    - g1[i][jp1][channel]) / 2
                                           )

        u = np.copy(newu)
        g1 = np.copy(newg1)
        g2 = np.copy(newg2)

    return u, np.subtract(im, u)

"""from vese+osher pp.8, this is simplified for p=1 since they noticed no differences in testing for p>1"""
def H(i, j):
    return 1/(math.sqrt(i**2 + j**2)+1)  # TODO: check what to do about div by 0

