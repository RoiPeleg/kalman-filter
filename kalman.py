import numpy as np


def kalman():
    print('s')


""""
predict stage 
returns new state vector X and the new P varince
"""
def predict(x_prev, P, F):
    u = np.array([[5,0],[0,15]])
    B = np.array([[0.5,0,1,0],[0,0.5,0,1]])
    print(B.dot(u))
    x_new = F.dot(x_prev) + B.dot(u)
    P_new = F.dot(P).dot(np.transpose(F))
    return x_new, P_new


def measure(x_k, P, H, z):
    K_g = P.dot(np.transpose(H)).dot(
        np.linalg.inv(H.dot(P).dot(np.transpose(H))))
    x_new = x_k + K_g.dot(z - H.dot(x_k))

    P_new = (1 - K_g.dot(H)).dot(P)
    return x_new, P_new


def main():
    dt = 1  # delta t
    H = np.array([[1, 0, 0, 0], [0, 1, 0, 0]])
    P = np.array([[7**2, 0, 0, 0], [0, 7**2, 0, 0],
                 [0, 0, 100**2, 0], [0, 0, 0, 100**2]])
    F = np.array([[1, 0, dt, 0], [0, 1, 0, dt], [0, 0, 0, 0], [0, 0, 0, 0]])
    print('H:\n', H)
    print('F:\n', F)
    print('P:\n', P)
    #state vector
    x_0 = np.array([200, 150, 0, 0])
    # sensor measuerments
    z = [np.array([240, 204]), np.array([284, 267]),
         np.array([334, 344]), np.array([390, 437]),
         np.array([450, 544]), np.array([516, 667])]

    # for m in z:
    #     x_0, P = predict(x_0, P, F)
    #     x_0, P = measure(x_0, P, H, m)
    x_0, P = predict(x_0, P, F)
    x_0, P = measure(x_0, P, H,z[0])    
    print('x: ', x_0)
    print('p: ', P)


if __name__ == '__main__':
    main()
