import numpy as np

""""
predict stage 
returns new state vector X and the new P variance
"""
def predict(x_prev, P, F):
    u = np.array([0,0,5,15])
    B = np.array([[0,0,0.5,0],[0,0,0,0.5],[0,0,1,0],[0,0,0,1]])
    x_new = F.dot(x_prev) + B.dot(u)
    P_new = F.dot(P).dot(np.transpose(F))
    return x_new, P_new

"""
gets a new measurement from the system and refines the guess
"""
def measure(x_k, P, H, z):
    R = np.array([[7,0],[0,7]])
    K_g = P.dot(np.transpose(H)).dot(np.linalg.inv(H.dot(P).dot(np.transpose(H)) + R )) #kalman gain
    x_new = x_k + K_g.dot(z - H.dot(x_k))
    P_new = (1 - K_g.dot(H)).dot(P)
    return x_new, P_new


def main():
    dt = 1  # delta t
    H = np.array([[1, 0, 0, 0], [0, 1, 0, 0]])
    P = np.array([[7, 0, 0, 0], [0, 7, 0, 0],
                 [0, 0, 100, 0], [0, 0, 0, 100]])
    F = np.array([[1, 0, dt, 0], [0, 1, 0, dt], [0, 0, 0, 0], [0, 0, 0, 0]])
    print('H:\n', H)
    print('F:\n', F)
    print('P:\n', P)
    #state vector
    x_0 = np.array([200, 150, 0, 0])
    # sensor measurements
    z = [np.array([240, 204]), np.array([284, 267]),
         np.array([334, 344]), np.array([390, 437]),
         np.array([450, 544]), np.array([516, 667])]

    for m in z:
        x_0, P = predict(x_0, P, F)
        x_0, P = measure(x_0, P, H, m)
    print('x: ', x_0)
    print('p:\n', P)


if __name__ == '__main__':
    main()
