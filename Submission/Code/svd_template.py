import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
def SVD(A, s, k):
    #A = X
    # TODO: Calculate probabilities p_i
    n,m = A.shape
    probs = [] #p1 ,p2, p3 .....pn
    for i in range (n):
        num = np.linalg.norm(A[i, :], ord=2)**2
        dnm = np.linalg.norm(A)**2
        pi = num/dnm
        probs.append(pi)
    # TODO: Construct S matrix of size s by m
    S = np.zeros((s,m))
    for i in range(s):
        j = np.random.choice(range(n),p=probs)
        S[i] = A[j,:]

    # TODO: Calculate SS^T
    sst = np.matmul(S, S.T)    #s,m * m,s
    # TODO: Compute SVD for SS^T
    W , lam2 , WT = np.linalg.svd(sst)
    # TODO: Construct H matrix of size m by k
    H = np.zeros((k,m))
    for i in range(k):
        STwt = np.matmul(S.T, W[:, i])
        h_i = STwt / np.linalg.norm(STwt, ord=2)
        H[i]=h_i
    lam2 = sorted(lam2, reverse=True)
    lam2= lam2[:k]
    # Return matrix H and top-k singular values sigma
    return H.T, lam2

    

def main():
    im = Image.open("../../Data/baboon.tiff")
    A = np.array(im)

    # TO DO: Compute SVD for A and calculate optimal k-rank approximation for A.
    k = 60
    U , S , Vh = np.linalg.svd(A)
    S= np.array(S[:k])
    LamdaMatrix = np.zeros((512,512))
    for i in range(k):
        LamdaMatrix[i][i] = S[i]
    optimal_approx = np.matmul(np.matmul(U, LamdaMatrix[:, :k]), Vh[:k, :])

    # TO DO: Use H to compute sub-optimal k rank approximation for A
    H, sigma = SVD(A, 80, 60)
    sub_optimal_approx = np.matmul(np.matmul(A, H), H.T)

    #To DO: Generate plots for original image, optimal k-rank and sub-optimal k rank approximation
    plt.title("Original Image")
    plt.imshow(A)
    plt.show()
    plt.savefig('../Figures/Orignal_Image.png')

    plt.title("Optimal K Rank Image")
    plt.imshow(optimal_approx)
    plt.show()
    plt.savefig('../Figures/Optimal_K_Rank_approximation.png')

    plt.title("Sub-Optimal K Rank Image")
    plt.imshow(sub_optimal_approx)
    plt.show()
    plt.savefig('../Figures/Sub-Optimal_K_Rank.png')




    # TO DO: Calculate the error in terms of the Frobenius norm for both the optimal-k
    # rank produced from the SVD and for the k-rank approximation produced using
    print("Optimal Error for Q2.3.b")
    print(np.linalg.norm(optimal_approx - A, ord='fro'))
    # sub-optimal k-rank approximation for A using H.
    print("Sub-Optimal Error for Q2.3.b")
    print(np.linalg.norm(sub_optimal_approx - A, ord='fro'))

if __name__ == "__main__":
    main()
