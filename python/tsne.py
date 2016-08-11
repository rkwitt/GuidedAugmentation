import numpy as np
from sklearn.manifold import TSNE
import sys


if __name__ == '__main__':

    X = np.load(sys.argv[1])

    model = TSNE(n_components=2, random_state=0)
    #np.set_printoptions(suppress=True)

    np.savetxt(sys.argv[2], model.fit_transform(X));
