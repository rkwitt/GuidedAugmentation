"""Fink05a.py

Implementation of the one-shot learning idea of 

M. Fink
Object Classification from a Single Example Utilizing Class Relevance Metrics
NIPS 2005
"""


# Author: Roland Kwitt, Sebastian Hegenbart, Univ. of Salzburg, 2015
# E-Mail: roland.kwitt@gmail.com


# Standart imports
import os
import sys
import pickle
import numpy as np 
import ConfigParser
from optparse import OptionParser
import pylab as Plot
import pdb

# ML imports
from sklearn import linear_model
from sklearn import preprocessing
from sklearn import cross_validation
from sklearn.svm import SVR, LinearSVC, SVC
from sklearn.tree import DecisionTreeRegressor
from sklearn.decomposition import PCA
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import accuracy_score, confusion_matrix, r2_score, mean_squared_error
from sklearn.metrics.pairwise import euclidean_distances
from sklearn import preprocessing
from sklearn import neighbors

# Metric learning
from metric_learn import ITML, LSML, NCA, LMNN

# Helper stuff
from core.tsne import tsne
from core.tools import split_by_categories, select_random_anchors, build_index
from core.parser import load_annotation_scores
from core import logger



def setup_parser():
    """Setup the CLI parsing.

    Returns
    -------

    parser : OptionParser object.
    """

    parser = OptionParser()
    parser.add_option("-a", "--annotation_file", help="Annotation file.")
    parser.add_option("-t", "--attribute_file", help="Attributes file.")
    parser.add_option("-f", "--feature_file", help="File with features (e.g., CNN, attributes).")
    parser.add_option("-e", "--seed", help="RNG seed (e.g., 1234)", default=1234)
    parser.add_option("-p", "--comp", help="Number of PCA components", default=-1, type="int")
    parser.add_option("-n", "--num_constraints", help="Number of metric learning constraints.", type="int", default=200)
    parser.add_option("-r", "--report_file", help="Report file to output.")
    parser.add_option("-v", action="store_true", default=False, dest="verbose", help="Verbose output.")
    return parser


def main(argv=None):
    if argv is None:
        argv = sys.argv
    
    # Setup parser
    parser = setup_parser()
    (options, args) = parser.parse_args()

    # Load configuration
    config = ConfigParser.ConfigParser()
    config.readfp(open('config.cfg'))

    seed = int(options.seed)
    train_ratio = float(config.get('Setup', 'train_ratio'))

    if options.verbose:
        logger.info("Seed=%d" % seed)
        logger.info("Train ratio=%.2f" % train_ratio)

    # Load meta data
    annotations, attributes, img_file_list, categories, labels, GTM = load_annotation_scores( 
    	options.annotation_file, 
    	options.attribute_file)
 
    # Load feature data
    X = np.load(options.feature_file) # CNN features
    assert (X.shape[0] == len(img_file_list))

    if options.comp > 0:
        np.random.seed(seed)
        pca = PCA(n_components=options.comp,whiten=False)
        pca.fit(X[np.random.choice(X.shape[0],2000,replace=False),:])
        X = pca.fit_transform(X)

    mms = preprocessing.MinMaxScaler(feature_range=(-1,1))
    X = mms.fit_transform(X)

    if options.verbose:
        logger.info("Loaded %d x %d feature (CNN or attributes) matrix!" % 
            (X.shape[0], X.shape[1]))

    # Split data according to RNG seed
    trn_idx, tst_idx = split_by_categories(
        labels, 
        alpha=train_ratio, 
        random_state=seed)

    X_trn = X[trn_idx,:] # CNN features or attributes
    X_tst = X[tst_idx,:] # CNN features or attributes
    y_trn = np.asarray(labels)[trn_idx] # Train labels
    y_tst = np.asarray(labels)[tst_idx] # Test labels

    # Get anchors, i.e., the one-shot examples (one per category)
    y, selection = select_random_anchors(y_tst, seed)
    X_anchors = X_tst[selection,:]
    
    # Strategy:
    #
    # (1) Metric learning on the external data
    # (2) Transform one-shot data via learned transform
    # (3) Learn 1NN classifier
    # (4) Test on transformed test data (without anchors, obviously)

    # Part (1)
    transformer = LSML()
    C = LSML.prepare_constraints(y_trn, options.num_constraints)
    transformer.fit(X_trn, C, verbose=False)

    X_tst = np.delete(X_tst, selection, axis=0)
    y_tst = np.delete(y_tst, selection, axis=0)

    scores = []
    #---------------------------------------------
    # Train 1-NN using transformed anchors
    #---------------------------------------------
    nn_clf = neighbors.KNeighborsClassifier(n_neighbors=1).fit(
        transformer.transform(X_anchors), y)
    nn1_predictions = nn_clf.predict(transformer.transform(X_tst))
    scores.append({
        'p' : nn1_predictions,
        't' : y_tst, 
        'desc' : '1NN-Fink05a',
        'n_trn' : X_anchors.shape[0],
        'n_tst' : X_tst.shape[0]})

    #---------------------------------------------
    # Train SVM using transformed anchors
    #---------------------------------------------
    svc = LinearSVC(C=1.0).fit(transformer.transform(X_anchors),y)
    svc_predictions = svc.predict(transformer.transform(X_tst))
    scores.append({
        'p' : svc_predictions, 
        't' : y_tst, 
        'desc' : 'SVM-Fink05a',
        'n_trn' : X_anchors.shape[0],
        'n_tst' : X_tst.shape[0]})

    #---------------------------------------------
    # Train 1-NN using original anchors
    #---------------------------------------------
    nn_clf = neighbors.KNeighborsClassifier(n_neighbors=1).fit(X_anchors, y)
    ref_predictions = nn_clf.predict(X_tst)

    if options.verbose:
        logger.info("Reference (SVM+ML): %.2f" % (100.0*accuracy_score(y_tst,svc_predictions)))
        logger.info("Reference (1NN+ML): %.2f" % (100.0*accuracy_score(y_tst,nn1_predictions)))
        logger.info("Reference (1NN-ML): %.2f" % (100.0*accuracy_score(y_tst,ref_predictions)))

    scores_file = os.path.join(options.report_file)
    pickle.dump(scores, open(scores_file, "wb" ))


if __name__ == "__main__":
    sys.exit( main() )
