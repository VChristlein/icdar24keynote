import pickle
import os
import shlex
import argparse
from tqdm import tqdm
import gzip
from sklearn.cluster import MiniBatchKMeans
from sklearn.preprocessing import normalize, StandardScaler
import numpy as np
import numpy.ma as ma

import cv2
from sklearn.decomposition import PCA
from parmap import parmap
from sklearn.linear_model import Ridge
from sklearn.svm import LinearSVC

from reranking import sgr
import torch
def parseArgs(parser):
    parser.add_argument('--tmp_folder', default='tmp', 
                        help='default temporary folder')
    parser.add_argument('--labels_test',default='icdar17_labels_test.txt',
                        help='contains test images/descriptors to load + labels')
    parser.add_argument('--labels_train',default='icdar17_labels_train.txt',
                        help='contains training images/descriptors to load + labels')
    parser.add_argument('-str', '--suffix_train',
                        default='.png',
                        help='only chose those images with a specific suffix')
    parser.add_argument('-ste', '--suffix_test',
                        default='.jpg',
                        help='only chose those images with a specific suffix')
    parser.add_argument('--to_binary', action='store_true',
                       help='use OTSU binarization')
    parser.add_argument('--in_test',
                        help='the input folder of the training images / features')
    parser.add_argument('--in_train',
                        help='the input folder of the training images / features')
    parser.add_argument('--overwrite', action='store_true',
                        help='do not load pre-computed encodings')
    parser.add_argument('--powernorm', action='store_true',
                        help='use powernorm')
    parser.add_argument('--n_clusters', default=100, type=int,
                        help='number of clusters')
    parser.add_argument('--n_mvlad', default=1, type=int,
                        help='number of multi-vlad')
    parser.add_argument('--gmp', action='store_true',
                        help='use generalized max pooling')
    parser.add_argument('--gamma', default=1000, type=float,
                        help='regularization parameter of GMP')
    parser.add_argument('--esvm', action='store_true',
                        help='run esvm')
    parser.add_argument('--iesvm', action='store_true',
                        help='run intermediate esvm')
    parser.add_argument('--C', default=1000, type=float, 
                        help='C parameter of the SVM')
    parser.add_argument('--pca_comps', default=-1, type=int,
                        help='use pca for the final descriptor')
    parser.add_argument('--ipca_comps', default=-1, type=int,
                        help='use pca on SIFT')
    parser.add_argument('--rerank', action='store_true', 
                        help='sgr reranking')
    parser.add_argument('--standardize', action='store_true', 
                        help='standardize')
    parser.add_argument('--rm_duplicates', action='store_true',
                        help='remove keypoint duplicates')

    return parser


def getFiles(folder, pattern, labelfile):
    """
    returns files and associated labels by reading the labelfile
    parameters:
        folder: inputfolder
        pattern: new suffix
        labelfiles: contains a list of filename and labels
    return: absolute filenames + labels
    """
    # read labelfile
    with open(labelfile, 'r') as f:
        all_lines = f.readlines()

    # get filenames from labelfile
    all_files = []
    labels = []
    check = True
    for line in all_lines:
        # using shlex we also allow spaces in filenames when escaped w. ""
        if labelfile.endswith('.txt'):
            splits = shlex.split(line)
        elif labelfile.endswith('.csv'):
            splits = line.rstrip('\n').split(';')
        else:
            raise ValueError('wrong labelfile filetype')

        file_name = splits[0]
        class_id = splits[1]

        # strip all known endings, note: os.path.splitext() doesnt work for
        # '.' in the filenames, so let's do it this way...
        for p in ['.pkl.gz', '.txt', '.png', '.jpg', '.tif', '.ocvmb', '.csv']:
            if file_name.endswith(p):
                file_name = file_name.replace(p, '')

        # get now new file name
        true_file_name = os.path.join(folder, file_name + pattern)
        all_files.append(true_file_name)
        labels.append(class_id)

    return all_files, labels
def remove_duplicates(keypoints, img_shape):
    """
    Remove duplicate keypoints retaining the one with the highest response.
    Parameters:
        keypoints: List of keypoints, where each keypoint has attributes `pt` (tuple) and `response`.
        img_shape: Shape of the image from which keypoints are detected (for boundary checks if needed).
    Returns:
        Array of filtered keypoints.
    """
	# old
#	kpt_coord = np.zeros(img.shape, dtype=int)
#	for e,kp in enumerate(keypoints):
#		tt = kpt_coord[ int(kp.pt[1]), int(kp.pt[0]) ]
#		# keep kpt w. strongest response
#		if tt == 0 or (tt != 0 and kp.response > keypoints[tt].response):
#			kpt_coord[ int(kp.pt[1]), int(kp.pt[0]) ] = e
#
#	rel_coords = np.nonzero(kpt_coord.ravel())
#	ind = kpt_coord.ravel()[rel_coords]
#	return np.array(keypoints)[ ind ]

	# this should be faster:
    # Dictionary to hold the best keypoint for each coordinate
    best_keypoints = {}
    for kp in keypoints:
        coord = (int(kp.pt[1]), int(kp.pt[0]))  # Tuple of y, x coordinates
        if coord not in best_keypoints or kp.response > best_keypoints[coord].response:
            best_keypoints[coord] = kp
    
    # Extract the best keypoints from the dictionary
    filtered_keypoints = list(best_keypoints.values())
    return np.array(filtered_keypoints)



def computeKpts(img, sampling='keypoints', angle2zero=True, rm_duplicates=True):
    """ compute the keypoints, i.e. where to extract the descriptors
    parameters:
        img: grayscale image
        sampling: canny, or keypoints
        angle2zero: only if keypoints -> set angle to 0
    """
    assert (img is not None)
    if sampling == 'keypoints':
        #sift = cv2.SIFT_create() # cv2.SIFT_create() instead of cv2.xfeatures2d.SIFT_create()
        # slightly better results
        # note that more octave-layers don't improve the results, but less
        # decrease them
        sift = cv2.SIFT_create(contrastThreshold = 0.01, edgeThreshold = 40,
                               enable_precise_upscale=True)

        keypoints = sift.detect(img, None)
        if angle2zero:
            for kp in keypoints:
                kp.angle = 0

        if rm_duplicates:
            keypoints = remove_duplicates(keypoints, img.shape)
        return keypoints

    if sampling == 'canny':
        keypoints = []
        edgeImg = cv2.Canny(img, 50, 200)
        for y in range(edgeImg.shape[0]):
            for x in range(edgeImg.shape[1]):
                if edgeImg[y, x] == 255:
                    keypoints.append(cv2.KeyPoint(x, y, 1))

        return keypoints

    return None


def computeSIFT(img, keypoints,norm_hellinger=True):
    """ compute SIFT at specific keypoints
    """
    sift = cv2.SIFT_create(enable_precise_upscale=True)
    _, descriptors = sift.compute(img, keypoints) # None anstatt norm_hellinger

    return descriptors


def loadRandomDescriptors(files, max_descriptors, rm_duplicates=True):
    """
    compute roughly `max_descriptors` random local descriptors of dimension D from the files.
    parameters:
        files: list of image filenames
        max_descriptors: maximum number of descriptors (Q)
    returns: QxD matrix of descriptors
    """
    max_files = 100
    indices = np.random.permutation(max_files)
    files = np.array(files)[indices]

    # rough number of descriptors per file that we have to load
    # Note: actually we could also choose to use a lower filenumber
    # but in this way features from all files will be used
    max_descs_per_file = int(max_descriptors / len(files))

    descriptors = []
    for i in tqdm(range(len(files))):
        desc = computeDescs(files[i], True, True,
                           rm_duplicates=rm_duplicates)
        # get some random ones
        indices = np.random.choice(len(desc),
                                   min(len(desc),
                                       int(max_descs_per_file)),
                                   replace=False)
        desc = desc[indices]
        descriptors.append(desc)

    descriptors = np.concatenate(descriptors, axis=0)
    return descriptors


def dictionary(descriptors, n_clusters):
    """
    return cluster centers for the descriptors
    parameters:
        descriptors: NxD matrix of local descriptors
        n_clusters: number of clusters = K
    returns: KxD matrix of K clusters
    """

	# TODO: possibly change n_init param (standard: 1)
    cluster = MiniBatchKMeans(n_clusters,
                              compute_labels=False,
                              batch_size=100 * n_clusters,
                              random_state=42).fit(descriptors)    
    return cluster.cluster_centers_


def assignments(descriptors, clusters):
    """
    compute assignment matrix
    parameters:
        descriptors: TxD descriptor matrix
        clusters: KxD cluster matrix
    returns: TxK assignment matrix
    """
    # compute nearest neighbors
    matcher = cv2.BFMatcher(cv2.NORM_L2)
    matches = matcher.knnMatch(descriptors.astype(np.float32),
                               clusters.astype(np.float32),
                               k=1)

    # create hard assignment
    assignment = np.zeros((len(descriptors), len(clusters)))
    for e, m in enumerate(matches):
        assignment[e, m[0].trainIdx] = 1

    return assignment

def toBinary(mask):
    # test if not already binary
    if mask[mask==255].sum() != np.sum(mask):
        # maybe binary between 0,1?
        if mask[mask==1].sum() == mask.sum():
            mask *= 255
        else: # make it binary
           ret, mask = cv2.threshold(mask, 125, 255, cv2.THRESH_OTSU + cv2.THRESH_BINARY)

    return mask

def computeDescs(fname, norm_hellinger=True, to_binary=False,
                rm_duplicates=True):
    img = cv2.imread(fname, cv2.IMREAD_GRAYSCALE)
    if img is None: 
        raise IOError('cannot read',fname)
    if to_binary:
        img = toBinary(img) 
        
    kpts = computeKpts(img, sampling='keypoints', angle2zero=True,
                      rm_duplicates=rm_duplicates)
    if kpts is None or len(kpts) == 0:
        raise ValueError('cannot find any kpt for:', fname)

    descs = computeSIFT(img, kpts)
    # Hellinger normalization
    if norm_hellinger:
        descs = normalize(descs, norm='l1') 
        descs = np.sign(descs) * np.sqrt(np.abs(descs))

    return descs

def powernormalize(encs):
    encs = np.sign(encs) * np.sqrt(np.abs(encs))
    encs = normalize(encs, norm='l2')
    return encs

def vlad(files, mus, powernorm, gmp=False, gamma=1000, 
         pca=None, rm_duplicates=True):
    """
    compute VLAD encoding for each files
    parameters:
        files: list of N files containing each T local descriptors of dimension
        D
        mus: KxD matrix of cluster centers
        gmp: if set to True use generalized max pooling instead of sum pooling
    returns: NxK*D matrix of encodings
    """
    K = mus.shape[0]
    encodings = []

    for f in tqdm(files):
        desc = computeDescs(f, True, True,
                           rm_duplicates=rm_duplicates)
        if pca is not None:
            desc = pca.transform(desc)
            desc = powernormalize(desc)
        a = assignments(desc, mus)

        T, D = desc.shape
        f_enc = np.zeros((D * K), dtype=np.float32)
        for k in range(K):
            # it's faster to select only those descriptors that have
            # this cluster as nearest neighbor and then compute the
            # difference to the cluster center than computing the differences
            # first and then select

            # get only descriptors that are possible for this cluster
            nn = desc[a[:, k] > 0]
            # it can happen that we don't have any descriptors associated for
            # this cluster
            if len(nn) > 0:
                res = nn - mus[k]
                if gmp:
                    clf = Ridge(alpha=gamma,
                             fit_intercept=False,
                             solver='sparse_cg', 
                             max_iter=500) # conjugate gradient                    
                    clf.fit( res, np.ones((len(nn))) )
                    f_enc[k*D:(k+1)*D] = clf.coef_
                # sum pooling
                else:
                    f_enc[k * D:(k + 1) * D] = np.sum(res, axis=0)
        
        encodings.append(f_enc)

    encodings = np.vstack(encodings)

    # c) power normalization
    if powernorm:
        encodings = powernormalize(encodings)
    else:
        # l2 normalization
        encodings = normalize(encodings, norm='l2')

    return encodings


def distances(encs):
    """
    compute pairwise distances
    parameters:
        encs:  TxK*D encoding matrix
    returns: TxT distance matrix
    """
    # compute cosine distance = 1 - dot product between l2-normalized
    # encodings
    dists = 1.0 - encs.dot(encs.T)   
    return dists

def evaluate(encs, labels, dist_matrix=None, rerank=True, sgr_gamma=0.9):
    evaluate_(encs, labels, dist_matrix)
    print('> rerank')
    # however it seems the last parameter depends on the features
    # -> the better they are, the smaller it probably should be chosen
    _, dists_r = sgr.sgr_reranking(torch.tensor(encs), 2, 1,
                                    sgr_gamma)
    evaluate_(None, labels, dists_r.numpy())


def evaluate_(encs, labels, dist_matrix=None):
    """
    evaluate encodings assuming using associated labels
    parameters:
        encs: TxK*D encoding matrix
        labels: array/list of T labels
    """
    if dist_matrix is None:
        dist_matrix = distances(encs)
   
    # mask out distance with itself
    np.fill_diagonal(dist_matrix, np.finfo(dist_matrix.dtype).max)
    
    # sort each row of the distance matrix
    indices = dist_matrix.argsort()

    n_encs = len(dist_matrix)

    mAP = []
    correct = 0
    for r in range(n_encs):
        precisions = []
        rel = 0
        for k in range(n_encs - 1):
            if labels[indices[r, k]] == labels[r]:
                rel += 1
                precisions.append(rel / float(k + 1))
                if k == 0:
                    correct += 1
        avg_precision = np.mean(precisions)
        mAP.append(avg_precision)
    mAP = np.mean(mAP)

    print('Top-1 accuracy: {} - mAP: {}'.format(float(correct) / n_encs, mAP))


def dump(fname, obj):
    error = False
    if fname.endswith('npy'):
        np.save(fname, obj.numpy()) 
    else:
        if not fname.endswith('.pkl.gz'):
            fname += '.pkl.gz'
        with gzip.open(fname, 'wb') as f_out:
            try:
                pickle.dump(obj, f_out)
            except TypeError as e:
                print('TypeError:',e)
                error = True
        # delete orphan
        if error and os.path.exists(fname):
            os.remove(fname)

    if not error:
        print('- dumped', fname)

def load(fname):
    # TODO: npy case
    if not fname.endswith('.pkl.gz'):
        fname += '.pkl.gz'
    with gzip.open(fname, 'rb') as f:
        mus = pickle.load(f)
    print('- loaded', fname)
    return mus


def runF(fname, overwrite, out_folder, func, *args, **kwargs):
    """
    before running a function it checks if we have already something saved
    """
    if not os.path.exists(out_folder):
        os.mkdir(out_folder)

    path = os.path.join(out_folder, fname)
    if not os.path.exists(path) or overwrite:
        ret = func(*args, **kwargs)
        dump(out_path, ret)
    else:
        ret = load(path)

    return ret

def esvm(encs_test, encs_train, C=1000, same=False):
    """ 
    compute a new embedding using Exemplar Classification
    parameters: 
        encs_test: NxD matrix
        encs_train: MxD matrix
        C: SVM margin parameter
        same: are encs_test and encs_train actually equal (can be used for
            cross-validating C
    returns: new encs_test matrix (NxD)
    """

    # compute for each test encoding an E-SVM using the
    # encs as negatives    
    labels = np.zeros(len(encs_train) + 1)
    labels[0] = 1

    to_classify = np.zeros((len(labels),encs_train.shape[1]),
                           dtype=encs_train.dtype)
    to_classify[1:] = encs_train
    def loop(i):
        esvm = LinearSVC(C=C, class_weight='balanced',dual='auto')
        to_classify[0] = encs_test[i]
        if same:
            # leave i+1 position out (first one is query)
            esvm.fit(to_classify[np.arange(len(labels))!=(i+1)], 
                     labels[np.arange(len(labels))!=(i+1)])
        else:
            esvm.fit(to_classify, labels)
        
        x = normalize(esvm.coef_, norm='l2')
        return x

    new_encs = list(parmap( loop, tqdm(range(len(encs_test)))))
#                           show_progress=True))
#    new_encs = list(map(loop, tqdm(range(len(encs_test)))))
    new_encs = np.concatenate(new_encs, axis=0)
    # return new encodings
    return new_encs


if __name__ == '__main__':
    parser = argparse.ArgumentParser('retrieval')
    parser = parseArgs(parser)
    args = parser.parse_args() # uebergarbeparameter
    np.random.seed(0)  # fix random seed
    print(args)
    # load files of training
    files_train, labels_train = getFiles(args.in_train, args.suffix_train,
                                         args.labels_train)
    assert (len(files_train) == len(labels_train))
    print('# train:', len(files_train))
    

    pca_v = None
    pca = None
    ipcas = []
    all_enc_train = []
    all_mus = []
    for i in range(args.n_mvlad):
        # a) dictionary
        print('> compute dictionary')
        descriptors = runF('rand_descs_{}.pkl.gz'.format(i), args.overwrite, args.tmp_folder,
                       loadRandomDescriptors, files_train, 150000)
        print('> loaded/computed {} descriptors:'.format(len(descriptors)))
        
        if args.ipca_comps >= 0:
            ipca_comps = args.ipca_comps if args.ipca_comps > 0 \
                else descriptors.shape[1]
            pca = PCA(ipca_comps,whiten=True)
            descriptors = pca.fit_transform(descriptors)
            descriptors = powernormalize(descriptors)
            ipcas.append(pca)

        mus = runF('dict_{}.pkl.gz'.format(i), args.overwrite, args.tmp_folder,
               dictionary, descriptors, args.n_clusters)
        all_mus.append(mus)

    # ... testing
    print('> compute VLAD for test')
    files_test, labels_test = getFiles(args.in_test, args.suffix_test,
                                       args.labels_test)
    assert (len(files_test) == len(labels_test))
    print('# test:', len(files_test))

    #    # b) VLAD encoding for test
    all_enc_test = []
    for i in range(args.n_mvlad):
        print('run ', i)
        enc_test = runF('encs_test_vlad_{}.pkl.gz'.format(i), args.overwrite,
                        args.tmp_folder,
                        vlad, files_test, all_mus[i], args.powernorm, args.gmp, args.gamma,
                        ipcas[i] if args.ipca_comps >= 0 else None,
                        rm_duplicates=args.rm_duplicates)
        evaluate(enc_test, labels_test)

        if args.iesvm or args.esvm or args.pca_comps >= 0 or args.standardize:
            # c) VLAD encoding training
            print('> compute VLAD for train')
            enc_train = runF('encs_train_vlad_{}.pkl.gz'.format(i), args.overwrite,
                             args.tmp_folder,
                             vlad, files_train, mus, args.powernorm, args.gmp, 
                             args.gamma,
                             ipcas[i] if args.ipca_comps >= 0 else None,
                             rm_duplicates=args.rm_duplicates)

            all_enc_train.append(enc_train)

        if args.iesvm:
            print('> intermediate esvm computation')
            # TODO: for a real setup you actually want to cross-validate C
    #        all_enc_test = esvm(all_enc_test, all_enc_train, args.C)
            enc_test_e = runF('encs_test_esvm_{}.pkl.gz'.format(i), args.overwrite,
                            args.tmp_folder, 
                            esvm, enc_test, all_enc_train[i], args.C)
            print('> evaluate')
            evaluate(enc_test_e, labels_test)

        # save here the original ones! not the e-svm transformed 
        all_enc_test.append(enc_test)

    # TODO
    if args.iesvm or args.esvm or args.pca_comps >= 0 or args.standardize:
        all_enc_train = np.concatenate(all_enc_train, axis=1)
        all_enc_test = np.concatenate(all_enc_test, axis=1)

    if args.pca_comps >= 0:
        print('fit PCA')
        if args.pca_comps == 0:
            comps = min(all_enc_train.shape[0],all_enc_train.shape[1])
        else:
            comps = args.pca_comps
        pca2 = PCA(comps, whiten=True)
        all_enc_train = pca2.fit_transform(all_enc_train)
        print('eval train')
        all_enc_train = powernormalize(all_enc_train)
        evaluate(all_enc_train, labels_train)


        print('perform pca')
        all_enc_test = pca2.transform(all_enc_test)
    
        print('> evaluate')
        evaluate(all_enc_test, labels_test)

        all_enc_test = powernormalize(all_enc_test)
        print('> evaluate after power norm')
        evaluate(all_enc_test, labels_test)

    if args.standardize:
        print('> standardize')
        scaler = StandardScaler()
        all_enc_train = scaler.fit_transform(all_enc_train)
        all_enc_test = scaler.transform(all_enc_test)
        print('> evaluate')
        evaluate(all_enc_test, labels_test)

    if args.esvm:
        print('> esvm computation')
        # TODO: for a real setup you actually want to cross-validate C
    #        all_enc_test = esvm(all_enc_test, all_enc_train, args.C)
        all_enc_test = runF('encs_test_esvm.pkl.gz', args.overwrite,
                            args.tmp_folder, 
                            esvm, all_enc_test, all_enc_train, args.C)
        print('> evaluate')
        evaluate(all_enc_test, labels_test)

