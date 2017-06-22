import numpy as np
import sys,os
import pickle
import glob
from sklearn.cluster import MiniBatchKMeans
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.decomposition import TruncatedSVD
import help_functions
# sys.path.insert(0, './Feature_Extraction')
import feature_extraction

def main(argv):
    audiofolder = 'test_wav/'
    featuresfolder = 'test_mfcc'
    modelfolder = 'test_adsm/'
    tagpath = 'clip_tags'
    train_adsm(audiofolder, featuresfolder, modelfolder, k=300, svd=-1, tagpath=tagpath)
    return


def train_adsm(audiofolder, featuresfolder, modelfolder, k=300, svd=-1, tagpath='clip_tags.txt'):
    '''
    Train an Audio-based Distributional Semantic Model (ADSM), i.e.:
    Obtain tag representations by averaging the Bag-of-Audio-Words representations of audio clips
    :param audiofolder:  a folder containing audiofiles
    :param featuresfolder: 
    :param modelfolder: 
    :param K: 
    :param svd: 
    :param tagpath: 
    :return: 
    '''

    # Extract Acoustic Features
    # wavfiles = glob.glob(os.path.join(sourcefolder, '*.flac'))
    # if not os.path.exists(featuresfolder):

    print("Feature Extraction...")
    # feature_extraction.mfcc_features(sourcefolder=audiofolder, savefolder=featuresfolder, audio_extension='flac')
    extract_features_for_dir(audiofolder=audiofolder, featuresfolder=featuresfolder)

    # create audio-word dictionary
    featurepaths = glob.glob(os.path.join(featuresfolder, '*.*'))
    print('Training Audio-word Dictionary using {} files'.format(len(featurepaths)))
    aw_dict = compute_audio_word_dictionary(featurepaths, k)

    # save audio-word dictionary as a pickle file
    help_functions.mkdir(modelfolder)
    savepath = os.path.join(modelfolder, 'audio-words')
    help_functions.save_dict(python_dict=aw_dict, savepath=savepath)

    # compute audio clip representations (BoAW encodings)
    encodedsounds = clip_encoding_BoAW(featurepaths, k=k, aw_dict=aw_dict)

    # compute tag encodings & PMI weighting
    # tagpath format: clip_id tag1 tag2 tagn
    soundnames = [os.path.splitext(os.path.basename(f))[0] for f in featurepaths]
    pmi_weighted_tag_matrix = pmi(sounds=soundnames, encodedsounds=encodedsounds, K=k, positive_pmi=1,
                                   savefolder=modelfolder, tagpath=tagpath, fold=-1)
    # write as dm file the pmi transformed matrix
    help_functions.save_dict_as_txt(os.path.join(modelfolder, 'BoAW_pmi.dm'), pmi_weighted_tag_matrix)

    # Dimensionality Reduction PCA/SVD
    if svd > 0:
        truncated_svd = TruncatedSVD(n_components=svd)
        return truncated_svd.fit_transform(pmi_weighted_tag_matrix)
    return


def extract_features_for_dir(audiofolder, featuresfolder):
    audiopaths = glob.glob(audiofolder + '*.*')
    help_functions.mkdir(featuresfolder)
    for fpath in audiopaths:
        fid = os.path.splitext(os.path.basename(fpath))[0]
        features = feature_extraction.extract_mfccdd(fpath)
        np.save(os.path.join(featuresfolder, fid), features)
    return


def compute_audio_word_dictionary(filelist, k):
    '''
    Compute the dictionary of k audio-words. 
    This step is necessary for building Bag-of-Audio-Word (BoAW) representations.
    Here, the k audio-words are computed as the k centroids of minibatch k-means.
    :param filelist: a list of filepaths corresponding to extracted features
    :param k: the number of audio-words i.e. the parameter k of kmeans
    :return: a dictionary containing the audio-words and other useful information
    '''
    aw_dict = {}
    print("--- Computing the audio-word dictionary ---")
    print("Feature Matrix Construction...")
    features = construct_feature_matrix(filelist)
    print("Feature Matrix: {}".format(features.shape))
    print("Clustering...")
    batch_size = 10000
    kmeans = MiniBatchKMeans(n_clusters=k, init='k-means++', n_init=3, batch_size=batch_size, max_iter=1000,
                             verbose=True, tol=0.0001, random_state=None)
    kmeans.fit(features)
    aw_dict['audio-words'] = kmeans.cluster_centers_
    # TODO: extend code to support cluster priors & variances
    return aw_dict


def clip_encoding_BoAW(featurepaths, k, aw_dict):
    print("--- Computing clip representations (Bag-of-Audio-Words)")
    encodedsounds = np.empty((0, k), dtype=float)
    for fpath in featurepaths:
        features = np.load(fpath)
        sound_encoding = np.zeros((1, k))
        distances = euclidean_distances(features, aw_dict['audio-words'], Y_norm_squared=None, squared=False)
        for column in distances:
            index = np.where(column == min(column))[0][0]
            sound_encoding[0, index] += 1
        encodedsounds = np.concatenate([encodedsounds, sound_encoding], axis=0)
    return encodedsounds


def construct_feature_matrix(filelist):
    # parse all feature files and create a feature matrix
    print("Parsing features for {} files".format(len(filelist)))
    features = np.empty((0, 0))
    try:
        for i, fpath in enumerate(filelist):
            if not os.path.exists(fpath):
                print("[ERROR] Nonexistent feature path: {}".format(fpath))
            featurization = np.load(fpath)                              # load the feature matrix for one file
            if i==0: features = np.empty((0, featurization.shape[1]))   # set the number of columns appropriately
            features = np.concatenate([features, featurization])        # concatenate to the global feature matrix
    except:
        print("Could not parse all extracted features")
        sys.exit(1)
    return features


def pmi(sounds, encodedsounds, K, positive_pmi, savefolder, tagpath, fold):
    if positive_pmi: print("Applying PPMI Weighting...")
    else: print("Applying PMI Weighting...")

    # Create tag*audioword matrix
    tags, tagCounter = help_functions.cutoff_taglist(tagpath)
    soundidwords = help_functions.soundid_words(tagpath)
    encodedsoundsDict = {sounds[i]: encodedsounds[i] for i in range(0, len(sounds))}
    word_soundid = help_functions.word_soundid(tags, soundidwords)
    tagembedding = {}
    tag_audioword_cooccurence = {}
    useless_tags = []
    for tag in word_soundid:
        embedding = np.zeros(K)
        accurate_sounds = 0
        for sound in word_soundid[tag]:
            if sound in sounds:
                embedding = embedding + encodedsoundsDict[
                    sound]  # add the embeddings of all sounds associated with a tag
                accurate_sounds += 1
        if accurate_sounds > 0:
            tag_audioword_cooccurence[tag] = embedding
            tagembedding[tag] = embedding / float(accurate_sounds)  # take the grand average
        else:  # There is no valid audio sample for the specific tag
            print "Tag %s is not associated with a valid audio sample..." % tag
            # tagembedding[tag] = np.zeros(K)
            useless_tags.append(tag)

    for tag in useless_tags:
        del word_soundid[tag]
        del tagCounter[tag]

    # PPMI (Positive Pointwise Mutual Information)
    N = sum(tagCounter.values())

    N_c = sum(sum(tag_audioword_cooccurence.values()))
    N_t = sum(tagCounter.values())
    N_aw = sum(sum(encodedsounds))
    N1 = N_t * N_aw / float(N_c)
    audiowordCounter = [sum(row[column] for row in encodedsounds) for column in range(0, K)]
    pmi_transformed = {}
    for tag in word_soundid:
        pmi = []
        for j in range(0, K):
            # print tag,j
            C = tag_audioword_cooccurence[tag][j]
            if tagCounter[tag] * audiowordCounter[j] > 0 and C > 0:
                # p_c = tag_audioword_cooccurence[tag][j]/float(N_c)
                # p_t = tagCounter[tag] / float(N_t)
                # p_aw = audiowordCounter[j] / float(N_aw)
                # LPI = -1.0*np.log10( p_c / (p_t*p_aw)  )
                PMI = 1.0 * np.log10(C * N1 / float(tagCounter[tag] * audiowordCounter[j]))
                # PMI =  C*np.log10(C*N/float(tagCounter[tag]*audiowordCounter[j]))     # PMI
            else:
                PMI = 0.0  # Positive PMI

            if PMI > 0 and positive_pmi:
                pmi.append(PMI)
            elif positive_pmi:
                pmi.append(0.0)
            else:
                pmi.append(PMI)
        pmi_transformed[tag] = pmi

    ### Save Results ###
    # write as dm file the initial matrix
    help_functions.save_dict_as_txt(os.path.join(savefolder, 'BoAW.dm'), tagembedding)

    return pmi_transformed


if __name__ == '__main__':
    main(sys.argv)