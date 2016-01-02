import skimage.io as skio
import skimage.color as skic
import skimage.transform as skit
import numpy as np
import pylab as plt
import sklearn.cluster as sklc
from skimage.util import img_as_float
from skimage.data import astronaut
import skimage.measure as skim
from scipy import ndimage as ndi 

from skimage.segmentation import felzenszwalb, slic, quickshift
from skimage.segmentation import mark_boundaries
import time
import os


def load_image(path):
    return img_as_float(skio.imread(path))

def apply_color_transform(img):
    return skic.rgb2lab(img)

def inverse_color_transform(img):
    return skic.lab2rgb(img)

def segment_image(img, n_segments=500, compactness=20, sigma=2):    
    segments_slic = slic(img, n_segments=n_segments, 
                              compactness=compactness,
                              sigma=sigma)
    
    # break disconnected regions
    segments = np.zeros_like(segments_slic)
    cur_seg_no = 0
    
    for i in range(np.max(segments_slic.ravel())):
        sep_segments = skim.label(segments_slic==i, background=0)
        for k in range(np.max(sep_segments.ravel()) + 1):
            segments[sep_segments == k] = cur_seg_no
            cur_seg_no += 1
    
    return segments
    
def plot_segments(img, segments):
    plt.imshow(mark_boundaries(img, segments, color=[0.8, 0.8, 0.8]))
    plt.title("SLIC")
    
    
def get_mean_of_segments(img, segments):
    ns = np.max(segments.ravel())
    means = {}
    
    for s in range(ns + 1):
        segment_pixels = img[segments == s]
        means[s] = np.mean(segment_pixels, axis=0)
        
    return means


def convert_mean_colors_to_lab(segment_means):    
    mean_colors_rgb = np.asarray([segment_means[x] for x in sorted(segment_means)])
    # making sure tmp is rectangular because otherwise skic conversions dont work..
    tmp = np.concatenate([mean_colors_rgb.reshape(-1, 1, 3),
                          mean_colors_rgb.reshape(-1, 1, 3)],
                        axis=1)
    
    mean_colors_lab = skic.rgb2lab(tmp)[:, 0, :].reshape(-1, 3)
    
    return mean_colors_lab
 

def get_dominant_colors(segment_means, n_colors=10):
    mean_colors_lab = convert_mean_colors_to_lab(segment_means)
    kmeans = sklc.KMeans(n_clusters=n_colors)
    kmeans.fit(mean_colors_lab)
    
    cc = kmeans.cluster_centers_
    
    tmp = np.concatenate([cc.reshape(-1, 1, 3),
                          cc.reshape(-1, 1, 3)],
                            axis=1)
    cc_rgb = skic.lab2rgb(tmp)[:, 0, :].reshape(-1, 3)
    
    cc_lab_2_rgb = {tuple(cc[i]): cc_rgb[i] for i in range(n_colors)}
    
    return kmeans, cc_lab_2_rgb
    
    
def segments_colors_to_dominant_colors(segment_means, n_colors=10):
    kmeans, cc_lab_2_rgb = get_dominant_colors(segment_means, n_colors)    
    cc = kmeans.cluster_centers_
    mean_colors_lab = convert_mean_colors_to_lab(segment_means)
    
    print len(segment_means)
    
    new_colors = {i: cc_lab_2_rgb[tuple(cc[kmeans.predict(c)][0])] for i, c in enumerate(mean_colors_lab)}
    
    return new_colors


def apply_values_to_segments(img, segments, segment_colors):
    new_img = np.zeros_like(img)
    colors = sorted(set(tuple(x) for x in segment_colors.values()))
    mapping = []
    
    for s, rgb in segment_colors.items():
        new_img[segments == s] = rgb        
        mapping += [colors.index(tuple(rgb))]
        
    return new_img, mapping

def get_segment_number_pos(segments, idx, bbs):
    x1, y1, x2, y2 = bbs[idx]
    distance = ndi.distance_transform_edt(segments[x1:x2, y1:y2] == idx)
    pos = np.unravel_index(np.argmax(distance.ravel()), distance.shape)
    pos += np.asarray([x1, y1])
    return pos

def get_segment_centroids(segments):
    # +1 because some segments will be 0 which is treated as background by region props
    # https://github.com/scikit-image/scikit-image/issues/941
    rps = skim.regionprops(segments + 1)
    bbs = [rp['bbox'] for rp in rps]
    
    c = [get_segment_number_pos(segments, idx, bbs) 
                                for idx in range(np.max(segments.ravel()))]

    return c

def plot_segment_numbers(img, segments, mapping):
    centroids = get_segment_centroids(segments)
    
    ax = plt.gca()
    
    for i, c in enumerate(centroids):
        color = mapping[i]
        ax.text(c[1], c[0], "{}".format(color), style='italic', size=8, alpha=0.5)
    

def calculate_mapping(img, segments, n_colors):
    means = get_mean_of_segments(img, segments)
    dom_colors = segments_colors_to_dominant_colors(means, n_colors)
    new_img, mapping = apply_values_to_segments(img, segments, dom_colors)
    
    return new_img, mapping, dom_colors, means
    
def plot_print_out(img, segments, mapping):
    plot_segments(np.ones_like(img), segments)
    plot_segment_numbers(img, segments, mapping)

def image_to_color_in(img, n_segments=500, compactness=20, 
                           sigma=2, n_colors=30,
                           folder_prefix=None):
    if folder_prefix is None:
        folder_prefix = "."

    # force trailing '/'
    folder_prefix = os.path.join(folder_prefix, '')

    segments = segment_image(img)
    new_img, mapping, dom_colors, means = calculate_mapping(img, 
                                                            segments,
                                                            n_colors)
    fig = plt.figure(figsize=(30,20))
    plot_print_out(img, segments, mapping)

    plotfile_segments = os.path.join(folder_prefix,
                                     'static', str(time.time()) + '_segments.png')
    plt.savefig(plotfile_segments)


    fig = plt.figure(figsize=(30,20))
    plt.imshow(new_img)

    plotfile_model = os.path.join(folder_prefix,
                                  'static', str(time.time()) + '_model.png')
    plt.savefig(plotfile_model)
    return plotfile_segments[len(folder_prefix):], plotfile_model[len(folder_prefix):]
