import numpy as np
from skimage import img_as_float
from skimage.io import imread
from skimage.measure import compare_psnr
from sklearn.cluster import KMeans

image = imread('C:\\Users\\AND\\Desktop\\MachineLearningAtCourcera\\hw\\week6task1\\parrot1.jpg')
image = img_as_float(image)

shape1 = image.shape[0]
shape2 = image.shape[1]
shape3 = image.shape[2]

pixel_count = shape1 * shape2

image = image.reshape((pixel_count, shape3))

for n_clusters in [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20]:
    cls = KMeans(init='k-means++', random_state=241, n_clusters=n_clusters, n_jobs=4)
    predicted = cls.fit_predict(image)

    t_image_mean = np.copy(image)
    t_image_median = np.copy(image)
    print("...0 %")
    for cl in np.arange(n_clusters):
        cluster = np.asarray([a for a in map(lambda a: True if a == cl else False, predicted)])
        cluster_mean = np.mean(image[cluster], axis=0)
        cluster_median = np.median(image[cluster], axis=0)
        t_image_mean[cluster] = cluster_mean
        t_image_median[cluster] = cluster_median
        print("..." + str(int((cl + 1) * 100 / n_clusters)) + " %")

    psnr_mean = compare_psnr(image, t_image_mean)
    psnr_median = compare_psnr(image, t_image_median)

    print('..for n_clusters = ' + str(n_clusters) + " psnr on mean = " + str(psnr_mean) + ", psnr on median = "
          + str(psnr_median))