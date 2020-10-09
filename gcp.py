import matplotlib.pyplot as plt
import cv2 as cv
import numpy as np

def find(img1,img2):
    MIN_MATCH_COUNT = 3
    # Initiate SIFT detector
    sift = cv.SIFT_create()
    # find the keypoints and descriptors with SIFT
    kp1, des1 = sift.detectAndCompute(img1,None)
    kp2, des2 = sift.detectAndCompute(img2,None)
    FLANN_INDEX_KDTREE = 1
    index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
    search_params = dict(checks = 50)
    flann = cv.FlannBasedMatcher(index_params, search_params)
    matches = flann.knnMatch(des1,des2,k=2)
    # store all the good matches as per Lowe's ratio test.
    good = []
    for m,n in matches:
        if m.distance < 0.5*n.distance:
            good.append(m)
    if len(good)>MIN_MATCH_COUNT:
        src_pts = np.float32([ kp1[m.queryIdx].pt for m in good ]).reshape(-1,1,2)
        dst_pts = np.float32([ kp2[m.trainIdx].pt for m in good ]).reshape(-1,1,2)
        M, mask = cv.findHomography(src_pts, dst_pts, cv.RANSAC, 5)
        print("M: ",M)
        print("mask: ",mask)
        matchesMask = mask.ravel().tolist()
        h,w = img1.shape
        pts = np.float32([ [0,0],[0,h-1],[w-1,h-1],[w-1,0] ]).reshape(-1,1,2)
        dst = cv.perspectiveTransform(pts,M)
        img2 = cv.polylines(img2,[np.int32(dst)],True,255,3, cv.LINE_AA)
    else:
        print( "Not enough matches are found - {}/{}".format(len(good), MIN_MATCH_COUNT) )
        matchesMask = None

    draw_params = dict(matchColor = (0,255,0), # draw matches in green color
                   singlePointColor = None,
                   matchesMask = matchesMask, # draw only inliers
                   flags = 2)
    img3 = cv.drawMatches(img1,kp1,img2,kp2,good,None,**draw_params)
    plt.imshow(img3, 'gray'),plt.show()

def main():

    gcp = cv.imread("gcp.jpg", cv.IMREAD_GRAYSCALE)
    img = cv.imread("T6_-_341.jpg")
    gray = cv.cvtColor(img,cv.COLOR_BGR2GRAY)

    find(gcp,gray)

    # sift = cv2.SIFT_create()
    # kp_image, desc_image = sift.detectAndCompute(gcp, None)

    # index_params = dict(algorithm=0, trees=5)
    # search_params = dict()
    # flann = cv2.FlannBasedMatcher(index_params, search_params)
    # kp_grayframe, desc_grayframe = sift.detectAndCompute(gray, None)
    # matches = flann.knnMatch(desc_image, desc_grayframe, k=2)
    # good_points = []
    # for m, n in matches:
    #     if m.distance < 0.6 * n.distance:
    #         good_points.append(m)

    # query_pts = np.float32([kp_image[m.queryIdx].pt for m in good_points]).reshape(-1, 1, 2)
    # train_pts = np.float32([kp_grayframe[m.trainIdx].pt for m in good_points]).reshape(-1, 1, 2)
    # print("query_pts: ", query_pts)
    # print("train_pts: ", train_pts)
    # matrix, mask = cv2.findHomography(query_pts, train_pts, cv2.RANSAC, 5.0)
    # matches_mask = mask.ravel().tolist()

    # h, w = gcp.shape
    # pts = np.float32([[0, 0], [0, h], [w, h], [w, 0]]).reshape(-1, 1, 2)
    # print("pts: ",pts)
    # dst = cv2.perspectiveTransform(pts, matrix)

    # homography = cv2.polylines(img, [np.int32(dst)], True, (255, 0, 0), 3)

    # cv2.imshow("Homography", homography)
    # cv2.waitKey(0)


    # gcp2 = cv2.drawKeypoints(gcp,kp_image,gcp,color=(255,0,0))

    #cv2.imshow("GCP",gcp2)
    #cv2.waitKey(0)
    
    #cv2.imshow("Gray",gray)
    #cv2.waitKey(0)

    #img3 = cv2.drawMatches(gcp,kp_image,gray,kp_grayframe,good_points,gray)
    #cv2.imshow("Matched",img3)
    #cv2.waitKey(0)

if __name__ == "__main__":
    main()