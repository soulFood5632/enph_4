import cv2 as cv2
import numpy as np

class ImageComparer:

    distance_thresh = 0.8
    number_of_matches_neeeded = 12

    def __init__(self, image) -> None:
        self.image = image 
        self.grayscale = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        self.sift = cv2.SIFT_create()
        self.kp_image, self.desc_image = self.__getSetPoints()
        
    def __getSetPoints(self):
        # collects the keypoints of the image. You can add a mask by deleting the None property. 
        return self.sift.detectAndCompute(self.grayscale, None)
        
    
    def matchFrame(self, frame):
        grayframe = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        kp_grayframe, desc_grayframe = self.sift.detectAndCompute(frame, None)
        frame_img = cv2.drawKeypoints(grayframe, kp_grayframe, grayframe, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

        index_params = dict(algorithm=0, trees=5)
        search_params = dict()
        flann = cv2.FlannBasedMatcher(index_params, search_params)

        matches = flann.knnMatch(self.desc_image, desc_grayframe, k=2)

        success_matches = []
        for m, n in matches:
            # the lower the distance the better the match. Play around threshold value to ensure that matches are occuring.
            if m.distance < self.distance_thresh * n.distance:
                success_matches.append(m)
        if len(success_matches) > self.number_of_matches_neeeded:
            query_points = np.float32([self.kp_image[m.queryIdx].pt for m in success_matches]).reshape(-1, 1, 2)
            train_points = np.float32([kp_grayframe[m.trainIdx].pt for m in success_matches]).reshape(-1, 1, 2)
            matrix, mask = cv2.findHomography(query_points, train_points, cv2.RANSAC, 5.0)
            # what the hell does this do. 
            matches_mask = mask.ravel().tolist()
            h, w = self.image.shape[:2]
            pts = np.float32([[0, 0], [0, h], [w, h], [w, 0]]).reshape(-1, 1, 2)
            dst = cv2.perspectiveTransform(pts, matrix)
            homography = cv2.polylines(frame, [np.int32(dst)], True, (0, 255, 0), 3)
            return homography
        
        return frame



        matched_img = cv2.drawMatches(self.grayscale, self.kp_image, grayframe, kp_grayframe, success_matches, grayframe)

        return matched_img

        # cv2.imshow('key pointed image', matched_img)

    def printKeyPoints(self) :
        self.image = cv2.drawKeypoints(self.grayscale, self.kp_image, self.image, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

        cv2.imshow('key pointed image', self.image)
        cv2.waitKey(0)
        




    
        

# img = cv2.imread("000_image.jpg")

# image_comparer = ImageComparer(img)

# # image_comparer.printKeyPoints()

# cap = cv2.VideoCapture(0)

# while True:
#     _, frame = cap.read()

#     newImg = image_comparer.matchFrame(frame)
#     cv2.imshow("updated Image", newImg)
#     key = cv2.waitKey(1)

#     if key == 's':
#         break


# cv2.destroyAllWindows()




# while True:
#     _, frame = cap.read()
#     cv2.imshow("Image", img)
#     cv2.imshow("Frame", frame)

#     key = cv2.waitKey(1)

#     if key == 's':
#         break

# cap.release()
# cv2.destroyAllWindows()

