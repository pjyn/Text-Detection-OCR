import cv2
import numpy as np
import pytesseract
import os

#E:\Tesseract-OCR

per = 25
pixelThreshold = 500

roi = [[(176, 312), (292, 328), 'text', 'Name'],
        [(180, 328), (296, 350), 'text', 'father\'s_name'],
       [(110, 356), (214, 374), 'text', 'Phone'],
       [(330, 356), (454, 370), 'text', 'Email'],
       [(118, 382), (222, 402), 'text', 'bank'],
       [(402, 374), (506, 394), 'text', 'ifsccode']]



pytesseract.pytesseract.tesseract_cmd = 'E:\\Tesseract-OCR\\tesseract.exe'

imgQ = cv2.imread('UserForms/real_testing.jpg')
h,w,c = imgQ.shape
# imgQ = cv2.resize(imgQ,(w//5,h//5))

orb = cv2.ORB_create(1000)
kp1, des1 = orb.detectAndCompute(imgQ,None)        #keypoints, descriptive
# impKp1 = cv2.drawKeypoints(imgQ,kp1,None)

path = 'UserForms'
myPicList = os.listdir(path)
print(myPicList)
for j,y in enumerate(myPicList):
    img = cv2.imread(path +"/"+y)
    # img = cv2.resize(img, (w // 5, h // 5))
    # cv2.imshow(y, img)
    kp2, des2 = orb.detectAndCompute(imgQ,None)
    bf = cv2.BFMatcher(cv2.NORM_HAMMING)
    matches = bf.match(des2,des1)
    matches.sort(key=lambda x: x.distance)   #lower the distance, the better the match is
    good = matches[:int(len(matches)*(per/100))]
    imgMatch = cv2.drawMatches(img,kp2,imgQ,kp1,good[:100],None,flags=2)

    # cv2.imshow(y, imgMatch)

    srcPoints = np.float32([kp2[m.queryIdx].pt for m in good]).reshape(-1,1,2)
    dstPoints = np.float32([kp1[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)

    M, _ = cv2.findHomography(srcPoints,dstPoints,cv2.RANSAC,5.0)
    imgScan = cv2.warpPerspective(img,M,(w,h))

    # cv2.imshow(y, imgScan)
    imgShow = imgScan.copy()
    imgMask = np.zeros_like(imgShow)

    myData = []

    print(f' ##############  Extracting Data from Form {j}  #############')

    for x,r in enumerate(roi):

        cv2.rectangle(imgMask, ((r[0][0]),r[0][1]),((r[1][0]),r[1][1]),(0,255,0),cv2.FILLED)
        imgShow = cv2.addWeighted(imgShow,0.99,imgMask,0.1,0)

        imgCrop = imgScan[r[0][1]:r[1][1], r[0][0]:r[1][0]]
        # cv2.imshow(str(x), imgCrop)

        if r[2] == 'text':
            print(f'{r[3]} :{pytesseract.image_to_string(imgCrop)}')
            myData.append(pytesseract.image_to_string(imgCrop))
        if r[2] == 'box':
            imgGray = cv2.cvtColor(imgCrop,cv2.COLOR_BGR2GRAY)
            imgThresh = cv2.threshold(imgGray,170,255, cv2.THRESH_BINARY_INV)[1]
            totalPixels = cv2.countNonZero(imgThresh)
            print(totalPixels)
            if totalPixels>pixelThreshold: totalPixels =1;
            else: totalPixels=0
            print(f'{r[3]} :{totalPixels}')
            myData.append(totalPixels)

        cv2.putText(imgShow,str(myData[x]),(r[0][0],r[0][1]),   # writes on output
                    cv2.FONT_HERSHEY_PLAIN,1.5,(0,0,255),2)

    with open('DataOutput.csv','a+') as f:
        for data in myData:
            f.write((str(data)+','))
        f.write('\n')

    # imgShow = cv2.resize(imgShow, (w // 3, h // 3))
    print(myData)
    cv2.imshow(y+"2", imgShow)


# cv2.imshow("KeyPointsQuery",impKp1)
cv2.imshow("Output",imgQ)
cv2.waitKey(0)