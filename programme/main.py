from myUtils import *

def main():
    imgList = ['imgs/diamond2.png', 'imgs/Dugong.jpg']

    activity_one(imgList)

    cv.waitKey(0)

def activity_one(imgList):
    diamond_img = cv.imread(imgList[0])
    dugong_img = cv.imread(imgList[1])

    diamond_img_copy = diamond_img.copy()
    dugong_img_copy = dugong_img.copy()

    scaled_diamonds = []
    scaled_dugongs = []

    rotated_diamonds = [rotate_image(diamond_img_copy, angle) for angle in range(15,360,15)]

    #for angle in(range(15, 360, 15)):
        #rotated_diamonds.append(rotate_image(diamond_img_copy, angle))

    show_img_ls(rotated_diamonds)

if __name__ == '__main__':
    main()
