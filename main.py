import os



if __name__ == '__main__':
    os.system('python3 detect.py --view-img --source Videos/ --cfg cfg/yolov3-tiny3-1cls.cfg --weights weights/best_tiny3.pt')

