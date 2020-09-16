import json
import urllib.request as request
import os
import re
import copy
from tqdm import tqdm

class Download:
    def __init__(self, file_paths):
        self.file_paths = file_paths
        self.file_names = self.get_file_name()
        self.current_path = os.getcwd()
        self.memory = {}
        
    def read_json(self,file_path):
        with open(file_path, 'r') as json_file:
            json_data = json.load(json_file)
        
        return json_data
    
    def get_file_name(self):
        tmp = []
        for i,x in enumerate(self.file_paths):
            full_name = os.path.basename(x)
            name = re.findall('(.*)\.', full_name)
            tmp += name
        return tmp
    
    def get_bbox(self, image, anno):
        
        # find center
        # normalize
        bbox = []
        x_center = anno['bbox'][0] + anno['bbox'][2]/2
        y_center = anno['bbox'][1] + anno['bbox'][3]/2
        
        height, width = image['height'], image['width']
        
        bbox.append(round(x_center/width,6))
        bbox.append(round(y_center/height,6))
        bbox.append(round(anno['bbox'][2]/width,6))
        bbox.append(round(anno['bbox'][3]/height,6))
        
        return bbox # normalized list: [x,y,w,h]
        
    def __call__(self):
        # 이미지를 불러오고
        # 저장을 하면서 레이블도 같이 저장하고
        # coco1cls.txt에 경로를 저장해야함
        for i,x in enumerate(tqdm(self.file_paths)):
            json_data = self.read_json(x)
            
            for j,d in enumerate(json_data['images']):
                # download image
                route = self.current_path + '/coco/images/{0}/{1}'.format(self.file_names[i],d['file_name'])
                try: 
                    request.urlretrieve(d['coco_url'], route)
                except:
                    try:
                        request.urlretrieve(d['flickr_url'], route)
                    except:
                        continue
                        
                # write coco1cls.txt
                f = open(self.current_path+'/data/coco1cls.txt','a')
                f.write(route)
                f.close()
                        
                # save label
                for k,a in enumerate(json_data['annotations']):
                    try:
                        if self.memory[k] == 1:
                            continue
                    except:
                        if a['image_id'] == d['id']:
                            f = open('./coco/labels/{0}/{1}'.format(self.file_names[i], d['file_name'][:-4]+'.txt'), 'a')
                            bbox = self.get_bbox(d,a)
                            line = '0 {0} {1} {2} {3}\n'.format(bbox[0],bbox[1],bbox[2],bbox[3])
                            f.write(line)
                            f.close()
                        
                            self.memory[k] = 1




start = Download("파일명 리스트")
start()

# 끝나면 python3 train.py --data data/coco_1cls.data
