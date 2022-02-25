import xml.etree.ElementTree as ET
import cv2
import sys
import os

#input image path
in_img_path = sys.argv[1]
in_xml_path = sys.argv[2]

#output image path
op_img_path = sys.argv[3]


for image in os.listdir(in_img_path):
    if len(image) > 4 and (image[-3:] == 'jpg' or image[-3:] == 'png'):
        img = cv2.imread(os.path.join(in_img_path, image))
        tree = ET.parse(os.path.join(in_xml_path, image[0:-3]+'xml'))
        for obj in tree.getroot().findall('object'):
            for box in obj.findall('bndbox'):
                cv2.rectangle(img, (int(box[0].text.strip()), int(box[1].text.strip())), (int(box[2].text.strip()), int(box[3].text.strip())), (255,0,0))
                # print(type(int(box[0].text)))

        cv2.imwrite(os.path.join(op_img_path, image), img)

