import xml.etree.ElementTree as ET 
import sys
import os


#input xml files
in_xml_path = sys.argv[1]

#output xml files
op_xml_path = sys.argv[2]

for xml_file in os.listdir(in_xml_path):

    #for a single xml file

    #parse this xml file
    tree = ET.parse(os.path.join(in_xml_path, xml_file))

    #get root of the tree
    root = tree.getroot()

    #first find all objects then manipulate otherwise concurrent modification may lead to inconsistencies
    for child in root.findall('object'):
        #if child is not an object
        if child[0].text.strip() == 'isolated':
            #element is an isolated math
            root.remove(child)
        
    tree.write(os.path.join(op_xml_path, xml_file))
    