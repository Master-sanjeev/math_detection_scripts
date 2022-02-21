from posix import POSIX_FADV_WILLNEED
import sys
import os
from os import listdir
from os.path import isfile, join
import time
from PyPDF2 import pdf

start = time.time()

print('Running detection for : ', sys.argv[1])


# python3 automate.py /home/shivansh/mtp/pdf2charinfo-samples/samples/xyz.pdf
###########################################################################################################################
######################### command line args : pdf_path ####################################################################
###########################################################################################################################
# pdf_path = "/home/shivansh/mtp/pdf2charinfo-samples/samples/constructed/sanjeev/sanjeev3/dummy_maths_equtions-merged.pdf"
pdf_path = sys.argv[1]
pdf_root_dir = "/".join(pdf_path.split('/')[:-1])
images_path = pdf_root_dir + "/images_from_pdf/"
# output_path = pdf_root_dir + "/outputs/"

os.system("rm -rf " + images_path)

print('pdf root dir : ', pdf_root_dir)

cmdToRun = "python3 create_images_from_pdf.py " + pdf_root_dir + " " + images_path
print('Creating images from pdf : ', cmdToRun)

os.system(cmdToRun)



# imagefiles = [f for f in listdir(pdf_root_dir + "/images_from_pdf/") if isfile(join(pdf_root_dir + "/images_from_pdf/", f))]


# os.system("rm -rf " + output_path)
# os.system("mkdir " + output_path)

print("Running detection .......")

os.system('python3 detect_math.py {}'.format(images_path))

# for img in imagefiles:
#     # cmdToRun = "./darknet/darknet detector test darknet/data/multiple_images.data darknet/cfg/test.cfg darknet/backup/research_and_ncert.weights " + pdf_root_dir + "/images_from_pdf/" + img + " -thresh 0.3 -save_labels -dont_show"
#     cmdToRun = ""
#     cmdToRun += "cd darknet/ && "
#     cmdToRun += "./darknet detector test data/multiple_images.data cfg/test.cfg backup/research_and_ncert.weights " + pdf_root_dir + "/images_from_pdf/" + img + " -thresh 0.3 -save_labels -dont_show"
#     cmdToRun += " && mv predictions.jpg " + pdf_root_dir + "/outputs/" + img.split(".")[0] + ".jpg"
#     os.system(cmdToRun)

# os.system("rm -rf " + pdf_root_dir + "/aidetect/")
# os.system("mkdir " + pdf_root_dir + "/aidetect/")
# os.system("cp " + pdf_root_dir + "/images_from_pdf/*.txt " + pdf_root_dir + "/aidetect")

os.system("python3 convertTojson.py " + pdf_root_dir + "/aidetect/")

print('\n\n\n ************Total time taken : ', time.time()-start, ' seconds *****************')