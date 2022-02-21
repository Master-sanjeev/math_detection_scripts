from pdf2image import convert_from_path
from PyPDF2 import PdfFileReader
import sys
import os
import tempfile
from PIL import Image

"""
    This module converts directory containing PDF files
    to images with 600 DPI and save them in given output directory
    Usage: python convert_pdf_to_image.py <PDF_files_dir> <output_img_dir>
"""

def create_images_from_pdfs(pdf_dir, output_dir):
    '''
    Render each PDF file as image in 600 DPI
    :param pdf_dir: Directory with PDF files
    :param output_dir: Output directory for storing image files for each PDF file
    :return: None
    '''
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    pdf_files = []
    for _, _, fileList in os.walk(pdf_dir):
        pdf_files.extend(fileList)
        break

    temp = []

    for t in pdf_files:
        if ".pdf" in t:
            temp.append(t)
    
    pdf_files = temp

    for pdf_file in pdf_files:
        pdf_name = pdf_file.split(".pdf")[0]
        output_path = os.path.join(output_dir, "")

        pdf = PdfFileReader(open(os.path.join(pdf_dir, pdf_file),'rb'))
        print(pdf.getNumPages())

        numofpages = pdf.getNumPages()

        i = 1
        pages = []
        while (i <= numofpages):
            
            page = convert_from_path(os.path.join(pdf_dir, pdf_file), 600, first_page=i, last_page=i, thread_count=100)[0]
            page.save(os.path.join(output_path, str(i-1) + ".png"), 'PNG')
            img = Image.open(os.path.join(output_path, str(i-1) + ".png")).convert('LA')
            img.save(os.path.join(output_path, str(i-1) + ".png"))
            print("Converted page " + str(i))
            i += 1

        


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Usage: python3 convert_pdf_to_image.py <PDF_files_dir> <output_dir>")
        #raise ValueError("Incorrect usage")
        exit(0)
    pdf_dir = sys.argv[1]
    print (pdf_dir)
    output_dir = sys.argv[2]
    print(output_dir)
    create_images_from_pdfs(pdf_dir, output_dir)
