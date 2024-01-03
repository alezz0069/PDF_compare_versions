# -*- coding: utf-8 -*-
"""Find_differences.py

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1125s9nnedUOes5ZTVuipwNR4-hZJs23-
"""

import fitz  # PyMuPDF
from PIL import Image, ImageDraw, ImageSequence
import csv
import cv2
import numpy as np
import os
import pandas as pd
from skimage.metrics import structural_similarity


def convert_pdfs_to_images(pdf_paths, output_path):
    for idx, pdf_path in enumerate(pdf_paths, start=1):
        pdf_document = fitz.open(pdf_path)

        for page_number in range(pdf_document.page_count):
            page = pdf_document[page_number]
            pixmap = page.get_pixmap()
            image = Image.frombytes("RGB", [pixmap.width, pixmap.height], pixmap.samples)
            image.save(f"{output_path}/im{idx}_page_{page_number + 1}.png")

        pdf_document.close()


pdf_file_paths = ['/content/IG-10309-DA-0000-09-LY-00002_A.pdf', '/content/IG-10309-DA-0000-09-LY-00002_B.pdf']
output_folder_path = '/content'
convert_pdfs_to_images(pdf_file_paths, output_folder_path)

def calculate_image_similarity(image1_path, image2_path):
    before = cv2.imread(image1_path)
    after = cv2.imread(image2_path)
    before_gray = cv2.cvtColor(before, cv2.COLOR_BGR2GRAY)
    after_gray = cv2.cvtColor(after, cv2.COLOR_BGR2GRAY)
    score, _ = structural_similarity(before_gray, after_gray, full=True)
    print(f"Image Similarity: {score * 100:.4f}%")


image1_path = '/content/im1_page_1.png'
image2_path = '/content/im2_page_1.png'
calculate_image_similarity(image1_path, image2_path)

def highlight_image_changes(image1_path, image2_path, output_image_path, csv_path):
    before = cv2.imread(image1_path)
    after = cv2.imread(image2_path)
    before_gray = cv2.cvtColor(before, cv2.COLOR_BGR2GRAY)
    after_gray = cv2.cvtColor(after, cv2.COLOR_BGR2GRAY)
    score, diff = structural_similarity(before_gray, after_gray, full=True)
    diff = (diff * 255).astype("uint8")
    diff_box = cv2.merge([diff, diff, diff])
    thresh = cv2.threshold(diff, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]

    contours = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]
    mask = np.zeros(before.shape, dtype='uint8')
    filled_after = after.copy()
    bounding_boxes = []

    for c in contours:
        area = cv2.contourArea(c)
        if area > 0:
            x, y, w, h = cv2.boundingRect(c)
            bounding_boxes.append([x, y, x + w, y + h])
            cv2.rectangle(before, (x, y), (x + w, y + h), (36, 255, 12), 2)
            cv2.rectangle(after, (x, y), (x + w, y + h), (36, 255, 12), 2)
            cv2.rectangle(diff_box, (x, y), (x + w, y + h), (36, 255, 12), 2)
            cv2.drawContours(mask, [c], 0, (255, 255, 255), -1)
            cv2.drawContours(filled_after, [c], 0, (0, 255, 0), -1)

    with open(csv_path, 'w', newline='') as csvfile:
        csv_writer = csv.writer(csvfile)
        csv_writer.writerow(['x1', 'y1', 'x2', 'y2'])
        csv_writer.writerows(bounding_boxes)

    cv2.imwrite(output_image_path, filled_after)


output_image_path = '/content/image_changes.png'
csv_path = '/content/bounding_boxes.csv'
highlight_image_changes(image1_path, image2_path, output_image_path, csv_path)

def convert_pdfs_to_gif_and_save_images(pdf_paths, output_path):
    gif_frames = []

    for idx, pdf_path in enumerate(pdf_paths, start=1):
        pdf_document = fitz.open(pdf_path)

        for page_number in range(pdf_document.page_count):
            page = pdf_document[page_number]
            pixmap = page.get_pixmap(matrix=fitz.Matrix(300/72, 300/72))
            image = Image.frombytes("RGB", [pixmap.width, pixmap.height], pixmap.samples)
            gif_frames.append(image)

        pdf_document.close()

    gif_frames[0].save(os.path.join(output_path, "output.gif"), save_all=True, append_images=gif_frames[1:], duration=250, loop=0)


convert_pdfs_to_gif_and_save_images(pdf_file_paths, output_folder_path)

def load_bounding_boxes(csv_path):
    return pd.read_csv(csv_path)

def draw_bounding_boxes(image, bounding_boxes, scale_factor):
    draw = ImageDraw.Draw(image)
    for _, box in bounding_boxes.iterrows():
        x1, y1, x2, y2 = box[['x1', 'y1', 'x2', 'y2']]
        scaled_x1 = int(x1 * scale_factor)
        scaled_y1 = int(y1 * scale_factor)
        scaled_x2 = int(x2 * scale_factor)
        scaled_y2 = int(y2 * scale_factor)
        draw.rectangle([scaled_x1, scaled_y1, scaled_x2, scaled_y2], outline="red", width=2)

def convert_pdfs_to_gif_and_save_images_with_bounding_boxes(pdf_paths, bounding_boxes_csv, output_path):
    gif_frames = []
    bounding_boxes = load_bounding_boxes(bounding_boxes_csv)

    for idx, pdf_path in enumerate(pdf_paths, start=1):
        pdf_document = fitz.open(pdf_path)

        for page_number in range(pdf_document.page_count):
            page = pdf_document[page_number]
            pixmap = page.get_pixmap(matrix=fitz.Matrix(300/72, 300/72))
            image = Image.frombytes("RGB", [pixmap.width, pixmap.height], pixmap.samples)
            draw_bounding_boxes(image, bounding_boxes, 300/72)
            gif_frames.append(image)

        pdf_document.close()

    gif_frames[0].save(os.path.join(output_path, "output_bboxes.gif"), save_all=True, append_images=gif_frames[1:], duration=250, loop=0)

bounding_boxes_csv = '/content/bounding_boxes.csv'
convert_pdfs_to_gif_and_save_images_with_bounding_boxes(pdf_file_paths, bounding_boxes_csv, output_folder_path)
print(f"Images created succesfully")