import fitz  # PyMuPDF
from PIL import Image, ImageDraw
import cv2
import numpy as np
import pandas as pd
import csv
from skimage.metrics import structural_similarity
from matplotlib.patches import Arc, Rectangle
import matplotlib.pyplot as plt

def convert_pdfs_to_images(pdf_paths, output_path):
    for idx, pdf_path in enumerate(pdf_paths, start=1):
        pdf_document = fitz.open(pdf_path)
        for page_number in range(pdf_document.page_count):
            page = pdf_document[page_number]
            pixmap = page.get_pixmap()
            image = Image.frombytes("RGB", [pixmap.width, pixmap.height], pixmap.samples)
            image.save(f"{output_path}/im{idx}_page_{page_number + 1}.png")
        pdf_document.close()

def calculate_image_similarity(image1_path, image2_path):
    before = cv2.imread(image1_path)
    after = cv2.imread(image2_path)
    before_gray = cv2.cvtColor(before, cv2.COLOR_BGR2GRAY)
    after_gray = cv2.cvtColor(after, cv2.COLOR_BGR2GRAY)
    score, _ = structural_similarity(before_gray, after_gray, full=True)
    print(f"Image Similarity: {score * 100:.4f}%")

def highlight_image_changes(image1_path, image2_path, output_image_path, csv_path):
    before = cv2.imread(image1_path)
    after = cv2.imread(image2_path)
    before_gray = cv2.cvtColor(before, cv2.COLOR_BGR2GRAY)
    after_gray = cv2.cvtColor(after, cv2.COLOR_BGR2GRAY)
    score, diff = structural_similarity(before_gray, after_gray, full=True)
    diff = (diff * 255).astype("uint8")
    thresh = cv2.threshold(diff, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
    contours = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]
    bounding_boxes = []

    scale_factor = 1.05  # Scale factor for enlarging the bounding boxes

    for c in contours:
        area = cv2.contourArea(c)
        if area > 0:
            x, y, w, h = cv2.boundingRect(c)
            # Center of the original bbox
            center_x, center_y = x + w / 2, y + h / 2
            # New dimensions
            new_w, new_h = w * scale_factor, h * scale_factor
            # New top-left corner
            new_x, new_y = center_x - new_w / 2, center_y - new_h / 2
            bounding_boxes.append([new_x, new_y, new_x + new_w, new_y + new_h])
            cv2.rectangle(after, (int(new_x), int(new_y)), (int(new_x + new_w), int(new_y + new_h)), (36, 255, 12), 2)

    with open(csv_path, 'w', newline='') as csvfile:
        csv_writer = csv.writer(csvfile)
        csv_writer.writerow(['x1', 'y1', 'x2', 'y2'])
        csv_writer.writerows(bounding_boxes)

    cv2.imwrite(output_image_path, after)


def process_images_for_clouds_and_combine(image_paths, csv_path, output_path):
    # Process only the second image for clouds
    img = Image.open(image_paths[1])
    df = pd.read_csv(csv_path)
    fig, ax = plt.subplots(figsize=(img.width / 100, img.height / 100), dpi=100)
    ax.imshow(img)
    ax.axis('off')

    for index, row in df.iterrows():
        points = [(row['x1'], row['y1']), (row['x1'], row['y2']), (row['x2'], row['y2']), (row['x2'], row['y1']), (row['x1'], row['y1'])]
        draw_revision_cloud(ax, points)

    plt.savefig(image_paths[1].replace('.png', '_cloud.png'), bbox_inches='tight', pad_inches=0)
    plt.close()

    # Combine the first image (without clouds) and the second image (with clouds)
    combine_images(image_paths[0], image_paths[1].replace('.png', '_cloud.png'), output_path)


def draw_revision_cloud(ax, points, arc_radius=1, arc_angle=45):
    """
    Draw a revision cloud with a semi-transparent bluish rectangle inside it.
    """
    for i in range(len(points)):
        start_point = points[i]
        end_point = points[(i + 1) % len(points)]
        line_angle = np.arctan2(end_point[1] - start_point[1], end_point[0] - start_point[0])
        distance = np.hypot(end_point[1] - start_point[1], end_point[0] - start_point[0])
        num_arcs = int(distance // (2 * arc_radius))

        for j in range(num_arcs):
            center_x = start_point[0] + (2 * arc_radius * j + arc_radius) * np.cos(line_angle)
            center_y = start_point[1] + (2 * arc_radius * j + arc_radius) * np.sin(line_angle)
            angle_offset = np.deg2rad(arc_angle / 2) if j % 2 == 0 else -np.deg2rad(arc_angle / 2)
            arc = Arc((center_x, center_y), 2 * arc_radius, 2 * arc_radius,
                      angle=np.rad2deg(line_angle) - 90,
                      theta1=np.rad2deg(angle_offset),
                      theta2=np.rad2deg(angle_offset) + 180,
                      color='blue')
            ax.add_patch(arc)

    # Draw a semi-transparent bluish rectangle inside the cloud
    x_min = min([x for x, y in points])
    x_max = max([x for x, y in points])
    y_min = min([y for x, y in points])
    y_max = max([y for x, y in points])
    rect = Rectangle((x_min, y_min), x_max - x_min, y_max - y_min,
                     color=(0, 0, 1, 0.2))  # Blue color with 20% transparency
    ax.add_patch(rect)

def combine_images(image1_path, image2_path, output_path):
    image1 = Image.open(image1_path)
    image2 = Image.open(image2_path)
    new_height = max(image1.height, image2.height)
    image1 = image1.resize((int(image1.width * new_height / image1.height), new_height), Image.ANTIALIAS)
    image2 = image2.resize((int(image2.width * new_height / image2.height), new_height), Image.ANTIALIAS)
    new_width = image1.width + image2.width + 50
    new_image = Image.new('RGB', (new_width, new_height), (255, 255, 255))
    new_image.paste(image1, (0, 0))
    new_image.paste(image2, (image1.width + 50, 0))
    new_image.save(output_path)

# Example usage
pdf_file_paths = ['/content/IG-10309-DA-0000-07-GA-00002_A.pdf', '/content/IG-10309-DA-0000-07-GA-00002_B.pdf']
output_folder_path = '/content'
convert_pdfs_to_images(pdf_file_paths, output_folder_path)
calculate_image_similarity('/content/im1_page_1.png', '/content/im2_page_1.png')
highlight_image_changes('/content/im1_page_1.png', '/content/im2_page_1.png', '/content/image_changes.png', '/content/bounding_boxes.csv')
process_images_for_clouds_and_combine(['/content/im1_page_1.png', '/content/im2_page_1.png'], '/content/bounding_boxes.csv', '/content/combined_image.png')
