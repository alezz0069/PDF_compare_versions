# PDF_compare_versions
Find the changes between versions of two PDFs. 
Reads the 2 PDFs, converts them into images, finds the similarity percentage between them, and then produces an image with the changes highlighted in green. Also makes a GIF image showing the differences inside a bounding box.

## Installation

```bash
pip install -r requirements.txt
pip install pymupdf  # If not already installed
