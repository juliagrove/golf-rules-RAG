import json
import os

import fitz  # opens and reads PDFs

from config import PDF_FOLDER, SCRAPED_FOLDER


def scrape_pdfs(output_folder=SCRAPED_FOLDER, pdf_dir=PDF_FOLDER):

    os.makedirs(output_folder, exist_ok=True)

    for file in os.listdir(pdf_dir):
        if file.endswith(".pdf"):
            file_name = os.path.splitext(file)[0]
            file_path = os.path.join(pdf_dir, file)

            pages = []
            doc = fitz.open(file_path)

            for i, page in enumerate(doc):
                text = page.get_text()
                pages.append({"page": i + 1, "text": text})

            output = os.path.join(output_folder, f"{file_name}.json")
            with open(output, "w") as f:
                json.dump(pages, f, indent=2)

            print(f"Successfully scraped: {file}")


if __name__ == "__main__":
    scrape_pdfs()
