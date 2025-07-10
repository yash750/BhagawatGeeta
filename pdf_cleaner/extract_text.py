import pdfplumber

PDF_PATH = "/Users/yeshwardhan/Documents/BhagawatGeeta/data/Bhagwad/Holy Geeta by Swami Chinmayana_removed.pdf"

def get_text_from_pdf(pdf_path):
    text = ""
    start = 0
    for i in range(18):
        end_points = [37, 172, 243, 328, 385, 490, 550, 608, 688, 773, 854, 901, 975, 1046, 1091, 1140, 1181, 1346]
        with pdfplumber.open(pdf_path) as pdf:
            for j in range(start, end_points[i]):
                page = pdf.pages[j]
                text += page.extract_text() + "\n"

        #create new file for each chapter
        with open(f"Chapter-{i+1}.txt", "w") as f:
            f.write(text)

        print(f"Chapter {i+1} extracted")

        start = end_points[i]
        text = ""

    return "Chapters Extracted Successfully"


if __name__ == "__main__":
    print(get_text_from_pdf(PDF_PATH))
