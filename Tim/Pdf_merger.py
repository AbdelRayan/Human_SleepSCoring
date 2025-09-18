from PyPDF2 import PdfMerger
import os

base_dir = "D:/Intercranial_sleep_data"
pdf_files = []
for subject in os.listdir(base_dir):
    for file in os.listdir(os.path.join(base_dir, subject, "Implantation_Scheme")):
        if '.pdf' in file:
            pdf_files.append(os.path.join(base_dir, subject, "Implantation_Scheme", file))
print(pdf_files)

merger = PdfMerger()

for pdf in pdf_files:
    merger.append(pdf)

merger.write("merged_output.pdf")
merger.close()