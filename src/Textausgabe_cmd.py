import pdfplumber

with pdfplumber.open(r"file") as pdf:
    for i, page in enumerate(pdf.pages):
        text = page.extract_text()
        # Steuerzeichen sichtbar machen
        visible = text.replace('\n', '↵\n').replace('\r', '↩').replace('\t', '→\t')
        print(f"--- Seite {i+1} ---")
        print(visible)