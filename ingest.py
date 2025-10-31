# ingest.py
import json
import re
from pathlib import Path
from typing import List
import pdfplumber  # <--- Import pdfplumber
import logging   # <--- Import logging module

# --- NEW: Suppress pdfminer warnings ---
# These warnings are common, non-fatal, but flood the console.
logging.getLogger("pdfminer").setLevel(logging.CRITICAL)
# --- END NEW ---

DATA_DIR = Path("data")
OUT_FILE = Path("chunks.json")

def convert_table_to_markdown(table: List[List[str]]) -> str:
    """
    Converts a table (list of lists) extracted by pdfplumber into a
    clean Markdown string.
    """
    # Remove None values and strip whitespace
    clean_table = []
    for row in table:
        clean_row = [str(cell).strip() if cell is not None else "" for cell in row]
        # Don't add completely empty rows
        if any(clean_row):
            clean_table.append(clean_row)

    if not clean_table:
        return ""

    # Build the Markdown table string
    markdown = "\n\n"  # Start with newlines to separate from text
    
    # Header row
    header = clean_table[0]
    markdown += "| " + " | ".join(header) + " |\n"
    
    # Separator row
    markdown += "| " + " | ".join(["---"] * len(header)) + " |\n"
    
    # Data rows
    for row in clean_table[1:]:
        # Ensure row has the same number of columns as header
        if len(row) == len(header):
            markdown += "| " + " | ".join(row) + " |\n"
            
    markdown += "\n"  # Add newline at the end
    return markdown

def extract_text_from_pdf(path: Path) -> str:
    """
    Extracts text and tables from a PDF using pdfplumber.
    Tables are converted to Markdown format.
    """
    full_text = ""
    with pdfplumber.open(path) as pdf:
        for i, page in enumerate(pdf.pages):
            
            # --- Table Extraction ---
            # Extract tables from the page
            tables = page.extract_tables()
            
            # Convert all found tables to Markdown
            for table in tables:
                full_text += convert_table_to_markdown(table)

            # --- Text Extraction ---
            # Extract text, *excluding* text that is part of the tables
            # This prevents duplicating the table data
            page_text = page.extract_text(x_tolerance=2, keep_blank_chars=True)
            
            if page_text:
                full_text += "\n" + page_text

    return full_text


def extract_text_from_txt(path: Path) -> str:
    return path.read_text(encoding="utf-8")

def clean_text(text: str) -> str:
    """
    Cleans the extracted text by removing PDF artifacts and fixing broken lines.
    (This function is largely unchanged, but will now also process markdown tables)
    """
    # Remove common PDF headers/footers (e.g., "--- PAGE 1 ---")
    # This might also be less necessary with pdfplumber, but good to keep.
    text = re.sub(r"--- PAGE \d+ ---", "", text)
    
    # Remove page numbers that might be on their own lines
    text = re.sub(r"^\d+\s*$", "", text, flags=re.MULTILINE)
    
    # Join lines that are not new paragraphs OR markdown table lines.
    # We update the regex to NOT join lines that start with '|' (markdown table)
    text = re.sub(r"\n(?!(\n|\|))", " ", text)
    
    # Optional: Fix hyphenated words broken across lines
    text = re.sub(r"(\w+)-\s*\n\s*(\w+)", r"\1\2", text)
    
    # Consolidate multiple spaces/newlines into one
    text = re.sub(r"(\n\s*){3,}", "\n\n", text)  # Max 2 newlines
    text = re.sub(r"[ \t]+", " ", text)       # Consolidate spaces
    
    return text.strip()


def chunk_text(text: str, chunk_size: int = 1500, chunk_overlap: int = 200) -> List[str]:
    """
    Splits text into semantic chunks using a recursive approach.
    (This function is unchanged. Its logic will work well with the new text.)
    """
    
    if chunk_overlap >= chunk_size:
        raise ValueError("Chunk overlap must be smaller than chunk size.")

    separators = ["\n\n", "\n", ". ", " ", ""]
    text_splits = [text]
    final_chunks = []
    
    for sep in separators:
        new_text_splits = []
        for split in text_splits:
            if len(split) <= chunk_size:
                new_text_splits.append(split)
            else:
                sub_splits = filter(None, split.split(sep))
                new_text_splits.extend(sub_splits)
        text_splits = new_text_splits
        
        if sep == "":
            merged_chunks = []
            current_chunk = ""
            for s in text_splits:
                if len(current_chunk) + len(s) + (1 if sep else 0) > chunk_size:
                    if current_chunk:
                        merged_chunks.append(current_chunk)
                    if len(s) > chunk_size:
                        merged_chunks.append(s[:chunk_size])
                        current_chunk = s[chunk_size:]
                    else:
                        current_chunk = s
                else:
                    current_chunk += (s + sep)
            if current_chunk:
                merged_chunks.append(current_chunk)

            for i in range(len(merged_chunks)):
                if i > 0:
                    overlap = merged_chunks[i-1][-chunk_overlap:]
                    chunk_with_overlap = overlap + merged_chunks[i]
                else:
                    chunk_with_overlap = merged_chunks[i]
                final_chunks.append(chunk_with_overlap.strip())
            break
            
    return [chunk for chunk in final_chunks if chunk]


def main():
    DATA_DIR.mkdir(exist_ok=True)
    all_chunks = []
    
    # --- NEW: Counters ---
    pdf_count = 0
    txt_count = 0
    # --- END NEW ---
    
    for p in sorted(DATA_DIR.iterdir()):
        if p.suffix.lower() == ".pdf":
            # --- UPDATED CALL ---
            print(f"Extracting tables and text from {p.name}...")
            raw_text = extract_text_from_pdf(p)
            pdf_count += 1  # <--- Increment PDF counter
        elif p.suffix.lower() in [".txt", ".md"]:
            print(f"Extracting text from {p.name}...")
            raw_text = extract_text_from_txt(p)
            txt_count += 1  # <--- Increment Txt counter
        else:
            print(f"Skipping unsupported file: {p.name}")
            continue
        
        # --- NEW STEP ---
        # Clean the text before chunking
        cleaned_text = clean_text(raw_text)
        
        if not cleaned_text.strip():
            print(f"Skipping {p.name} as it is empty after cleaning.")
            continue
        
        # --- UPDATED STEP ---
        # Use the new semantic chunker
        print(f"Chunking {p.name}...")
        chunks = chunk_text(cleaned_text, chunk_size=1500, chunk_overlap=200)
        
        for i, ch in enumerate(chunks):
            all_chunks.append({
                "id": f"{p.name}__chunk_{i}",
                "text": ch,
                "meta": {"source": p.name, "chunk_index": i}
            })

    if not all_chunks:
        print("\nNo documents found in data/ or content was empty.")
        return

    OUT_FILE.write_text(json.dumps(all_chunks, ensure_ascii=False, indent=2), encoding="utf-8")
    
    # --- NEW: Final Summary ---
    print("\n---------------------------------")
    print("Ingestion Complete.")
    print(f"Processed {pdf_count} PDF file(s).")
    print(f"Processed {txt_count} Text file(s).")
    print(f"Wrote {len(all_chunks)} chunks to {OUT_FILE}")
    print("---------------------------------")
    # --- END NEW ---

if __name__ == "__main__":
    main()