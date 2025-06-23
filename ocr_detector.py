import sys
import os
import fitz  # PyMuPDF
from PIL import Image
import easyocr
import warnings
import shutil
import re
import json

# Suppress PIL decompression bomb warning for large images
warnings.filterwarnings("ignore", "(Possible )?[Dd]ecompression bomb", Image.DecompressionBombWarning)


def parse_contract_text(text):
    """
    Parse the entire contract document into structured JSON format
    without relying on specific regex patterns.
    """
    parsed = {
        "full_text": text.strip(),
        "document_structure": {},
        "paragraphs": [],
        "sections": {},
        "metadata": {}
    }
    
    # Split text into lines and clean them
    lines = [line.strip() for line in text.split('\n') if line.strip()]
    
    current_section = None
    current_paragraph = []
    paragraph_index = 0
    
    for i, line in enumerate(lines):
        # Store each line with its position
        line_data = {
            "line_number": i + 1,
            "content": line,
            "length": len(line)
        }
        
        # Detect if this looks like a section header (starts with number and dot)
        section_match = re.match(r'^(\d+)\.\s*(.+)', line)
        if section_match:
            # Save previous paragraph if exists
            if current_paragraph:
                parsed["paragraphs"].append({
                    "paragraph_id": paragraph_index,
                    "section": current_section,
                    "content": " ".join(current_paragraph),
                    "line_count": len(current_paragraph)
                })
                paragraph_index += 1
                current_paragraph = []
            
            # Start new section
            section_num = section_match.group(1)
            section_title = section_match.group(2)
            current_section = f"section_{section_num}"
            
            parsed["sections"][current_section] = {
                "number": section_num,
                "title": section_title,
                "full_header": line,
                "content": [],
                "line_number": i + 1
            }
            
        else:
            # Regular content line
            if current_section:
                parsed["sections"][current_section]["content"].append(line)
            
            current_paragraph.append(line)
            
            # If we detect a potential header/title (all caps, short line, etc.)
            if line.isupper() and len(line) < 100:
                if "headers" not in parsed["document_structure"]:
                    parsed["document_structure"]["headers"] = []
                parsed["document_structure"]["headers"].append({
                    "line_number": i + 1,
                    "content": line,
                    "type": "potential_header"
                })
    
    # Save the last paragraph
    if current_paragraph:
        parsed["paragraphs"].append({
            "paragraph_id": paragraph_index,
            "section": current_section,
            "content": " ".join(current_paragraph),
            "line_count": len(current_paragraph)
        })
    
    # Add metadata
    parsed["metadata"] = {
        "total_lines": len(lines),
        "total_characters": len(text),
        "total_sections": len(parsed["sections"]),
        "total_paragraphs": len(parsed["paragraphs"]),
        "sections_found": list(parsed["sections"].keys())
    }
    
    # Convert section content lists to strings
    for section_key in parsed["sections"]:
        parsed["sections"][section_key]["content"] = "\n".join(parsed["sections"][section_key]["content"])
    
    return parsed


def convert_pdf_to_image(pdf_path):
    """
    Convert PDF pages to images and save them in the 'output' directory.
    """
    if not os.path.exists('output'):
        os.makedirs('output')
    
    doc = fitz.open(pdf_path)
    num_pages = len(doc)  # Save length before processing
    
    for i, page in enumerate(doc):
        # Reduce DPI to avoid memory issues with large images
        pix = page.get_pixmap(dpi=300)
        output_path = f'output/page_{i}.png'
        pix.save(output_path)
    
    doc.close()  # Close the document properly
    print(f'Converted {num_pages} pages to images at 200 DPI.')
    return num_pages


def run_easyocr(folder='output', lang='cs'):
    """
    Use EasyOCR for text extraction.
    """
    try:
        reader = easyocr.Reader([lang])
    except Exception as e:
        print(f"Error initializing EasyOCR: {e}")
        return []
    
    results = []
    for file in sorted(os.listdir(folder)):
        if file.endswith('.png'):
            print(f"Reading text from {file} using EasyOCR")
            try:
                result = reader.readtext(os.path.join(folder, file))
                results.append((file, result))            
            except Exception as e:
                print(f"Error processing {file}: {e}")
                results.append((file, []))
    return results


def extract_contract_sections(text):
    """
    Extract the three specific sections from the contract:
    1. Header (everything before "1.")
    2. Section "1. ÚVODNÍ USTANOVENÍ"
    3. Section "3. KUPNÍ CENA"
    """
    extracted_sections = {
        "header": "",
        "section_1_uvodni_ustanoveni": "",
        "section_3_kupni_cena": ""
    }
    
    # Split text into lines for easier processing
    lines = text.split('\n')
    full_text = ' '.join(lines)
    
    # Find the position of "1. ÚVODNÍ USTANOVENÍ"
    section_1_match = re.search(r'1\.\s*ÚVODNÍ\s+USTANOVENÍ', full_text, re.IGNORECASE)
    
    # Find the position of "3. KUPNÍ CENA"
    section_3_match = re.search(r'3\.\s*KUPNÍ\s+CENA', full_text, re.IGNORECASE)
    
    if section_1_match:
        # Header is everything before section 1
        header_end = section_1_match.start()
        extracted_sections["header"] = full_text[:header_end].strip()
        
        # Section 1 starts from the match
        section_1_start = section_1_match.start()
        
        # Find where section 1 ends (look for next numbered section like "2.")
        section_2_match = re.search(r'2\.\s+[A-ZÁČĎÉĚÍŇÓŘŠŤÚŮÝŽ]', full_text[section_1_start:], re.IGNORECASE)
        if section_2_match:
            section_1_end = section_1_start + section_2_match.start()
            extracted_sections["section_1_uvodni_ustanoveni"] = full_text[section_1_start:section_1_end].strip()
        else:
            # If no section 2 found, take everything until section 3 or end
            if section_3_match:
                section_1_end = section_3_match.start()
                extracted_sections["section_1_uvodni_ustanoveni"] = full_text[section_1_start:section_1_end].strip()
            else:
                extracted_sections["section_1_uvodni_ustanoveni"] = full_text[section_1_start:].strip()
    
    if section_3_match:
        # Section 3 starts from the match
        section_3_start = section_3_match.start()
        
        # Find where section 3 ends (look for next numbered section like "4.")
        section_4_match = re.search(r'4\.\s+[A-ZÁČĎÉĚÍŇÓŘŠŤÚŮÝŽ]', full_text[section_3_start:], re.IGNORECASE)
        if section_4_match:
            section_3_end = section_3_start + section_4_match.start()
            extracted_sections["section_3_kupni_cena"] = full_text[section_3_start:section_3_end].strip()
        else:
            # If no section 4 found, take everything from section 3 to end or until signatures
            # Look for signature area indicators
            signature_patterns = [
                r'Prodávající\s+Kupující',
                r'V\s+dne\s+V\s+dne',
                r'Stránka\s+\d+/\d+'
            ]
            
            section_3_end = len(full_text)
            for pattern in signature_patterns:
                sig_match = re.search(pattern, full_text[section_3_start:], re.IGNORECASE)
                if sig_match:
                    section_3_end = section_3_start + sig_match.start()
                    break
            
            extracted_sections["section_3_kupni_cena"] = full_text[section_3_start:section_3_end].strip()
    
    # Clean up the extracted sections
    for key in extracted_sections:
        # Remove excessive whitespace and normalize
        extracted_sections[key] = re.sub(r'\s+', ' ', extracted_sections[key]).strip()
    
    return extracted_sections


def main(file_path="smlouva.pdf"):
    """
    Main function using EasyOCR for text extraction.
    """
    print(f"\nStarting PDF processing: {file_path}")
    print("OCR Method: EasyOCR")
    
    # Run conversion to individual page images
    doc_size = convert_pdf_to_image(file_path)
      # Process each page separately (no merging)
    print(f"Processing {doc_size} pages individually...")
    all_ocr_results = []
    
    for i in range(doc_size):
        page_image = f'output/page_{i}.png'
        print(f"Processing page {i+1}/{doc_size}")  # Fix: show 1-based indexing for user
        
        # Create a temporary folder for this page
        page_folder = f'output/temp_page_{i}'
        if not os.path.exists(page_folder):
            os.makedirs(page_folder)
          # Copy page image to temp folder
        temp_image_path = os.path.join(page_folder, f'page_{i}.png')
        shutil.copy(page_image, temp_image_path)
        
        # Run EasyOCR on this single page
        page_ocr_results = run_easyocr(page_folder)
        
        # Add page number to results
        for filename, text_results in page_ocr_results:
            all_ocr_results.append((f"page_{i}.png", text_results))
        
        # Clean up temp folder
        shutil.rmtree(page_folder)
        # Remove original page image
        os.remove(page_image)

    # Extract text from all OCR results and combine
    print("Combining text from all pages...")
    full_text = ""
    page_texts = {}
    
    for page_filename, text_results in all_ocr_results:
        page_text = ""
        for entry in text_results:
            page_text += entry[1] + " "
        
        # Store individual page text
        page_number = page_filename.split('_')[1].split('.')[0]
        page_texts[f"page_{page_number}"] = page_text.strip()
          # Add to full document text
        full_text += page_text + "\n"
    
    if not full_text.strip():
        print("Warning: No text was extracted from the document!")
        print("Try a different OCR method or check if the PDF contains readable text.")
        return
    
    # Parse the contract text to extract complete document structure
    print("Parsing document structure...")
    parsed_data = parse_contract_text(full_text)
    
    # Extract the three specific sections requested
    print("Extracting specific contract sections...")
    extracted_sections = extract_contract_sections(full_text)
    parsed_data["extracted_sections"] = extracted_sections
    
    # Add individual page texts to the parsed data
    parsed_data["page_texts"] = page_texts
    parsed_data["metadata"]["total_pages"] = doc_size
    
    # Print parsed results
    print("\nComplete Document Analysis:")
    print("=" * 50)
    
    # Show document metadata
    print(f"Document Metadata:")
    for key, value in parsed_data["metadata"].items():
        print(f"   {key}: {value}")
    
    # Show individual page text lengths
    print(f"\nIndividual Page Analysis:")
    for page_key, page_text in page_texts.items():
        print(f"   {page_key}: {len(page_text)} characters")
      # Show detected headers
    if "headers" in parsed_data["document_structure"] and parsed_data["document_structure"]["headers"]:
        print(f"\nDetected Headers:")
        for header in parsed_data["document_structure"]["headers"]:
            print(f"   Line {header['line_number']}: {header['content']}")
    else:
        print(f"\nNo uppercase headers detected")
    
    # Show all sections
    if parsed_data["sections"]:
        print(f"\nDocument Sections:")
        for section_key, section_data in parsed_data["sections"].items():
            print(f"\n{section_data['number']}️⃣ Section {section_data['number']}: {section_data['title']}")
            content_preview = section_data['content'][:200] if len(section_data['content']) > 200 else section_data['content']
            print(f"   Content: {content_preview}...")
    else:
        print(f"\nNo numbered sections found")
    
    # Show extracted specific sections
    print(f"\nExtracted Specific Sections:")
    for section_name, section_content in parsed_data["extracted_sections"].items():
        if section_content:
            content_preview = section_content[:150] if len(section_content) > 150 else section_content
            print(f"   📄 {section_name}: {len(section_content)} chars")
            print(f"      Preview: {content_preview}...")
        else:
            print(f"   ❌ {section_name}: Not found")
      # Show paragraphs summary
    if parsed_data["paragraphs"]:
        print(f"\nParagraphs Summary (first 5):")
        for para in parsed_data["paragraphs"][:5]:
            content_preview = para['content'][:100] if len(para['content']) > 100 else para['content']
            print(f"   Para {para['paragraph_id']}: {content_preview}...")
        if len(parsed_data["paragraphs"]) > 5:
            print(f"   ... and {len(parsed_data['paragraphs']) - 5} more paragraphs")
    else:
        print(f"\nNo paragraphs detected")
    
    # Save complete parsed data to JSON file
    output_file = 'output/parsed_contract.json'
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(parsed_data, f, ensure_ascii=False, indent=2)
    
    print(f"\nComplete document structure saved to: {output_file}")
    print(f"Full document text available in 'full_text' field")
    print(f"Individual page texts available in 'page_texts' field")
    print(f"Document structure available in 'document_structure' field")
    print(f"All sections available in 'sections' field")
    print(f"All paragraphs available in 'paragraphs' field")
    print(f"🎯 Extracted specific sections available in 'extracted_sections' field:")
    print(f"   - header: Contract header (before section 1)")
    print(f"   - section_1_uvodni_ustanoveni: Section 1. ÚVODNÍ USTANOVENÍ")
    print(f"   - section_3_kupni_cena: Section 3. KUPNÍ CENA")
    
    # Show some stats
    print(f"\nProcessing Summary:")
    print(f"   Pages processed: {doc_size}")
    print(f"   Total text characters: {len(full_text)}")
    print(f"   Sections found: {len(parsed_data['sections'])}")
    print(f"   Paragraphs found: {len(parsed_data['paragraphs'])}")
    
    # Show per-page breakdown
    print(f"\nPer-page breakdown:")
    for page_key, page_text in page_texts.items():
        print(f"   {page_key}: {len(page_text)} chars")


if __name__ == "__main__":
    # Get file path from command line argument
    file_path = sys.argv[1] if len(sys.argv) > 1 else "smlouva.pdf"
    
    print("EasyOCR PDF Parser")
    print("=" * 30)
    print("Usage: python read_from_pdf.py [pdf_file]")
    print("Uses EasyOCR for Czech contract text extraction")
    print()
    
    # Check if file exists
    if not os.path.exists(file_path):
        print(f"❌ Error: File '{file_path}' not found!")
        print("Usage: python read_from_pdf.py <pdf_file>")
        sys.exit(1)
    
    main(file_path=file_path)
