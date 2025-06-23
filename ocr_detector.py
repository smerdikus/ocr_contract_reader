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
warnings.filterwarnings("ignore")


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
    return num_pages


def run_easyocr(folder='output', lang='cs'):
    """
    Use EasyOCR for text extraction.
    """
    try:
        reader = easyocr.Reader([lang])
    except Exception as e:
        print(f"ERROR initializing EasyOCR: {e}")
        return []
    
    results = []
    for file in sorted(os.listdir(folder)):
        if file.endswith('.png'):
            print(f"Reading text from {file} using EasyOCR")
            try:
                result = reader.readtext(os.path.join(folder, file))
                results.append((file, result))            
            except Exception as e:
                print(f"ERROR processing {file}: {e}")
                results.append((file, []))
    return results


def extract_contract_sections(text):
    """
    Extract the three specific sections from the contract:
    1. Header (everything before "1.")
    2. Section "1. ÚVODNÍ USTANOVENÍ" (until "2. PŘEDMĚT SMLOUVY")
    3. Section "3. KUPNÍ CENA" (until "4. PROHLÁŠENÍ")
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
    section_1_match = re.search(r'\s*ÚVODNÍ\s+USTANOVENÍ', full_text, re.IGNORECASE)
    
    # Find the position of "2. PŘEDMĚT SMLOUVY"
    section_2_match = re.search(r'\s*PŘEDMĚT\s+SMLOUVY', full_text, re.IGNORECASE)
    
    # Find the position of "3. KUPNÍ CENA"
    section_3_match = re.search(r'\s*KUPNÍ\s+CENA', full_text, re.IGNORECASE)
    
    # Find the position of "4. PROHLÁŠENÍ"
    section_4_match = re.search(r'\s*PROHLÁŠENÍ', full_text, re.IGNORECASE)
    
    if section_1_match:
        # Header is everything before section 1
        header_end = section_1_match.start()
        extracted_sections["header"] = full_text[:header_end].strip()
        
        # Section 1 starts from the match and ends at section 2
        section_1_start = section_1_match.start()
        if section_2_match:
            section_1_end = section_2_match.start()
            extracted_sections["section_1_uvodni_ustanoveni"] = full_text[section_1_start:section_1_end].strip()
        else:
            # If section 2 not found, take until section 3 or end
            if section_3_match:
                section_1_end = section_3_match.start()
                extracted_sections["section_1_uvodni_ustanoveni"] = full_text[section_1_start:section_1_end].strip()
            else:
                extracted_sections["section_1_uvodni_ustanoveni"] = full_text[section_1_start:].strip()
    
    if section_3_match:
        # Section 3 starts from the match and ends at section 4
        section_3_start = section_3_match.start()
        if section_4_match:
            section_3_end = section_4_match.start()
            extracted_sections["section_3_kupni_cena"] = full_text[section_3_start:section_3_end].strip()
        else:
            # If section 4 not found, take everything from section 3 to end or until signatures
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


def extract_template_from_contract(json_path_or_data):
    """
    Extract template structure from parsed contract JSON.
    
    Args:
        json_path_or_data: Either a file path to JSON or the JSON data itself
    
    Returns:
        dict: Template structure with contract details and validation info
    """
    # Load data if path is provided, otherwise use the data directly
    if isinstance(json_path_or_data, str):
        with open(json_path_or_data, 'r', encoding='utf-8') as f:
            contract_data = json.load(f)
    else:
        contract_data = json_path_or_data
    
    # Extract text sections
    header = contract_data.get("header", "")
    section_1 = contract_data.get("section_1_uvodni_ustanoveni", "")
    section_3 = contract_data.get("section_3_kupni_cena", "")
    
    # Initialize template structure
    template = {
        "smlouva": "",
        "celkem cena": "",
        "smluvni strany": [],
        "validation": {
            "price_sum_matches": False,
            "calculated_sum": "0",
            "difference": "0"
        }
    }
    
    # Extract contract number from header
    contract_match = re.search(r'číslo:\s*([A-Z0-9]+)', header, re.IGNORECASE)
    if contract_match:
        template["smlouva"] = contract_match.group(1)
    
    # Extract total price from section 3
    total_price_match = re.search(r'celková kupní cena.*?činí\s*(\d+[\s\d]*)\s*Kč', section_3, re.IGNORECASE)
    total_price = 0
    if total_price_match:
        # Remove spaces from the price
        price_str = total_price_match.group(1).replace(' ', '')
        template["celkem cena"] = price_str
        total_price = int(price_str)    # Extract selling parties (Prodávající) from header
    # Improved pattern to handle Czech names and addresses better
    sellers_pattern = r'([A-ZÁČĎÉĚÍŇÓŘŠŤÚŮÝŽ][a-záčďéěíňóřšťúůýž]+(?:\s+[A-ZÁČĎÉĚÍŇÓŘŠŤÚŮÝŽ][a-záčďéěíňóřšťúůýž]+)*),\s*RČ\s*[\d/]+,\s*bytem\s+([^(]+?)\s*\(dále jen Prodávajicí'
    
    sellers = re.findall(sellers_pattern, header, re.IGNORECASE)
    print(f"DEBUG: Found {len(sellers)} sellers: {sellers}")
    
    # If no sellers found with the first pattern, try a more flexible one
    if not sellers:
        # Alternative pattern for names with different formats
        sellers_pattern_alt = r'([A-ZÁČĎÉĚÍŇÓŘŠŤÚŮÝŽ][a-záčďéěíňóřšťúůýž]+(?:\s+[A-ZÁČĎÉĚÍŇÓŘŠŤÚŮÝŽ][a-záčďéěíňóřšťúůýž]+)*),\s*RČ\s*[\d/]+,\s*bytem\s+([^,()]+?)(?:\s*\(|,)'
        sellers = re.findall(sellers_pattern_alt, header, re.IGNORECASE)
        print(f"DEBUG: Alternative pattern found {len(sellers)} sellers: {sellers}")
      # Extract parcel information from section 1
    # Improved pattern to handle various parcel number formats and shared ownership
    print(f"\nDEBUG: Analyzing section 1 for parcel extraction...")
    print(f"Section 1 preview: {section_1[:500]}...")
    
    # Pattern to find parcel number and LV
    parcel_basic_pattern = r'parc\s*č\.\s*([^,]+?)(?:,.*?)?.*?LV\s*č[:.]?\s*(\d+)'
    basic_parcel_matches = re.findall(parcel_basic_pattern, section_1, re.IGNORECASE)
    
    # Pattern to find individual ownership shares for each seller
    ownership_pattern = r'Prodávající\s*(\d+)\s*vlastní\s*spoluvlastnický.*?velikosti\s*(\d+/\d+)'
    ownership_matches = re.findall(ownership_pattern, section_1, re.IGNORECASE)
    
    print(f"DEBUG: Basic parcel matches: {basic_parcel_matches}")
    print(f"DEBUG: Ownership matches: {ownership_matches}")
    
    # Create ownership mapping
    ownership_map = {}
    for seller_num, share in ownership_matches:
        ownership_map[seller_num] = share
    
    # Extract individual prices from section 3 subsections
    # First, let's find all subsections with the pattern "a) Kupní cena ve výši XXXXX Kč"
    print(f"\nDEBUG: Analyzing section 3 for price extraction...")
    print(f"Section 3 preview: {section_3[:500]}...")
    
    # Pattern 1: Full subsection pattern with seller reference
    subsection_pattern_1 = r'([a-z])\)\s*Kupní cena ve výši\s*([\d\s]+)\s*Kč.*?(?:byla uhrazena|uhrazena).*?Prodávajícímu\s*(\d+)'
    subsection_prices_1 = re.findall(subsection_pattern_1, section_3, re.IGNORECASE | re.DOTALL)
    
    # Pattern 2: Simplified pattern just for price in subsections
    subsection_pattern_2 = r'([a-z])\)\s*Kupní cena ve výši\s*([\d\s]+)\s*Kč'
    subsection_prices_2 = re.findall(subsection_pattern_2, section_3, re.IGNORECASE)
    
    # Pattern 3: Any "Kupní cena ve výši" with seller reference
    general_price_pattern = r'Kupní cena ve výši\s*([\d\s]+)\s*Kč.*?(?:byla uhrazena|uhrazena).*?Prodávajícímu\s*(\d+)'
    general_prices = re.findall(general_price_pattern, section_3, re.IGNORECASE | re.DOTALL)
    
    print(f"DEBUG: Pattern 1 found: {subsection_prices_1}")
    print(f"DEBUG: Pattern 2 found: {subsection_prices_2}")
    print(f"DEBUG: General pattern found: {general_prices}")
    
    # Choose the best extraction method
    subsection_prices = []
    if subsection_prices_1:
        subsection_prices = subsection_prices_1
        print("Using pattern 1 (full subsection with seller)")
    elif subsection_prices_2 and general_prices:
        # Combine subsection letters with seller info from general pattern
        for i, (subsection, price) in enumerate(subsection_prices_2):
            if i < len(general_prices):
                seller_num = general_prices[i][1]  # Get seller number from general pattern
                subsection_prices.append((subsection, price, seller_num))
        print("Using combined pattern 2 + general")
    elif general_prices:
        # Create artificial subsection letters
        for i, (price, seller_num) in enumerate(general_prices):
            subsection_letter = chr(ord('a') + i)  # a, b, c, ...
            subsection_prices.append((subsection_letter, price, seller_num))
        print("Using general pattern with artificial subsections")
    
    # Create a mapping of seller numbers to prices and subsections
    price_map = {}
    subsection_map = {}
    total_individual_sum = 0
    
    for subsection, price, seller_num in subsection_prices:
        clean_price = price.replace(' ', '')
        price_map[seller_num] = clean_price
        subsection_map[seller_num] = subsection
        total_individual_sum += int(clean_price)
    
    # Build the smluvni strany structure
    for i, (name, address) in enumerate(sellers, 1):
        seller_num = str(i)
        seller_price = price_map.get(seller_num, "0")
        seller_subsection = subsection_map.get(seller_num, chr(ord('a') + i - 1))
        
        seller_data = {
            "jmeno": name.strip(),
            "bydliste": address.strip(),
            "parcely": [],
            "cena": seller_price,
            "subsection": f"{seller_subsection})"
        }
          # Add parcel information with individual ownership shares
        if basic_parcel_matches:
            parcel_num, lv_num = basic_parcel_matches[0]  # Take the first (and likely only) parcel
            seller_share = ownership_map.get(seller_num, "1/3")  # Default to 1/3 if not found
            
            seller_data["parcely"].append({
                "cislo parcely": parcel_num.strip(),
                "podil": seller_share.strip(),
                "LV": lv_num.strip()
            })
        
        template["smluvni strany"].append(seller_data)
    
    # Validation: Check if sum of individual prices matches total price
    template["validation"]["calculated_sum"] = str(total_individual_sum)
    template["validation"]["difference"] = str(abs(total_price - total_individual_sum))
    template["validation"]["price_sum_matches"] = (total_price == total_individual_sum)
    
    # Print validation results
    print(f"\n💰 Price Validation:")
    print(f"   Total price (celkem cena): {total_price} Kč")
    print(f"   Sum of individual prices: {total_individual_sum} Kč")
    print(f"   Difference: {abs(total_price - total_individual_sum)} Kč")
    if total_price == total_individual_sum:
        print(f"   prices are equal!")
    else:
        print(f"   prices are not equal!")
    
    return template


def save_template(template_data, output_path="template.json"):
    """
    Save template data to JSON file.
    
    Args:
        template_data: Template dictionary
        output_path: Output file path
    """
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(template_data, f, ensure_ascii=False, indent=4)
    print(f"Template saved to: {output_path}")


def main(file_path="smlouva.pdf"):
    """
    Main function using EasyOCR for text extraction.
    """
    print(f"\nStarting PDF processing: {file_path}")
    print("OCR Method: EasyOCR")
      # Run conversion to individual page images
    doc_size = convert_pdf_to_image(file_path)
    
    # Process each page separately (no merging)
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
        os.remove(page_image)    # Extract text from all OCR results and combine
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
    
    # Extract the three specific sections requested
    print("Extracting specific contract sections...")
    extracted_sections = extract_contract_sections(full_text)
    
    # Create simplified output with only requested fields
    simplified_output = {
        "total_characters": len(full_text),
        "full_text": full_text.strip(),
        "header": extracted_sections["header"],
        "section_1_uvodni_ustanoveni": extracted_sections["section_1_uvodni_ustanoveni"],
        "section_3_kupni_cena": extracted_sections["section_3_kupni_cena"]
    }
    
    # Print results summary
    print("\nDocument Processing Summary:")
    print("=" * 40)
    print(f"Total characters: {simplified_output['total_characters']}")
    print(f"Header length: {len(simplified_output['header'])} chars")
    print(f"Section 1 length: {len(simplified_output['section_1_uvodni_ustanoveni'])} chars")
    print(f"Section 3 length: {len(simplified_output['section_3_kupni_cena'])} chars")
    
    # Show preview of extracted sections
    print(f"\nExtracted Sections Preview:")
    sections_preview = [
        ("Header", simplified_output["header"]),
        ("Section 1 - ÚVODNÍ USTANOVENÍ", simplified_output["section_1_uvodni_ustanoveni"]),
        ("Section 3 - KUPNÍ CENA", simplified_output["section_3_kupni_cena"])
    ]
    
    for section_name, section_content in sections_preview:
        if section_content:
            preview = section_content[:150] if len(section_content) > 150 else section_content
            print(f"\n{section_name}:")
            print(f"{preview}...")
        else:
            print(f"\n{section_name}: Not found")
      # Save simplified data to JSON file
    output_file = 'output/parsed_contract.json'
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(simplified_output, f, ensure_ascii=False, indent=2)
    
    print(f"\nSimplified contract data saved to: {output_file}")
    print("Contains only: total_characters, full_text, header, section_1_uvodni_ustanoveni, section_3_kupni_cena")
    
    # Extract and save template
    print("\nExtracting template structure...")
    template_data = extract_template_from_contract(simplified_output)
    save_template(template_data, "output/template.json")
    
    # Show template preview
    print(f"\nTemplate Preview:")
    print(f"   Contract: {template_data['smlouva']}")
    print(f"   Total Price: {template_data['celkem cena']} Kč")
    print(f"   Number of Sellers: {len(template_data['smluvni strany'])}")
    
    for i, seller in enumerate(template_data['smluvni strany'], 1):
        print(f"   Seller {i}: {seller['jmeno']} - {seller['cena']} Kč")
        if seller['parcely']:
            for parcel in seller['parcely']:
                print(f"      Parcel: {parcel['cislo parcely']}, Share: {parcel['podil']}, LV: {parcel['LV']}")
    
    return simplified_output, template_data


def process_existing_json(json_file_path):
    """
    Process an existing parsed_contract.json file to extract template.
    
    Args:
        json_file_path: Path to the parsed_contract.json file
    
    Returns:
        dict: Template data
    """
    if not os.path.exists(json_file_path):
        print(f"ERROR: File '{json_file_path}' not found!")
        return None

    print(f"Processing existing JSON file: {json_file_path}")
    
    try:
        # Extract template from the existing JSON
        template_data = extract_template_from_contract(json_file_path)
        
        # Save template
        output_dir = os.path.dirname(json_file_path)
        template_output_path = os.path.join(output_dir, "template.json")
        save_template(template_data, template_output_path)
        
        # Show results
        print(f"\nTemplate extracted successfully!")
        print(f"   Contract: {template_data['smlouva']}")
        print(f"   Total Price: {template_data['celkem cena']} Kč")
        print(f"   Number of Sellers: {len(template_data['smluvni strany'])}")
        
        for i, seller in enumerate(template_data['smluvni strany'], 1):
            print(f"   Seller {i}: {seller['jmeno']} - {seller['cena']} Kč")
        
        return template_data
        
    except Exception as e:
        print(f"ERROR processing JSON file: {e}")
        return None


if __name__ == "__main__":
    # Get file path from command line argument
    file_path = sys.argv[1] if len(sys.argv) > 1 else "smlouva.pdf"
    
    # Check if file exists
    if not os.path.exists(file_path):
        print(f"ERROR: File '{file_path}' not found!")
        print("Usage: python read_from_pdf.py <pdf_file_or_json_file>")
        sys.exit(1)
    

    main(file_path=file_path)
    process_existing_json("output/parsed_contract.json")
