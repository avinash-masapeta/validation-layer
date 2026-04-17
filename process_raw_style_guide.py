from pathlib import Path
import cv2
import fitz
import pytesseract
import numpy as np

def get_model():
    from pathlib import Path
    from huggingface_hub import hf_hub_download
    from ultralytics import YOLO
    DOWNLOAD_PATH = Path("./models")
    DOWNLOAD_PATH.mkdir(exist_ok=True)
    model_files = [
        "yolo26n_doc_layout.pt",
        "yolo26s_doc_layout.pt",
        "yolo26m_doc_layout.pt",
    ]
    selected_model_file = model_files[0] 
    model_path = hf_hub_download(
        repo_id="Armaggheddon/yolo26-document-layout",
        filename=selected_model_file,
        repo_type="model",
        local_dir=DOWNLOAD_PATH,
    )
    model = YOLO(model_path)
    return model

def extract_pdf_layout(pdf_path, model, out_root="layout_output", dpi=200):
    text_classes = {
        "Caption", "Footnote", "Formula", "List-item", "Page-footer",
        "Page-header", "Section-header", "Text", "Title"
    }

    pdf_name = Path(pdf_path).stem
    out_dir = Path(out_root) / pdf_name

    (out_dir / "text").mkdir(parents=True, exist_ok=True)
    (out_dir / "figures").mkdir(parents=True, exist_ok=True)

    result = {"pdf_path": str(pdf_path), "pages": []}

    doc = fitz.open(pdf_path)

    for page_num, page in enumerate(doc, start=1):
        pix = page.get_pixmap(dpi=dpi)

        img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(
            pix.height, pix.width, pix.n
        )
        if pix.n == 4:
            img = cv2.cvtColor(img, cv2.COLOR_RGBA2BGR)
        else:
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

        page_info = {"page_num": page_num, "blocks": []}

        for r in model(img):
            for i, b in enumerate(r.boxes):
                cls = r.names[int(b.cls)]
                x1, y1, x2, y2 = map(int, b.xyxy[0])
                crop = img[y1:y2, x1:x2]

                block = {"class": cls, "bbox": [x1, y1, x2, y2]}

                if cls in text_classes:
                    text = pytesseract.image_to_string(crop).strip()

                    img_path = out_dir / "text" / f"p{page_num}_{i}_{cls}.png"
                    txt_path = img_path.with_suffix(".txt")

                    cv2.imwrite(str(img_path), crop)
                    txt_path.write_text(text)

                    block.update({
                        "type": "text",
                        "image_path": str(img_path),
                        "text": text
                    })
                else:
                    img_path = out_dir / "figures" / f"p{page_num}_{i}_{cls}.png"
                    cv2.imwrite(str(img_path), crop)

                    block.update({
                        "type": "figure",
                        "image_path": str(img_path)
                    })

                page_info["blocks"].append(block)

        result["pages"].append(page_info)

    doc.close()
    return result

from pathlib import Path
import json

def process_pdf_dir(pdf_dir, model, out_root="layout_output", dpi=200):
    pdf_dir = Path(pdf_dir)
    all_results = {}

    for pdf_path in pdf_dir.glob("*.pdf"):
        result = extract_pdf_layout(pdf_path, model, out_root=out_root, dpi=dpi)
        all_results[pdf_path.name] = result

        out_json = Path(out_root) / pdf_path.stem / "layout.json"
        out_json.write_text(json.dumps(result, indent=2))

    return all_results

model = get_model()
process_pdf_dir("/Users/rohitk4/Downloads/VALIDATION LAYER/data/Packaging guides", model)