from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
import easyocr
import tempfile
import os
import traceback

app = FastAPI()

# بيتحمل مرة واحدة لما السيرفر يبدأ
print("Loading EasyOCR model...")
reader = easyocr.Reader(['en'], gpu=False)
print("EasyOCR model loaded!")

@app.get("/health")
def health():
    return {"status": "ok", "model": "easyocr"}

@app.post("/ocr")
async def run_ocr(image: UploadFile = File(...)):
    tmp_path = None
    try:
        # احفظ الصورة مؤقتاً
        suffix = os.path.splitext(image.filename)[1] or ".png"
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            contents = await image.read()
            tmp.write(contents)
            tmp_path = tmp.name

        # شغّل الـ OCR
        results = reader.readtext(tmp_path)

        # استخرج النصوص اللي confidence بتاعتها أكتر من 0.2
        texts = [
            {
                "text": res[1],
                "confidence": round(res[2], 2)
            }
            for res in results
            if res[2] > 0.2
        ]

        print(f"OCR completed: {len(texts)} texts found")
        return JSONResponse(content={"texts": texts})
        
        except Exception as e:
            traceback.print_exc()
            return JSONResponse(
                status_code=500,
                content={"error": str(e)}
                )

    
        return JSONResponse(
            status_code=500,
            content={"error": str(e)}
        )

    finally:
        # امسح الصورة المؤقتة دايماً
        if tmp_path and os.path.exists(tmp_path):
            os.remove(tmp_path)