
from fastapi import FastAPI, Form, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
import uvicorn
from pipeline import find_possible_cvr, hent_aarsrapport, pdf_to_text, index_txt, qna_loop

app = FastAPI()
templates = Jinja2Templates(directory="templates")

@app.get("/", response_class=HTMLResponse)
def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/process", response_class=HTMLResponse)
def process(request: Request, name: str = Form(...)):
    matches = find_possible_cvr(name)
    if not matches:
        return templates.TemplateResponse("result.html", {"request": request, "name": name, "error": "Ingen matches fundet"})
    
    valgt_navn, valgt_cvr, _ = matches[0]  # for now, just take first
    local_pdf, gcs_pdf = hent_aarsrapport(valgt_cvr)
    if not local_pdf:
        return templates.TemplateResponse("result.html", {"request": request, "name": name, "error": "Ingen årsrapport fundet"})

    gcs_txt = gcs_pdf.replace(".pdf", ".txt")
    pdf_to_text(local_pdf, gcs_txt)
    index_txt(gcs_txt)

    return templates.TemplateResponse("result.html", {"request": request, "name": valgt_navn, "cvr": valgt_cvr, "success": True})

if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=8080, reload=True)

