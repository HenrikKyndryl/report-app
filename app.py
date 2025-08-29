import os
from fastapi import FastAPI, Form, Request
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

# import your real pipeline
import pipeline
from pipeline import OCRRequired

APP_TITLE = "Report App"

app = FastAPI(title=APP_TITLE)

# static + templates
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")


@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


@app.post("/search", response_class=HTMLResponse)
async def search(request: Request, company: str = Form(...)):
    # Real CVR search
    try:
        matches = pipeline.find_possible_cvr(company)
    except Exception as e:
        return templates.TemplateResponse(
            "error.html",
            {"request": request, "message": f"CVR-søgning fejlede: {e}"}
        )

    # matches is list of (navn, cvr, form)
    companies = [{"name": n, "cvr": c, "form": f} for (n, c, f) in matches]
    return templates.TemplateResponse(
        "companies.html",
        {"request": request, "companies": companies, "query": company}
    )


@app.get("/choose/{cvr}", response_class=HTMLResponse)
async def choose(request: Request, cvr: str):
    """
    After the user chooses a company:
    - fetch årsrapport
    - text-extract (no OCR; if OCR needed -> friendly page)
    - upload TXT to GCS
    - index to Firestore + Matching Engine (tagged by CVR)
    - show Ask page
    """
    # fetch some name for header (best-effort: re-search by CVR)
    display_name = cvr
    try:
        matches = pipeline.find_possible_cvr(cvr)
        if matches:
            display_name = matches[0][0]
    except:
        pass

    try:
        local_pdf, gcs_pdf = pipeline.hent_aarsrapport(cvr)
        if not local_pdf:
            return templates.TemplateResponse(
                "error.html",
                {"request": request, "message": "Ingen årsrapport fundet for denne virksomhed."}
            )

        gcs_txt = gcs_pdf.replace(".pdf", ".txt")
        # web-safe (no sys.exit)
        pipeline.pdf_to_text_websafe(local_pdf, gcs_txt)
        # per-company indexing
        pipeline.index_txt_for_company(gcs_txt, cvr)

    except OCRRequired as e:
        return templates.TemplateResponse(
            "ocr_required.html",
            {"request": request, "message": str(e), "company": {"name": display_name, "cvr": cvr}}
        )
    except Exception as e:
        return templates.TemplateResponse(
            "error.html",
            {"request": request, "message": f"Behandling fejlede: {e}"}
        )

    # success → ask page
    return templates.TemplateResponse(
        "ask.html",
        {"request": request, "company": {"name": display_name, "cvr": cvr}}
    )


@app.post("/ask", response_class=HTMLResponse)
async def ask(request: Request, question: str = Form(...), cvr: str = Form(...)):
    # Q&A over the *indexed* content for this CVR
    try:
        answer = pipeline.answer_question(cvr, question)
    except Exception as e:
        return templates.TemplateResponse(
            "error.html",
            {"request": request, "message": f"Q&A fejlede: {e}"}
        )

    # Show result + two choices
    # (Ask new about same company) or (Search new company)
    company_name = cvr
    try:
        matches = pipeline.find_possible_cvr(cvr)
        if matches:
            company_name = matches[0][0]
    except:
        pass

    return templates.TemplateResponse(
        "result.html",
        {
            "request": request,
            "company": {"name": company_name, "cvr": cvr},
            "question": question,
            "answer": answer
        }
    )
