
import os
import uuid
import fitz  # PyMuPDF
import pytesseract
from PIL import Image
import requests
import json
import io
import vertexai
import warnings
warnings.filterwarnings(
    "ignore",
    category=UserWarning,
    module="vertexai._model_garden._model_garden_models"
)

from google.cloud.aiplatform.matching_engine import MatchingEngineIndexEndpoint
from google.cloud import storage, firestore, aiplatform_v1

# from vertexai.preview.language_models import TextEmbeddingModel, TextEmbeddingInput
from vertexai.language_models import TextEmbeddingModel

from google.cloud import aiplatform_v1
from google.cloud import aiplatform



# === CONFIGURATION ===
PROJECT_NO_ID = "1011789646399"
PROJECT_ID = "nordic-genai-usecases"
LOCATION = "us-central1"

INDEX_ID = "2748376648184233984"
INDEX_RESOURCE = f"projects/{PROJECT_ID}/locations/{LOCATION}/indexes/{INDEX_ID}"
INDEX_ENDPOINT_RESOURCE_NAME = (
    "projects/1011789646399/locations/us-central1/indexEndpoints/961819786629611520"
)
DEPLOYED_INDEX_ID = "virk_index_deploy_1756367187623"
INDEX_RESOURCE_NAME = "projects/1011789646399/locations/us-central1/indexes/2748376648184233984"

FIRESTORE_COLLECTION = "cvr_chunks"
BUCKET_NAME = "yearly-reports"




# CVR API
CVR_API_KEY = "cvr.dev_c76b18186777d8f7b96d7dfce4219cde"
SEARCH_URL  = "https://api.cvr.dev/api/cvr/virksomhed"

# Gemini API
API_KEY = os.getenv("PALM_API_KEY")
PRIMARY_MODEL = "models/gemini-1.5-pro-002"
API_KEY="AIzaSyCY1_at2uoZnl1oEPTjnsxqBMrnJujdDrU"

# === INIT CLIENTS ===
storage_client = storage.Client()
firestore_client = firestore.Client(project=PROJECT_ID)

aiplatform.init(project=PROJECT_ID, location=LOCATION)
index_endpoint = MatchingEngineIndexEndpoint(INDEX_ENDPOINT_RESOURCE_NAME)

vertexai.init(project=PROJECT_ID, location=LOCATION)
embedding_model = TextEmbeddingModel.from_pretrained("text-embedding-005")

# For datapoint management (index-level ops)
index_service = aiplatform_v1.IndexServiceClient(
    client_options={"api_endpoint": f"{LOCATION}-aiplatform.googleapis.com"}
)

# For querying (endpoint-level ops)
index_endpoint = MatchingEngineIndexEndpoint(INDEX_ENDPOINT_RESOURCE_NAME)


# ---- Add near the top of pipeline.py ----
class OCRRequired(Exception):
    """Raised when the PDF requires OCR (image-only pages)."""
    pass


# --- CVR SEARCH ---
ALLOWED_FORMS = {"A/S", "APS"}  # only these legal forms

def find_possible_cvr(name):
    headers = {"Authorization": f"Bearer {CVR_API_KEY}"}
    params = {
        "navn": name,
        "virksomhedsformer": "A/S,APS"
    }
    resp = requests.get(SEARCH_URL, headers=headers, params=params)
    resp.raise_for_status()
    data = resp.json()

    matches = []
    for org in data:
        navn = org.get("navne", [{}])[0].get("navn", "–")
        cvr = org.get("cvrNummer") or org.get("cvrnummer")

        # virksomhedsform may be dict OR list
        vf = org.get("virksomhedsform")
        if isinstance(vf, list) and vf:
            form = vf[0].get("kortBeskrivelse") or vf[0].get("langBeskrivelse")
        elif isinstance(vf, dict):
            form = vf.get("kortBeskrivelse") or vf.get("langBeskrivelse")
        else:
            form = None

        # ✅ enforce strict filter here
        if form in ALLOWED_FORMS:
            matches.append((navn, cvr, form))

    print(f"\nPossible enterprise matches for '{name}':")
    for idx, (navn, cvr, form) in enumerate(matches, start=1):
        print(f"{idx}. {navn} – CVR: {cvr} – Form: {form}")
    return matches



def hent_aarsrapport(cvr_nummer):
    url = "http://distribution.virk.dk/offentliggoerelser/_search"
    params = {"q": f"cvrNummer:{cvr_nummer}", "size": 50, "sort": "offentliggoerelsesTidspunkt:desc"}
    response = requests.get(url, params=params)
    response.raise_for_status()
    hits = response.json().get("hits", {}).get("hits", [])

    for hit in hits:
        source = hit.get("_source", {})
        for dok in source.get("dokumenter", []):
            if dok.get("dokumentType") == "AARSRAPPORT" and dok.get("dokumentMimeType") == "application/pdf":
                pdf_url = dok.get("dokumentUrl")
                if pdf_url:
                    print(f"🔽 Henter årsrapport: {pdf_url}")
                    r = requests.get(pdf_url)
                    if r.status_code == 200:
                        local_path = f"/tmp/{cvr_nummer}_årsrapport_2024.pdf"
                        with open(local_path, "wb") as f:
                            f.write(r.content)

                        blob_name = f"{cvr_nummer}_årsrapport_2024.pdf"
                        gcs_pdf = f"gs://{BUCKET_NAME}/{blob_name}"
                        storage_client.bucket(BUCKET_NAME).blob(blob_name).upload_from_filename(local_path)
                        print(f"✅ Yearly report for 2024 has been downloaded")
                        return local_path, gcs_pdf
    return None, None



# --- OCR PDF TO TXT ---
# ---- Add (web-safe) OCR check version; do NOT sys.exit here ----
def pdf_to_text_websafe(local_pdf_path, gcs_txt_path, min_text_length=50):
    """
    Same as pdf_to_text, but raises OCRRequired instead of exiting the process.
    Returns gcs_txt_path on success.
    """
    pages_text = []
    with fitz.open(local_pdf_path) as doc:
        for page_num, page in enumerate(doc, start=1):
            page_text = page.get_text("text") or ""
            if len(page_text.strip()) < min_text_length:
                raise OCRRequired(f"Dette pdf dokument kræver OCR scanning (side {page_num})")
            pages_text.append(page_text.strip())

    full_text = "\n\n".join(pages_text)
    bucket_name, blob_name = gcs_txt_path.replace("gs://", "").split("/", 1)
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(blob_name)
    blob.upload_from_string(full_text.encode("utf-8"), content_type="text/plain; charset=utf-8")

    if not blob.exists():
        raise RuntimeError(f"❌ Upload failed: {gcs_txt_path} not found in GCS!")

    return gcs_txt_path



# --- INDEXING ---
# ---- Update indexing to be per-company and non-destructive ----
def index_txt_for_company(gcs_txt_path, cvr_nummer):
    """
    Download TXT from GCS, split to chunks, embed and upsert into Matching Engine.
    Firestore docs are tagged with the CVR so we can filter at query time.
    Old docs for *this CVR* are removed before re-indexing.
    """
    bucket_name, blob_name = gcs_txt_path.replace("gs://", "").split("/", 1)
    text = storage_client.bucket(bucket_name).blob(blob_name).download_as_text()
    docs = [c.strip() for c in text.split("\n\n") if c.strip()]

    # Remove old docs for this CVR only
    old_ids = []
    for d in firestore_client.collection(FIRESTORE_COLLECTION).where("cvr", "==", cvr_nummer).stream():
        old_ids.append(d.id)
        d.reference.delete()

    if old_ids:
        try:
            index_service.remove_datapoints(
                request=aiplatform_v1.RemoveDatapointsRequest(
                    index=INDEX_RESOURCE,
                    datapoint_ids=old_ids
                )
            )
        except Exception as e:
            print(f"⚠️ Could not clear old datapoints for {cvr_nummer}: {e}")

    # Index new docs
    datapoints = []
    for chunk in docs:
        doc_id = str(uuid.uuid4())
        emb = embedding_model.get_embeddings([chunk])[0]

        # Save to Firestore with CVR tag
        firestore_client.collection(FIRESTORE_COLLECTION).document(doc_id).set(
            {"text": chunk, "source": gcs_txt_path, "cvr": cvr_nummer}
        )

        datapoints.append(
            aiplatform_v1.IndexDatapoint(
                datapoint_id=doc_id,
                feature_vector=emb.values
            )
        )

    if datapoints:
        req = aiplatform_v1.UpsertDatapointsRequest(index=INDEX_RESOURCE, datapoints=datapoints)
        index_service.upsert_datapoints(request=req)


# ---- Add a web-friendly Q&A helper that filters by CVR ----
def answer_question(cvr_nummer: str, question: str) -> str:
    """
    Embed the question, query Matching Engine, then keep only contexts whose Firestore doc has this CVR.
    Ask Gemini and return the text.
    """
    # 1) Embed question
    q_emb = embedding_model.get_embeddings([question])[0]

    # 2) Neighbors
    response = index_endpoint.find_neighbors(
        deployed_index_id=DEPLOYED_INDEX_ID,
        queries=[q_emb.values],
        num_neighbors=25
    )

    # 3) Collect contexts for this CVR only
    contexts = []
    for neighbor in response[0]:
        nn_id = neighbor.id
        doc = firestore_client.collection(FIRESTORE_COLLECTION).document(nn_id).get()
        if doc.exists and doc.get("cvr") == cvr_nummer:
            contexts.append(doc.get("text"))
        if len(contexts) >= 5:
            break

    context_text = "\n---\n".join(contexts) if contexts else "(ingen relevante uddrag fundet)"

    # 4) Ask Gemini (HTTP call like in your script)
    headers = {"Content-Type": "application/json", "x-goog-api-key": API_KEY}
    body = {
        "contents": [{"role": "user", "parts": [{"text": f"Brug kun denne kontekst:\n{context_text}\n\nSpørgsmål: {question}\nSvar på dansk."}]}],
        "generationConfig": {"temperature": 0.2, "maxOutputTokens": 256},
    }
    url = f"https://generativelanguage.googleapis.com/v1/{PRIMARY_MODEL}:generateContent"

    r = requests.post(url, headers=headers, data=json.dumps(body))
    if r.status_code != 200:
        return f"⚠️ LLM error: {r.text}"

    data = r.json()
    return (
        data.get("candidates", [{}])[0]
        .get("content", {})
        .get("parts", [{}])[0]
        .get("text", "")
        or "(intet svar)"
    )



# --- Q&A LOOP ---
def qna_loop():
    while True:
        q = input("\nAsk your question (or 'exit'): ").strip()
        if q.lower() == "exit":
            break

        # --- Embed the question ---
        emb = embedding_model.get_embeddings([q])[0]

        # --- Query Matching Engine ---
        deployed_index_id = index_endpoint.deployed_indexes[0].id
        response = index_endpoint.find_neighbors(
            deployed_index_id=DEPLOYED_INDEX_ID,
            queries=[emb.values],
            num_neighbors=10
        )

        # --- Collect nearest neighbors from Firestore ---
        contexts = []
        for neighbor in response[0]:
            nn_id = neighbor.id
            doc = firestore_client.collection(FIRESTORE_COLLECTION).document(nn_id).get()
            if doc.exists:
                contexts.append(doc.get("text"))
                # print(f" • Neighbor ID: {nn_id[:8]}, distance={neighbor.distance:.4f}, ✅ found in Firestore")
            else:
                print(f" • Neighbor ID: {nn_id[:8]}, distance={neighbor.distance:.4f}, ⚠️ Not in Firestore")

        # --- Build prompt for Gemini ---
        context_text = "\n---\n".join(contexts[:3])
        prompt = f"""Brug kun nedenstående kontekst til at besvare spørgsmålet.

Spørgsmål: {q}

Kontekst:
{context_text}

Svar:"""

        headers = {
            "Content-Type": "application/json",
            "x-goog-api-key": API_KEY,
        }
        body = {
            "contents": [{"role": "user", "parts": [{"text": prompt}]}],
            "generationConfig": {"temperature": 0.2, "maxOutputTokens": 256},
        }
        url = f"https://generativelanguage.googleapis.com/v1/{PRIMARY_MODEL}:generateContent"

        resp = requests.post(url, headers=headers, data=json.dumps(body))
        if resp.status_code != 200:
            print(f"⚠️ LLM error: {resp.text}")
            continue

        data = resp.json()
        answer = (
            data.get("candidates", [{}])[0]
            .get("content", {})
            .get("parts", [{}])[0]
            .get("text", "")
        )

        print("\n💡 LLM Answer:")
        print(answer)




# === MAIN ===
if __name__ == "__main__":
    navn = input("Indtast virksomhedsnavn (delvist ok): ").strip()
    matches = find_possible_cvr(navn)
    if not matches:
        exit()

    if len(matches) > 1:
        valg = input(f"Vælg virksomhed (1–{len(matches)}): ")
        idx = int(valg) - 1
    else:
        idx = 0

    valgt_navn, valgt_cvr, _ = matches[idx]
    print(f"\nValgt virksomhed: {valgt_navn} (CVR: {valgt_cvr})")

    local_pdf, gcs_pdf = hent_aarsrapport(valgt_cvr)
    if not local_pdf:
        print("❌ Ingen årsrapport fundet.")
        exit()

    gcs_txt = gcs_pdf.replace(".pdf", ".txt")
    pdf_to_text(local_pdf, gcs_txt)
    print("\n⚙️ Starter indeksering af rapporten...")
    index_txt(gcs_txt)

    print("\n💬 Klar til Q&A med Gemini...")
    qna_loop()




