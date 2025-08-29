
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
def pdf_to_text(local_pdf_path, gcs_txt_path, min_text_length=50):
    """
    Extracts text from a PDF using native text only.
    If any page requires OCR, exit early with a message.
    Always returns gcs_txt_path.
    """
    # print(f"📄 Extracting text (no OCR) from {local_pdf_path} ...")

    pages_text = []
    with fitz.open(local_pdf_path) as doc:
        for page_num, page in enumerate(doc, start=1):
            page_text = page.get_text("text") or ""

            if len(page_text.strip()) < min_text_length:
                print("❌ This document demands OCR scanning")
                #return gcs_txt_path  # exit early, do NOT upload
                sys.exit(0)  # 🚪 exit application with error code

            #print(f"✅ Extracted text directly from page {page_num}")
            pages_text.append(page_text.strip())

    full_text = "\n\n".join(pages_text)

    bucket_name, blob_name = gcs_txt_path.replace("gs://", "").split("/", 1)
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(blob_name)

    # 🔑 ensure UTF-8 upload
    blob.upload_from_string(full_text.encode("utf-8"), content_type="text/plain; charset=utf-8")

    # Optional sanity check
    if not blob.exists():
        raise RuntimeError(f"❌ Upload failed: {gcs_txt_path} not found in GCS!")

    # print(f"✅ TXT uploaded til: {gcs_txt_path} (length {len(full_text)} chars)")
    return gcs_txt_path


# --- INDEXING ---
def index_txt(gcs_txt_path):
    bucket_name, blob_name = gcs_txt_path.replace("gs://", "").split("/", 1)
    text = storage_client.bucket(bucket_name).blob(blob_name).download_as_text()
    docs = [c.strip() for c in text.split("\n\n") if c.strip()]
    # print(f"✂️ Splitting into {len(docs)} chunks...")

    # Clear old docs in Firestore
    old_ids = []
    for doc in firestore_client.collection(FIRESTORE_COLLECTION).stream():
        old_ids.append(doc.id)
        doc.reference.delete()

    # print(f"🧹 Cleared {len(old_ids)} old docs from Firestore.")

    # Clear old datapoints in Matching Engine (optional, will fail gracefully if none exist)
    if old_ids:
        try:
            index_service.remove_datapoints(
                request=aiplatform_v1.RemoveDatapointsRequest(
                    index=f"projects/{PROJECT_NO_ID}/locations/{LOCATION}/indexes/{INDEX_ID}",
                    datapoint_ids=old_ids
                )
            )
            # print(f"🧹 Cleared {len(old_ids)} old datapoints from Matching Engine.")
        except Exception as e:
            print(f"⚠️ Could not clear old datapoints: {e}")

    # Index new docs
    datapoints = []
    for chunk in docs:
        doc_id = str(uuid.uuid4())   # unique ID shared between Firestore + Matching Engine
        emb = embedding_model.get_embeddings([chunk])[0]

        # Save to Firestore
        firestore_client.collection(FIRESTORE_COLLECTION).document(doc_id).set(
            {"text": chunk, "source": gcs_txt_path}
        )

        # Add to datapoints for Matching Engine
        datapoints.append(
            aiplatform_v1.IndexDatapoint(
                datapoint_id=doc_id,
                feature_vector=emb.values
            )
        )
        # print(f"   → Saved Firestore+Index doc {doc_id[:8]}... "
        #      f"({len(chunk)} chars, embedding={len(emb.values)})")

    # Push datapoints to Matching Engine (using IndexServiceClient)
    if datapoints:
        req = aiplatform_v1.UpsertDatapointsRequest(
            index=INDEX_RESOURCE,
            datapoints=datapoints
        )
        index_service.upsert_datapoints(request=req)
        # print(f"✅ Uploaded {len(datapoints)} datapoints to Matching Engine")


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




