from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import sys, os, json

sys.path.append(os.path.dirname(__file__))

from scraper.scraper import scrape_brokers
from ai_engine.ner_pipeline import extract_pii, classify_risk, score_identity_match
from database.db import init_db, insert_record, get_all_records, update_status
from scraper.browser import run_optout_sync

app = FastAPI(title="TraceZero API", version="1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

init_db()

class UserProfile(BaseModel):
    name:  str
    city:  str = ""
    state: str = ""
    phone: str = ""
    email: str = ""

class EnforceRequest(BaseModel):
    record_id: int
    broker:    str
    profile:   UserProfile


@app.get("/")
def root():
    return {"status": "TraceZero API is running ✅", "version": "1.0"}


@app.post("/scan")
def scan(profile: UserProfile):
    user_dict = profile.dict()

    name  = profile.name.strip()
    city  = profile.city  or ""
    phone = profile.phone or ""
    email = profile.email or ""

    if len(name.split()) < 2:
        return {"success": False, "error": "Please provide both a first and last name (e.g. 'John Smith').", "total": 0, "records": []}

    broker_meta = {
        "Addresses":  {"optout_url": "https://www.addresses.com/optout",       "optout_email": "privacy@addresses.com",  "optout_notes": "Submit removal request via their online form."},
        "Intelius":   {"optout_url": "https://www.intelius.com/opt-out",       "optout_email": "privacy@intelius.com",   "optout_notes": "Submit opt-out form with email confirmation."},
        "ZabaSearch": {"optout_url": "https://www.zabasearch.com/block_user/", "optout_email": "privacy@zabasearch.com", "optout_notes": "Submit removal via zabasearch.com/block_user."},
    }

    # ── Step 1: Real scraping ─────────────────────────────────────────────
    print(f"\n🔍 Scraping live brokers for: {name}")
    scraped_list = []
    try:
        scraped_list = scrape_brokers(user_dict)
    except Exception as e:
        print(f"  ⚠️ Scraping error: {e}")

    # ── Step 2: Build pages list from scraped results only ────────────────
    all_pages = []
    for r in scraped_list:
        broker_name = r["broker"]
        raw_text    = r.get("raw_text", "")
        status      = r.get("status", "")
        meta        = broker_meta.get(broker_name, {
            "optout_url":   f"https://www.{broker_name.lower().replace(' ','')}.com/optout",
            "optout_email": f"privacy@{broker_name.lower().replace(' ','')}.com",
            "optout_notes": "Submit opt-out request on their website.",
        })

        # Only include pages where we actually got text back
        if status == "scraped" and raw_text:
            print(f"  ✅ {broker_name}: {len(raw_text)} chars scraped")
            all_pages.append({
                "broker":       broker_name,
                "raw_text":     raw_text,
                "optout_url":   r.get("optout_url",   meta["optout_url"]),
                "optout_email": r.get("optout_email", meta["optout_email"]),
                "optout_notes": r.get("optout_notes", meta["optout_notes"]),
            })
        else:
            print(f"  ⚠️ {broker_name}: status={status} — skipped")

    # ── Step 3: Run REAL AI pipeline on every page ────────────────────────
    results = []
    for record in all_pages:
        raw_text    = record["raw_text"]
        broker_name = record["broker"]

        # Real RoBERTa NER
        pii = extract_pii(raw_text)

        # Real DistilBERT risk classification
        risk = classify_risk(raw_text, pii)

        # Real Sentence-BERT identity match scoring
        match = score_identity_match(pii, user_dict)

        # Build clean extracted_pii from NER output
        extracted = {}
        for entity_type, entities in pii.items():
            if entities:
                val = entities[0]["value"] if isinstance(entities[0], dict) else str(entities[0])
                extracted[entity_type] = val

        # Supplement with obvious fields NER might have missed
        if "PER" not in extracted and name.split()[0].lower() in raw_text.lower():
            extracted["PER"] = name
        if "PHONE" not in extracted and phone in raw_text:
            extracted["PHONE"] = phone
        if "EMAIL" not in extracted and email in raw_text:
            extracted["EMAIL"] = email
        if "LOC" not in extracted and city.lower() in raw_text.lower():
            extracted["LOC"] = city

        # Save to database
        insert_record(
            broker      = broker_name,
            broker_type = "data_broker",
            url         = record.get("optout_url", ""),
            raw_text    = raw_text[:300],
            pii_found   = json.dumps(extracted),
            match_score = round(match, 2),
        )

        results.append({
            "broker":          broker_name,
            "optout_url":      record["optout_url"],
            "optout_email":    record["optout_email"],
            "optout_notes":    record["optout_notes"],
            "extracted_pii":   extracted,
            "ai_pii":          pii,
            "risk_level":      risk.get("label", "Medium"),
            "risk_confidence": round(risk.get("confidence", 0.75), 2),
            "match_score":     round(match, 2),
            "status":          "Found",
        })

    print(f"\n✅ Scan complete: {len(results)} records processed")

    return {
        "success": True,
        "total":   len(results),
        "records": results,
    }


@app.get("/records")
def records():
    rows = get_all_records()
    cols = ["id","broker","broker_type","url","raw_text","pii_found","match_score","status","created_at"]
    data = [dict(zip(cols, row)) for row in rows]
    return {"success": True, "total": len(data), "records": data}


@app.post("/enforce")
def enforce(req: EnforceRequest):
    result = run_optout_sync(req.broker, req.profile.dict())
    update_status(req.record_id, "Removal Sent")
    return {
        "success":     result.get("success", False),
        "message":     result.get("message", ""),
        "manual_step": result.get("manual_step", ""),
        "broker":      req.broker,
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)