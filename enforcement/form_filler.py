import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from scraper.browser          import run_optout_sync
from enforcement.email_sender import generate_removal_email
from database.db              import get_all_records, update_status

def enforce_removal(user_profile: dict) -> dict:
    results = {
        "forms_submitted": [],
        "emails_generated": [],
        "errors": [],
    }

    # ── Step 1: Browser form automation ───────────────────────────────────────
    print("\n📋 Running form-based opt-outs...")
    records_all = get_all_records()
    for record in records_all:
        broker = record[1]
        optout_result = run_optout_sync(broker, user_profile)
        if optout_result.get("success"):
            results["forms_submitted"].append(broker)
        else:
            results["errors"].append({"broker": broker, "message": optout_result.get("message", "")})

    # ── Step 2: Email generation for flagged records ───────────────────────────
    print("\n📧 Generating removal emails for flagged records...")
    records = get_all_records()
    flagged = [r for r in records if r[6] >= 0.5 and r[7] == "Found"]

    for record in flagged:
        rid, broker = record[0], record[1]
        email_text  = generate_removal_email(user_profile, broker)
        output_path = os.path.join(
            os.path.dirname(__file__),
            f"removal_email_{broker.replace(' ', '_')}.txt"
        )
        with open(output_path, "w") as f:
            f.write(email_text)
        update_status(rid, "Removal Sent")
        results["emails_generated"].append({
            "broker": broker,
            "saved_to": output_path,
        })
        print(f"  ✅ Email saved: {output_path}")

    return results


def print_enforcement_summary(results: dict):
    print("\n" + "="*50)
    print("📊 ENFORCEMENT SUMMARY")
    print("="*50)
    print(f"✅ Forms Submitted : {len(results['forms_submitted'])}")
    for b in results["forms_submitted"]:
        print(f"   - {b}")
    print(f"📧 Emails Generated: {len(results['emails_generated'])}")
    for e in results["emails_generated"]:
        print(f"   - {e['broker']} → {e['saved_to']}")
    if results["errors"]:
        print(f"❌ Errors          : {len(results['errors'])}")
        for err in results["errors"]:
            print(f"   - {err}")
    print("="*50)


if __name__ == "__main__":
    # Quick test run
    test_profile = {
        "name":  "John Smith",
        "city":  "Lahore",
        "state": "Punjab",
        "phone": "555-1234",
    }
    res = enforce_removal(test_profile)
    print_enforcement_summary(res)