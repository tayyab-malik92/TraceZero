import asyncio
import random
import re
from playwright.async_api import async_playwright, TimeoutError as PWTimeout
from bs4 import BeautifulSoup

# ── Random user agents ─────────────────────────────────────────────────────────
AGENTS = [
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/123.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:125.0) Gecko/20100101 Firefox/125.0",
]

# ── Regex helpers ──────────────────────────────────────────────────────────────
def extract_emails(text):
    return list(set(re.findall(r"[a-zA-Z0-9._%+\-]+@[a-zA-Z0-9.\-]+\.[a-zA-Z]{2,}", text)))

def extract_phones(text):
    raw = re.findall(r"(\+?1?\s?[\(\-\.]?\d{3}[\)\-\.\s]\s?\d{3}[\-\.\s]\d{4})", text)
    cleaned = []
    for p in raw:
        digits = re.sub(r"\D", "", p)
        if len(digits) in (10, 11):
            cleaned.append(p.strip())
    return list(set(cleaned))

def extract_ages(text):
    matches = re.findall(r"\b(\d{2})\s*(?:years?\s*old)?\b", text, re.IGNORECASE)
    return [m for m in matches if 18 <= int(m) <= 90]

def extract_addresses(text):
    results = re.findall(
        r"\d+\s+[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\s+(?:St|Ave|Rd|Blvd|Dr|Ln|Way|Ct|Pl|Cir|Hwy)\b",
        text
    )
    return list(set(results))

def build_extracted(raw_text, user_profile):
    """Run all regex extractors and return a structured dict."""
    extracted = {}
    full_name   = user_profile.get("name", "")
    name_tokens = full_name.lower().split()

    if any(t in raw_text.lower() for t in name_tokens):
        extracted["name"] = full_name

    emails = extract_emails(raw_text)
    if emails:
        extracted["emails"] = ", ".join(emails[:3])

    phones = extract_phones(raw_text)
    if phones:
        extracted["phones"] = ", ".join(phones[:3])

    ages = extract_ages(raw_text)
    if ages:
        extracted["age"] = ages[0]

    addrs = extract_addresses(raw_text)
    if addrs:
        extracted["addresses"] = " | ".join(addrs[:2])

    if user_profile.get("city") and user_profile["city"].lower() in raw_text.lower():
        extracted["city"] = user_profile["city"]

    if user_profile.get("state") and user_profile["state"].lower() in raw_text.lower():
        extracted["state"] = user_profile["state"]

    return extracted

# ── Broker definitions ─────────────────────────────────────────────────────────
# Each broker: name, url_template, optout info, and optional wait_selector
# (a CSS selector that must appear for the page to be considered loaded)
BROKERS = [
    {
        "name":         "Addresses",
        "url_template": "https://www.addresses.com/people/{first}+{last}/{state}",
        "optout_url":   "https://www.addresses.com/optout",
        "optout_email": "privacy@addresses.com",
        "optout_notes": "Submit removal request via their online form.",
        "min_chars":    400,
        "name_check":   True,
    },
    {
        "name":         "Intelius",
        "url_template": "https://www.intelius.com/people/{first}-{last}/?state={state_upper}",
        "optout_url":   "https://www.intelius.com/opt-out",
        "optout_email": "privacy@intelius.com",
        "optout_notes": "Submit opt-out form with email confirmation.",
        "min_chars":    400,
        "name_check":   True,
    },
    {
        "name":         "ZabaSearch",
        "url_template": "https://www.zabasearch.com/people/{first}+{last}/{state_full}/",
        "optout_url":   "https://www.zabasearch.com/block_user/",
        "optout_email": "privacy@zabasearch.com",
        "optout_notes": "Submit removal via zabasearch.com/block_user.",
        "min_chars":    400,
        "name_check":   True,
    },
]

# ── Single broker scrape (Playwright) ─────────────────────────────────────────
async def _scrape_one(browser, broker: dict, user_profile: dict) -> dict:
    name_parts = user_profile.get("name", "").split()
    first = name_parts[0]  if name_parts          else ""
    last  = name_parts[-1] if len(name_parts) > 1 else ""
    city  = user_profile.get("city",  "").replace(" ", "+")
    state = user_profile.get("state", "")

    base = {
        "broker":       broker["name"],
        "optout_url":   broker["optout_url"],
        "optout_email": broker.get("optout_email"),
        "optout_notes": broker.get("optout_notes", ""),
        "extracted_pii": {},
        "raw_text":     "",
    }

    if not first or not last:
        return {**base, "status": "skipped"}

    STATE_FULL = {
        "AL":"alabama","AK":"alaska","AZ":"arizona","AR":"arkansas","CA":"california",
        "CO":"colorado","CT":"connecticut","DE":"delaware","FL":"florida","GA":"georgia",
        "HI":"hawaii","ID":"idaho","IL":"illinois","IN":"indiana","IA":"iowa","KS":"kansas",
        "KY":"kentucky","LA":"louisiana","ME":"maine","MD":"maryland","MA":"massachusetts",
        "MI":"michigan","MN":"minnesota","MS":"mississippi","MO":"missouri","MT":"montana",
        "NE":"nebraska","NV":"nevada","NH":"new-hampshire","NJ":"new-jersey","NM":"new-mexico",
        "NY":"new-york","NC":"north-carolina","ND":"north-dakota","OH":"ohio","OK":"oklahoma",
        "OR":"oregon","PA":"pennsylvania","RI":"rhode-island","SC":"south-carolina",
        "SD":"south-dakota","TN":"tennessee","TX":"texas","UT":"utah","VT":"vermont",
        "VA":"virginia","WA":"washington","WV":"west-virginia","WI":"wisconsin","WY":"wyoming",
    }
    state_full = STATE_FULL.get(state.upper(), state.lower().replace(" ", "-"))

    url = broker["url_template"].format(
        first=first.lower(), last=last.lower(),
        city=city.lower(),
        state=state.lower(),
        state_upper=state.upper(),
        state_full=state_full,
    )
    base["url"] = url

    context = await browser.new_context(
        viewport={"width": 1280, "height": 800},
        user_agent=random.choice(AGENTS),
        locale="en-US",
        timezone_id="America/New_York",
        java_script_enabled=True,
    )

    # Block images, fonts, media — speeds up page load significantly
    await context.route(
        "**/*",
        lambda route: route.abort()
        if route.request.resource_type in ("image", "media", "font", "stylesheet")
        else route.continue_()
    )

    page = await context.new_page()

    # Extra headers that make us look like a real browser
    await page.set_extra_http_headers({
        "Accept":          "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
        "Accept-Language": "en-US,en;q=0.9",
        "Accept-Encoding": "gzip, deflate, br",
        "DNT":             "1",
        "Upgrade-Insecure-Requests": "1",
    })

    try:
        print(f"  >> {broker['name']} ...")
        await page.goto(url, timeout=20000, wait_until="domcontentloaded")

        # Random human-like delay after page loads
        await page.wait_for_timeout(random.randint(1500, 3000))

        # Scroll down slightly — some sites only render content after scroll
        await page.evaluate("window.scrollBy(0, 400)")
        await page.wait_for_timeout(1000)

        html     = await page.content()
        soup     = BeautifulSoup(html, "html.parser")

        # Remove script/style/nav tags before extracting text
        for tag in soup(["script", "style", "nav", "footer", "header"]):
            tag.decompose()

        raw_text = soup.get_text(separator=" ", strip=True)[:4000]

        # Check for bot-block signals
        block_signals = [
            "cf-browser-verification", "captcha", "access denied",
            "unusual traffic", "verify you are human", "robot",
            "please enable javascript", "enable cookies",
        ]
        is_blocked = any(s in raw_text.lower() for s in block_signals)

        if is_blocked:
            print(f"     BLOCKED ({broker['name']})")
            await context.close()
            return {**base, "status": "blocked"}

        enough_text  = len(raw_text) > broker.get("min_chars", 200)
        name_parts   = user_profile.get("name", "").lower().split()
        name_on_page = any(p in raw_text.lower() for p in name_parts if len(p) > 2)
        found = enough_text and (not broker.get("name_check", False) or name_on_page)

        if found:
            extracted = build_extracted(raw_text, user_profile)
            print(f"     OK - {len(raw_text)} chars ({broker['name']})")
            await context.close()
            return {
                **base,
                "status":        "scraped",
                "raw_text":      raw_text,
                "extracted_pii": extracted,
            }
        else:
            reason = "name not on page" if enough_text and not name_on_page else f"{len(raw_text)} chars"
            print(f"     No data ({broker['name']}) - {reason}")
            await context.close()
            return {**base, "status": "no_data", "raw_text": raw_text}

    except PWTimeout:
        print(f"     TIMEOUT ({broker['name']})")
        await context.close()
        return {**base, "status": "blocked"}
    except Exception as e:
        print(f"     ERROR ({broker['name']}): {e}")
        try:
            await context.close()
        except Exception:
            pass
        return {**base, "status": "blocked"}


# ── Run all brokers ────────────────────────────────────────────────────────────
async def _scrape_all(user_profile: dict) -> list:
    print(f"\nScanning: {user_profile.get('name', '?')}")
    results = []

    async with async_playwright() as pw:
        browser = await pw.chromium.launch(
            headless=True,
            args=[
                "--no-sandbox",
                "--disable-blink-features=AutomationControlled",
                "--disable-infobars",
                "--disable-dev-shm-usage",
            ],
        )

        for broker in BROKERS:
            result = await _scrape_one(browser, broker, user_profile)
            results.append(result)
            # Pause between brokers — looks more human
            await asyncio.sleep(random.uniform(1.5, 3.0))

        await browser.close()

    scraped = sum(1 for r in results if r["status"] == "scraped")
    blocked = sum(1 for r in results if r["status"] == "blocked")
    no_data = sum(1 for r in results if r["status"] == "no_data")
    print(f"\nDone: {scraped} scraped, {blocked} blocked, {no_data} no data")
    return results


# ── Public sync wrapper (called from main.py) ──────────────────────────────────
def scrape_brokers(user_profile: dict) -> list:
    return asyncio.run(_scrape_all(user_profile))
