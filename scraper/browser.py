import asyncio
import logging
from playwright.async_api import async_playwright, TimeoutError as PlaywrightTimeout

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

DEFAULT_TIMEOUT = 15000

# ─────────────────────────────────────────────────────────────────────────────
# OPT-OUT HANDLERS — one per scraped broker (Addresses, Intelius, ZabaSearch)
# ─────────────────────────────────────────────────────────────────────────────

async def optout_addresses(page, user_profile: dict) -> dict:
    try:
        await page.goto("https://www.addresses.com/optout", timeout=DEFAULT_TIMEOUT)
        await page.wait_for_load_state("networkidle")

        name_parts = user_profile.get("name", "").split()
        first_name = name_parts[0] if name_parts else ""
        last_name  = name_parts[-1] if len(name_parts) > 1 else ""

        await page.fill('input[name="firstName"], input[placeholder*="First"]', first_name)
        await page.fill('input[name="lastName"], input[placeholder*="Last"]', last_name)

        email = user_profile.get("email", "")
        if email:
            await page.fill('input[type="email"], input[name="email"]', email)

        await page.click('button[type="submit"], input[type="submit"]')
        await page.wait_for_timeout(2000)

        return {
            "success": True,
            "message": "Addresses.com opt-out form submitted.",
            "manual_step": "Check your email for a confirmation link from Addresses.com."
        }

    except PlaywrightTimeout:
        return {"success": False, "message": "Timed out — Addresses.com page may have changed."}
    except Exception as e:
        return {"success": False, "message": f"Error: {str(e)}"}


async def optout_intelius(page, user_profile: dict) -> dict:
    try:
        await page.goto("https://www.intelius.com/opt-out", timeout=DEFAULT_TIMEOUT)
        await page.wait_for_load_state("networkidle")

        name_parts = user_profile.get("name", "").split()
        first_name = name_parts[0] if name_parts else ""
        last_name  = name_parts[-1] if len(name_parts) > 1 else ""

        await page.fill('input[name="firstName"], input[id="firstName"]', first_name)
        await page.fill('input[name="lastName"], input[id="lastName"]', last_name)

        state = user_profile.get("state", "")
        if state:
            try:
                await page.select_option('select[name="state"]', state)
            except Exception:
                pass

        await page.click('button[type="submit"], input[type="submit"]')
        await page.wait_for_timeout(3000)

        return {
            "success": True,
            "message": "Intelius opt-out form submitted.",
            "manual_step": "Select your record and enter your email for confirmation."
        }

    except Exception as e:
        return {"success": False, "message": f"Error: {str(e)}"}


async def optout_zabasearch(page, user_profile: dict) -> dict:
    try:
        await page.goto("https://www.zabasearch.com/block_user/", timeout=DEFAULT_TIMEOUT)
        await page.wait_for_load_state("networkidle")

        name_parts = user_profile.get("name", "").split()
        first_name = name_parts[0] if name_parts else ""
        last_name  = name_parts[-1] if len(name_parts) > 1 else ""

        await page.fill('input[name="fname"], input[placeholder*="First"]', first_name)
        await page.fill('input[name="lname"], input[placeholder*="Last"]', last_name)

        email = user_profile.get("email", "")
        if email:
            await page.fill('input[type="email"], input[name="email"]', email)

        await page.click('button[type="submit"], input[type="submit"]')
        await page.wait_for_timeout(2000)

        return {
            "success": True,
            "message": "ZabaSearch block request submitted.",
            "manual_step": "Check your email for a confirmation from ZabaSearch."
        }

    except PlaywrightTimeout:
        return {"success": False, "message": "Timed out — ZabaSearch page may have changed."}
    except Exception as e:
        return {"success": False, "message": f"Error: {str(e)}"}


# ─────────────────────────────────────────────────────────────────────────────
# BROKER → HANDLER MAPPING
# ─────────────────────────────────────────────────────────────────────────────
OPTOUT_HANDLERS = {
    "Addresses":  optout_addresses,
    "Intelius":   optout_intelius,
    "ZabaSearch": optout_zabasearch,
}


# ─────────────────────────────────────────────────────────────────────────────
# Run opt-out for a single broker
# ─────────────────────────────────────────────────────────────────────────────
async def run_optout(broker_name: str, user_profile: dict, headless: bool = True) -> dict:
    handler = OPTOUT_HANDLERS.get(broker_name)

    if not handler:
        return {
            "success": False,
            "message": f"No automated handler for '{broker_name}'.",
            "manual_step": f"Visit {broker_name}'s website manually to request removal."
        }

    logger.info(f"\n🤖 Starting automated opt-out for: {broker_name}")

    async with async_playwright() as playwright:
        browser = await playwright.chromium.launch(headless=headless)
        context = await browser.new_context(
            viewport={"width": 1280, "height": 800},
            user_agent=(
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/124.0.0.0 Safari/537.36"
            ),
            accept_downloads=True,
        )

        page = await context.new_page()
        result = await handler(page, user_profile)
        await browser.close()

    logger.info(f"  Result: {result['message']}")
    return result


# ─────────────────────────────────────────────────────────────────────────────
# Run opt-out for all records (batch)
# ─────────────────────────────────────────────────────────────────────────────
async def run_all_optouts(records: list[dict], user_profile: dict, headless: bool = True) -> list[dict]:
    for record in records:
        broker_name = record.get("broker", "")
        result = await run_optout(broker_name, user_profile, headless)
        record["optout_result"] = result
        # Pause between brokers to avoid detection
        await asyncio.sleep(2)

    successful = sum(1 for r in records if r.get("optout_result", {}).get("success"))
    logger.info(f"\n✅ Batch complete: {successful}/{len(records)} opt-outs successful/initiated")
    return records


# ─────────────────────────────────────────────────────────────────────────────
# Sync wrappers — called from main.py (non-async)
# ─────────────────────────────────────────────────────────────────────────────
def run_optout_sync(broker_name: str, user_profile: dict, headless: bool = True) -> dict:
    return asyncio.run(run_optout(broker_name, user_profile, headless))


def run_all_optouts_sync(records: list[dict], user_profile: dict, headless: bool = True) -> list[dict]:
    return asyncio.run(run_all_optouts(records, user_profile, headless))
