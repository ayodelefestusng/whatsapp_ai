"""
banking_tools_updated.py
────────────────────────────────────────────────────────────────────────────────
Amended / new LangChain tool definitions.

Replaces / extends banking_tools.py with:
  1.  create_customer_profile_tool  – now stores NIN + VFD account number,
                                       generates a single-use password-setup link.
  2.  authenticate_customer_tool    – password gate for every banking session.
  3.  evaluate_loan_eligibility_tool – credit bureau + social media + AI eval.

Convention mirrors the rest of the codebase:
  @tool("name", args_schema=Model)
  def fn(runtime: ToolRuntime[Context], **kwargs) -> str
"""

import hashlib
import json
import os
import uuid
from datetime import datetime, timedelta, timezone as dt_timezone

import requests
from langchain.tools import tool, ToolRuntime
from pydantic import BaseModel, Field
from sqlalchemy import create_engine, text

from .logger_utils import log_info, log_error, log_warning
from .base import Context

# ──────────────────────────────────────────────────────────────────────────────
# RE-USE CONFIG FROM banking_tools.py
# ──────────────────────────────────────────────────────────────────────────────

WALLET_BASE_URL   = os.getenv("VFD_WALLET_BASE_URL",
    "https://api-devapps.vfdbank.systems/vtech-wallet/api/v2/wallet2")
AUTH_URL          = os.getenv("VFD_AUTH_URL",
    "https://api-devapps.vfdbank.systems/vfd-tech/baas-portal/v1.1/baasauth/token")
CONSUMER_KEY      = os.getenv("VFD_CONSUMER_KEY",    "mL1dqaMcB760EP3fR18Vc23qUSZy")
CONSUMER_SECRET   = os.getenv("VFD_CONSUMER_SECRET", "ohAWPpabbj0UmMppmOgAFTazkjQt")
APP_BASE_URL      = os.getenv("APP_BASE_URL", "https://yourapp.com")   # public base URL

# Credit bureau – replace with your actual provider
CREDIT_BUREAU_URL = os.getenv("CREDIT_BUREAU_URL", "https://creditbureau.example.ng/api/v1")
CREDIT_BUREAU_KEY = os.getenv("CREDIT_BUREAU_API_KEY", "")

PASSWORD_SETUP_TOKEN_EXPIRY_HOURS = 24
PASSWORD_SETUP_PATH = "/banking/set-password"


# ──────────────────────────────────────────────────────────────────────────────
# PYDANTIC INPUT SCHEMAS
# ──────────────────────────────────────────────────────────────────────────────

class CustomerProfileInput(BaseModel):
    """Schema for create_customer_profile_tool."""
    first_name:    str = Field(..., description="Customer's first name")
    last_name:     str = Field(..., description="Customer's last name")
    email:         str = Field(..., description="Customer's email address")
    phone:         str = Field(..., description="Nigerian phone number (e.g. 08012345678)")
    gender:        str = Field(..., description="'male' or 'female'")
    date_of_birth: str = Field(..., description="Date of birth (YYYY-MM-DD)")
    nin:           str = Field(..., description="11-digit National Identification Number")
    occupation:    str = Field("Not Specified", description="Occupation")
    nationality:   str = Field("Nigeria", description="Nationality")


class AuthenticateCustomerInput(BaseModel):
    """Schema for authenticate_customer_tool."""
    phone_number: str = Field(..., description="Customer's registered phone number")
    password:     str = Field(..., description="Customer's service password")


class LoanEligibilityInput(BaseModel):
    """Schema for evaluate_loan_eligibility_tool."""
    phone_number:   str = Field(..., description="Customer's registered phone number")
    facebook_url:   str = Field("", description="Facebook profile URL (optional)")
    linkedin_url:   str = Field("", description="LinkedIn profile URL (optional)")
    instagram_url:  str = Field("", description="Instagram profile URL (optional)")
    twitter_url:    str = Field("", description="Twitter / X profile URL (optional)")
    tiktok_url:     str = Field("", description="TikTok profile URL (optional)")


# ──────────────────────────────────────────────────────────────────────────────
# PRIVATE HELPERS
# ──────────────────────────────────────────────────────────────────────────────

def _normalise_db_uri(uri: str) -> str:
    if uri and uri.startswith("postgres://"):
        return uri.replace("postgres://", "postgresql://", 1)
    return uri


def _get_access_token() -> str:
    resp = requests.post(
        AUTH_URL,
        json={"consumerKey": CONSUMER_KEY, "consumerSecret": CONSUMER_SECRET, "validityTime": "-1"},
        headers={"Content-Type": "application/json"},
        timeout=30,
    )
    data = resp.json()
    if data.get("status") == "00":
        return data["data"]["access_token"]
    raise RuntimeError(f"VFD auth failed: {data}")


def _hash_password(raw: str) -> str:
    """SHA-256 hash for the banking PIN (already used across this codebase).
    For the web-facing service password Django's make_password is used in the view;
    this helper is retained for PIN operations only."""
    return hashlib.sha256(raw.encode()).hexdigest()


def _get_customer_row(db_uri: str, phone_number: str):
    """Returns a row dict with keys: id, account_number, full_name, nin, password."""
    engine = create_engine(_normalise_db_uri(db_uri))
    try:
        with engine.connect() as conn:
            row = conn.execute(
                text("""
                    SELECT id, account_number,
                           first_name || ' ' || last_name AS full_name,
                           nin, password
                    FROM customer_customer
                    WHERE phone_number = :phone
                    LIMIT 1
                """),
                {"phone": phone_number},
            ).fetchone()
        if not row:
            return None
        return {
            "id":             row[0],
            "account_number": row[1],
            "full_name":      row[2],
            "nin":            row[3],
            "password":       row[4],
        }
    finally:
        engine.dispose()


def _create_password_token(db_uri: str, customer_id: int) -> str:
    """
    Inserts a new PasswordSetupToken row and returns the token UUID string.
    Expires after PASSWORD_SETUP_TOKEN_EXPIRY_HOURS.
    """
    token      = str(uuid.uuid4())
    expires_at = (
        datetime.now(tz=dt_timezone.utc)
        + timedelta(hours=PASSWORD_SETUP_TOKEN_EXPIRY_HOURS)
    ).isoformat()

    engine = create_engine(_normalise_db_uri(db_uri))
    try:
        with engine.connect() as conn:
            conn.execute(
                text("""
                    INSERT INTO customer_passwordsetuptoken
                        (token, customer_id, created_at, expires_at, is_used)
                    VALUES (:token, :cid, NOW(), :exp, FALSE)
                """),
                {"token": token, "cid": customer_id, "exp": expires_at},
            )
            conn.commit()
    finally:
        engine.dispose()

    return token


def _fetch_credit_bureau(nin: str, account_number: str) -> dict:
    """
    Calls your credit-bureau provider.
    Returns: { credit_rating, credit_score, reference }
    Replace the stub with the actual provider's request structure.
    """
    try:
        resp = requests.post(
            f"{CREDIT_BUREAU_URL}/enquiry",
            json={"nin": nin, "accountNumber": account_number},
            headers={
                "Authorization": f"Bearer {CREDIT_BUREAU_KEY}",
                "Content-Type": "application/json",
            },
            timeout=30,
        )
        data = resp.json()
        return {
            "credit_rating":           data.get("rating", ""),
            "credit_score":            data.get("score"),
            "credit_bureau_reference": data.get("reference", ""),
        }
    except Exception as exc:
        log_warning(f"Credit bureau lookup failed: {exc}")
        return {"credit_rating": "", "credit_score": None, "credit_bureau_reference": ""}


def _fetch_social_metrics(
    facebook_url: str,
    linkedin_url: str,
    instagram_url: str,
    twitter_url: str,
    tiktok_url: str,
) -> dict:
    """
    Stub: replace with real social-media API integrations.
    Returns a normalised metrics dict used by the AI evaluator.
    """
    # ── Facebook ──────────────────────────────────────────────────────────────
    # Use Facebook Graph API: GET /{page-id}?fields=followers_count,fan_count
    # ── LinkedIn ──────────────────────────────────────────────────────────────
    # LinkedIn does not expose public follower counts via API for personal
    # profiles; consider a scrape-safe approach or a 3rd-party enrichment API.
    # ── Instagram ─────────────────────────────────────────────────────────────
    # Use Instagram Basic Display API or Instagram Graph API (business accounts).
    # ── Twitter / X ───────────────────────────────────────────────────────────
    # Use X API v2: GET /2/users/by/username/:username?user.fields=public_metrics
    # ── TikTok ────────────────────────────────────────────────────────────────
    # Use TikTok Research API (requires developer approval).

    return {
        "facebook_followers":   0,
        "facebook_posts_30d":   0,
        "linkedin_connections": 0,
        "linkedin_posts_30d":   0,
        "instagram_followers":  0,
        "instagram_posts_30d":  0,
        "twitter_followers":    0,
        "twitter_tweets_30d":   0,
        "tiktok_followers":     0,
        "tiktok_videos_30d":    0,
    }


def _ai_evaluate_loan(credit: dict, social: dict, full_name: str) -> dict:
    """
    Calls the Gemini / configured LLM to produce a loan eligibility decision.
    Returns: { score, band, notes, raw_response }
    """
    from .llm_handler import get_llm_instance   # local import to avoid circular

    prompt = f"""
You are a financial risk analyst for a Nigerian digital bank.
Evaluate the loan eligibility of a customer named {full_name} based on the
following data. Return ONLY valid JSON with these keys:
  "score"  : integer 0–100
  "band"   : one of "excellent", "good", "fair", "poor"
  "notes"  : concise 2–3 sentence summary explaining the decision
  "flags"  : list of any risk flags (empty list if none)

Credit bureau data:
  Rating : {credit.get("credit_rating", "N/A")}
  Score  : {credit.get("credit_score", "N/A")}

Social media activity:
  Facebook  : {social.get("facebook_followers", 0)} followers, {social.get("facebook_posts_30d", 0)} posts/30d
  LinkedIn  : {social.get("linkedin_connections", 0)} connections, {social.get("linkedin_posts_30d", 0)} posts/30d
  Instagram : {social.get("instagram_followers", 0)} followers, {social.get("instagram_posts_30d", 0)} posts/30d
  Twitter/X : {social.get("twitter_followers", 0)} followers, {social.get("twitter_tweets_30d", 0)} tweets/30d
  TikTok    : {social.get("tiktok_followers", 0)} followers, {social.get("tiktok_videos_30d", 0)} videos/30d

Return JSON only. No markdown. No extra text.
""".strip()

    try:
        llm      = get_llm_instance()
        raw_text = llm.invoke(prompt).content.strip()
        raw_text = raw_text.replace("```json", "").replace("```", "").strip()
        result   = json.loads(raw_text)
        return {
            "score":        int(result.get("score", 0)),
            "band":         result.get("band", "poor"),
            "notes":        result.get("notes", ""),
            "flags":        result.get("flags", []),
            "raw_response": raw_text,
        }
    except Exception as exc:
        log_error(f"AI loan evaluation failed: {exc}")
        return {
            "score": 0, "band": "poor",
            "notes": "Evaluation could not be completed. Please try again.",
            "flags": ["evaluation_error"],
            "raw_response": str(exc),
        }


def _upsert_loan_profile(db_uri: str, customer_id: int, account_number: str,
                         credit: dict, social: dict, ai: dict) -> None:
    """Upserts the loan profile row in the DB."""
    engine = create_engine(_normalise_db_uri(db_uri))
    try:
        with engine.connect() as conn:
            conn.execute(
                text("""
                    INSERT INTO customer_loanprofile (
                        customer_id, account_number,
                        credit_rating, credit_score, credit_bureau_reference,
                        credit_bureau_last_checked,
                        facebook_followers, facebook_posts_30d,
                        linkedin_connections, linkedin_posts_30d,
                        instagram_followers, instagram_posts_30d,
                        twitter_followers, twitter_tweets_30d,
                        tiktok_followers, tiktok_videos_30d,
                        loan_eligibility_score, eligibility_band,
                        eligibility_notes, raw_ai_response,
                        last_evaluated
                    ) VALUES (
                        :cid, :acc,
                        :cr, :cs, :cbr,
                        NOW(),
                        :fb_fol, :fb_posts,
                        :li_conn, :li_posts,
                        :ig_fol, :ig_posts,
                        :tw_fol, :tw_tweets,
                        :tt_fol, :tt_videos,
                        :score, :band,
                        :notes, :raw,
                        NOW()
                    )
                    ON CONFLICT (customer_id)
                    DO UPDATE SET
                        credit_rating              = EXCLUDED.credit_rating,
                        credit_score               = EXCLUDED.credit_score,
                        credit_bureau_reference    = EXCLUDED.credit_bureau_reference,
                        credit_bureau_last_checked = NOW(),
                        facebook_followers         = EXCLUDED.facebook_followers,
                        facebook_posts_30d         = EXCLUDED.facebook_posts_30d,
                        linkedin_connections       = EXCLUDED.linkedin_connections,
                        linkedin_posts_30d         = EXCLUDED.linkedin_posts_30d,
                        instagram_followers        = EXCLUDED.instagram_followers,
                        instagram_posts_30d        = EXCLUDED.instagram_posts_30d,
                        twitter_followers          = EXCLUDED.twitter_followers,
                        twitter_tweets_30d         = EXCLUDED.twitter_tweets_30d,
                        tiktok_followers           = EXCLUDED.tiktok_followers,
                        tiktok_videos_30d          = EXCLUDED.tiktok_videos_30d,
                        loan_eligibility_score     = EXCLUDED.loan_eligibility_score,
                        eligibility_band           = EXCLUDED.eligibility_band,
                        eligibility_notes          = EXCLUDED.eligibility_notes,
                        raw_ai_response            = EXCLUDED.raw_ai_response,
                        last_evaluated             = NOW()
                """),
                {
                    "cid": customer_id, "acc": account_number,
                    "cr": credit.get("credit_rating", ""),
                    "cs": credit.get("credit_score"),
                    "cbr": credit.get("credit_bureau_reference", ""),
                    "fb_fol":   social["facebook_followers"],
                    "fb_posts": social["facebook_posts_30d"],
                    "li_conn":  social["linkedin_connections"],
                    "li_posts": social["linkedin_posts_30d"],
                    "ig_fol":   social["instagram_followers"],
                    "ig_posts": social["instagram_posts_30d"],
                    "tw_fol":   social["twitter_followers"],
                    "tw_tweets":social["twitter_tweets_30d"],
                    "tt_fol":   social["tiktok_followers"],
                    "tt_videos":social["tiktok_videos_30d"],
                    "score": ai["score"],
                    "band":  ai["band"],
                    "notes": ai["notes"],
                    "raw":   ai["raw_response"],
                },
            )
            conn.commit()
    finally:
        engine.dispose()


# ──────────────────────────────────────────────────────────────────────────────
# 1.  CREATE CUSTOMER PROFILE  (replaces version in tools.py)
# ──────────────────────────────────────────────────────────────────────────────

@tool("create_customer_profile_tool", args_schema=CustomerProfileInput)
def create_customer_profile_tool(runtime: ToolRuntime[Context], **kwargs) -> str:
    """
    Creates a new banking customer:
      1. Opens a VFD Bank account via the /client/tiers/individual API.
      2. Persists the customer record (including NIN + VFD account number) to the DB.
      3. Returns a single-use, time-limited password-creation link.
    """
    tenant_id       = runtime.context.tenant_id
    conversation_id = runtime.context.conversation_id
    db_uri          = runtime.context.db_uri

    first_name = kwargs["first_name"]
    last_name  = kwargs["last_name"]
    email      = kwargs["email"]
    phone      = kwargs["phone"]
    gender     = kwargs["gender"]
    dob_str    = kwargs["date_of_birth"]
    nin        = kwargs["nin"].strip()
    occupation = kwargs.get("occupation", "Not Specified")
    nationality= kwargs.get("nationality", "Nigeria")

    log_info(
        f"create_customer_profile_tool: {first_name} {last_name} / {phone}",
        tenant_id, conversation_id,
    )

    # ── Step 1: Open VFD Account ──────────────────────────────────────────────
    try:
        token   = _get_access_token()
        url     = f"{WALLET_BASE_URL}/client/tiers/individual"
        headers = {"AccessToken": token, "Content-Type": "application/json"}
        resp    = requests.post(
            url,
            params={"nin": nin, "dateOfBirth": dob_str},
            json={},
            headers=headers,
            timeout=30,
        )
        vfd_data = resp.json()
        log_info(
            f"VFD account opening status: {vfd_data.get('status')}",
            tenant_id, conversation_id,
        )

        if vfd_data.get("status") != "00":
            return (
                f"Account opening was unsuccessful: "
                f"{vfd_data.get('message', 'Unknown error')}. "
                "Please verify the NIN and date of birth and try again."
            )

        account_info   = vfd_data.get("data", {})
        account_number = (
            account_info.get("accountNumber")
            or account_info.get("account_number", "")
        )
        full_name = (
            account_info.get("fullName")
            or account_info.get("name", f"{first_name} {last_name}")
        )

    except Exception as exc:
        log_error(f"VFD API error: {exc}", tenant_id, conversation_id)
        return f"An error occurred while contacting VFD Bank: {exc}"

    # ── Step 2: Persist to tenant DB ─────────────────────────────────────────
    if not db_uri:
        return "Error: Database configuration missing."

    try:
        import random
        customer_id_str = f"CUST{random.randint(10000, 99999)}"

        engine = create_engine(_normalise_db_uri(db_uri))
        try:
            with engine.connect() as conn:
                # Resolve tenant PK
                t_row = conn.execute(
                    text("SELECT id FROM org_tenant WHERE code = :code"),
                    {"code": tenant_id},
                ).fetchone()
                if not t_row:
                    return f"Error: Tenant '{tenant_id}' not found in database."
                tenant_db_id = t_row[0]

                # Insert / update customer row
                result = conn.execute(
                    text("""
                        INSERT INTO customer_customer (
                            customer_id, first_name, last_name, email,
                            phone_number, account_number, gender,
                            nationality, occupation, date_of_birth,
                            nin, password, tenant_id
                        ) VALUES (
                            :cid, :fn, :ln, :email,
                            :phone, :acc, :gender,
                            :nat, :occ, :dob,
                            :nin, '', :tid
                        )
                        ON CONFLICT (phone_number)
                        DO UPDATE SET
                            account_number = EXCLUDED.account_number,
                            nin            = EXCLUDED.nin,
                            first_name     = EXCLUDED.first_name,
                            last_name      = EXCLUDED.last_name
                        RETURNING id
                    """),
                    {
                        "cid":   customer_id_str,
                        "fn":    first_name,
                        "ln":    last_name,
                        "email": email,
                        "phone": phone,
                        "acc":   account_number,
                        "gender":gender,
                        "nat":   nationality,
                        "occ":   occupation,
                        "dob":   dob_str,
                        "nin":   nin,
                        "tid":   tenant_db_id,
                    },
                )
                customer_db_id = result.fetchone()[0]
                conn.commit()
        finally:
            engine.dispose()

    except Exception as exc:
        log_error(f"DB persist error: {exc}", tenant_id, conversation_id)
        return f"Account created with VFD but failed to save locally: {exc}"

    # ── Step 3: Generate single-use password-setup link ───────────────────────
    try:
        setup_token = _create_password_token(db_uri, customer_db_id)
        setup_link  = f"{APP_BASE_URL}{PASSWORD_SETUP_PATH}/{setup_token}/"
    except Exception as exc:
        log_error(f"Token creation error: {exc}", tenant_id, conversation_id)
        setup_link = f"{APP_BASE_URL}{PASSWORD_SETUP_PATH}/"  # fallback (no token)

    return (
        f"🎉 Account successfully created!\n\n"
        f"  Account Number : {account_number}\n"
        f"  Bank           : VFD Microfinance Bank\n"
        f"  Account Name   : {full_name}\n\n"
        f"To complete your setup, please create your banking password using the "
        f"secure link below. This link is valid for {PASSWORD_SETUP_TOKEN_EXPIRY_HOURS} "
        f"hours and can only be used once:\n\n"
        f"🔐 {setup_link}\n\n"
        f"After creating your password, return to WhatsApp to access all banking services."
    )


# ──────────────────────────────────────────────────────────────────────────────
# 2.  AUTHENTICATE CUSTOMER  (new)
# ──────────────────────────────────────────────────────────────────────────────

@tool("authenticate_customer_tool", args_schema=AuthenticateCustomerInput)
def authenticate_customer_tool(runtime: ToolRuntime[Context], **kwargs) -> str:
    """
    Verifies a customer's service password before granting access to banking.
    Call this at the start of every banking session.

    Outcomes
      • No account found     → prompt to register.
      • No password set yet  → return a fresh single-use setup link.
      • Wrong password       → increment failure count, warn customer.
      • Correct password     → return confirmation and let the agent proceed.
    """
    tenant_id       = runtime.context.tenant_id
    conversation_id = runtime.context.conversation_id
    db_uri          = runtime.context.db_uri
    phone_number    = kwargs["phone_number"]
    raw_password    = kwargs["password"]

    log_info(
        f"authenticate_customer_tool: phone={phone_number}",
        tenant_id, conversation_id,
    )

    if not db_uri:
        return "Error: Database configuration missing."

    try:
        customer = _get_customer_row(db_uri, phone_number)

        # ── No account ────────────────────────────────────────────────────────
        if not customer:
            return (
                "No banking account was found for this number. "
                "Please use the *Open Account* option to register first."
            )

        # ── Password not yet created ──────────────────────────────────────────
        if not customer["password"]:
            try:
                setup_token = _create_password_token(db_uri, customer["id"])
                setup_link  = f"{APP_BASE_URL}{PASSWORD_SETUP_PATH}/{setup_token}/"
            except Exception:
                setup_link = f"{APP_BASE_URL}{PASSWORD_SETUP_PATH}/"

            return (
                f"Hi {customer['full_name']}, you haven't created a password yet.\n\n"
                f"Please set your banking password using the secure link below. "
                f"It is valid for {PASSWORD_SETUP_TOKEN_EXPIRY_HOURS} hours and "
                f"single-use:\n\n"
                f"🔐 {setup_link}\n\n"
                f"Return here once your password is created."
            )

        # ── Verify password (Django PBKDF2 check) ────────────────────────────
        from django.contrib.auth.hashers import check_password as django_check
        if not django_check(raw_password, customer["password"]):
            # Increment failure counter
            engine = create_engine(_normalise_db_uri(db_uri))
            try:
                with engine.connect() as conn:
                    conn.execute(
                        text("""
                            UPDATE customer_customer
                            SET failed_password_attempts =
                                COALESCE(failed_password_attempts, 0) + 1
                            WHERE phone_number = :phone
                        """),
                        {"phone": phone_number},
                    )
                    conn.commit()
            finally:
                engine.dispose()

            return (
                "❌ Incorrect password. Please check and try again. "
                "After 5 failed attempts your account will be locked."
            )

        # ── Success: reset counter ────────────────────────────────────────────
        engine = create_engine(_normalise_db_uri(db_uri))
        try:
            with engine.connect() as conn:
                conn.execute(
                    text("""
                        UPDATE customer_customer
                        SET failed_password_attempts = 0
                        WHERE phone_number = :phone
                    """),
                    {"phone": phone_number},
                )
                conn.commit()
        finally:
            engine.dispose()

        return (
            f"✅ Authentication successful. Welcome back, {customer['full_name']}! "
            f"Your account ({customer['account_number']}) is now unlocked for this session."
        )

    except Exception as exc:
        log_error(f"authenticate_customer_tool error: {exc}", tenant_id, conversation_id)
        return f"Authentication error: {exc}"


# ──────────────────────────────────────────────────────────────────────────────
# 3.  EVALUATE LOAN ELIGIBILITY  (new)
# ──────────────────────────────────────────────────────────────────────────────

@tool("evaluate_loan_eligibility_tool", args_schema=LoanEligibilityInput)
def evaluate_loan_eligibility_tool(runtime: ToolRuntime[Context], **kwargs) -> str:
    """
    Evaluates a customer's loan eligibility by:
      1. Checking whether stored data is older than 6 months.
      2. If stale (or absent): fetching credit bureau data + social media metrics.
      3. Running an AI analysis to produce an eligibility score and recommendation.
      4. Persisting results to LoanProfile.

    If cached data is still fresh (< 180 days) it returns the stored result immediately.
    """
    tenant_id       = runtime.context.tenant_id
    conversation_id = runtime.context.conversation_id
    db_uri          = runtime.context.db_uri
    phone_number    = kwargs["phone_number"]

    log_info(
        f"evaluate_loan_eligibility_tool: phone={phone_number}",
        tenant_id, conversation_id,
    )

    if not db_uri:
        return "Error: Database configuration missing."

    try:
        customer = _get_customer_row(db_uri, phone_number)
        if not customer:
            return (
                "No account found for this number. "
                "Please complete account opening first."
            )

        account_number = customer["account_number"]
        customer_db_id = customer["id"]
        nin            = customer["nin"]
        full_name      = customer["full_name"]

        # ── Check cached evaluation ────────────────────────────────────────────
        engine = create_engine(_normalise_db_uri(db_uri))
        try:
            with engine.connect() as conn:
                cached = conn.execute(
                    text("""
                        SELECT loan_eligibility_score, eligibility_band,
                               eligibility_notes, last_evaluated
                        FROM customer_loanprofile
                        WHERE customer_id = :cid
                        LIMIT 1
                    """),
                    {"cid": customer_db_id},
                ).fetchone()
        finally:
            engine.dispose()

        if cached and cached[3]:
            age_days = (
                datetime.now(tz=dt_timezone.utc)
                - cached[3].replace(tzinfo=dt_timezone.utc)
            ).days
            if age_days <= 180:
                return (
                    f"📊 Loan Eligibility Report (cached {age_days} days ago)\n\n"
                    f"  Account    : {account_number}\n"
                    f"  Name       : {full_name}\n"
                    f"  Score      : {cached[0]}/100\n"
                    f"  Band       : {cached[1].capitalize()}\n\n"
                    f"{cached[2]}\n\n"
                    f"ℹ️ Data is valid for {180 - age_days} more days."
                )

        log_info("Cache stale or absent – running full evaluation.", tenant_id, conversation_id)

        # ── Step 1: Credit bureau ─────────────────────────────────────────────
        credit = _fetch_credit_bureau(nin, account_number)
        log_info(
            f"Credit bureau: rating={credit['credit_rating']}, score={credit['credit_score']}",
            tenant_id, conversation_id,
        )

        # ── Step 2: Social media metrics ──────────────────────────────────────
        social = _fetch_social_metrics(
            facebook_url  = kwargs.get("facebook_url",  ""),
            linkedin_url  = kwargs.get("linkedin_url",  ""),
            instagram_url = kwargs.get("instagram_url", ""),
            twitter_url   = kwargs.get("twitter_url",   ""),
            tiktok_url    = kwargs.get("tiktok_url",    ""),
        )
        log_info(
            f"Social metrics gathered: {json.dumps(social)}",
            tenant_id, conversation_id,
        )

        # ── Step 3: AI evaluation ─────────────────────────────────────────────
        ai = _ai_evaluate_loan(credit, social, full_name)
        log_info(
            f"AI eval: score={ai['score']}, band={ai['band']}",
            tenant_id, conversation_id,
        )

        # ── Step 4: Persist ───────────────────────────────────────────────────
        _upsert_loan_profile(db_uri, customer_db_id, account_number, credit, social, ai)

        flag_lines = ""
        if ai.get("flags"):
            flag_lines = "\n⚠️ Risk flags: " + ", ".join(ai["flags"])

        return (
            f"📊 Loan Eligibility Report\n\n"
            f"  Account      : {account_number}\n"
            f"  Name         : {full_name}\n"
            f"  Credit Score : {credit.get('credit_score', 'N/A')} "
            f"({credit.get('credit_rating', 'N/A')})\n"
            f"  AI Score     : {ai['score']}/100\n"
            f"  Band         : {ai['band'].capitalize()}\n\n"
            f"{ai['notes']}"
            f"{flag_lines}"
        )

    except Exception as exc:
        log_error(
            f"evaluate_loan_eligibility_tool error: {exc}",
            tenant_id, conversation_id,
        )
        return f"An error occurred during loan evaluation: {exc}"


# ──────────────────────────────────────────────────────────────────────────────
# EXPORT  – append these to banking_tools list in banking_tools.py
# ──────────────────────────────────────────────────────────────────────────────

new_banking_tools = [
    create_customer_profile_tool,   # replaces the version in tools.py
    authenticate_customer_tool,
    evaluate_loan_eligibility_tool,
]
