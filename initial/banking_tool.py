"""
banking_tools.py
────────────────────────────────────────────────────────────────────────────────
VFD Bank – LangChain tool definitions for the chatbot agent.

All API orchestration is handled here.  The agent (and therefore the customer)
never sees internal concepts such as billerId, divisionId, productId, paymentCode
or SHA-512 signatures.  Those are resolved silently inside each tool function.

Covered services
────────────────
  1.  Account Opening          – create_vfd_account_tool
  2.  Fund Wallet              – fund_wallet_info_tool
  3.  Balance Enquiry          – balance_enquiry_tool
  4.  Airtime Purchase         – buy_airtime_tool
  5.  Bills Payment            – pay_bill_tool  (intelligent biller resolution)
  6.  Transfer Money           – transfer_money_tool
  7.  Change PIN               – change_pin_tool
  8.  Forgot PIN               – forgot_pin_tool
  9.  Saved Billers            – get_saved_billers_tool / save_biller_tool
 10.  Bank List                – get_bank_list_tool   (helper exposed to agent)
"""

import hashlib
import os
import uuid
from datetime import datetime
from typing import Optional

import requests
from langchain.tools import tool
from sqlalchemy import text

from database import SessionLocal
from logger_utils import log_error, log_info

# ──────────────────────────────────────────────────────────────────────────────
# CONFIGURATION
# ──────────────────────────────────────────────────────────────────────────────

WALLET_BASE_URL = os.getenv(
    "VFD_WALLET_BASE_URL",
    "https://api-devapps.vfdbank.systems/vtech-wallet/api/v2/wallet2",
)
BILLS_BASE_URL = os.getenv(
    "VFD_BILLS_BASE_URL",
    "https://api-devapps.vfdbank.systems/vtech-bills/api/v2/billspaymentstore",
)
AUTH_URL = os.getenv(
    "VFD_AUTH_URL",
    "https://api-devapps.vfdbank.systems/vfd-tech/baas-portal/v1.1/baasauth/token",
)
CONSUMER_KEY    = os.getenv("VFD_CONSUMER_KEY",    "mL1dqaMcB760EP3fR18Vc23qUSZy")
CONSUMER_SECRET = os.getenv("VFD_CONSUMER_SECRET", "ohAWPpabbj0UmMppmOgAFTazkjQt")
WALLET_PREFIX   = os.getenv("VFD_WALLET_PREFIX",   "rosapay")

# Reference-label mapping so the agent always uses human-readable labels
CATEGORY_REFERENCE_LABEL: dict[str, str] = {
    "utility":               "Meter Number",
    "cable tv":              "Smart Card Number",
    "airtime":               "Phone Number",
    "data":                  "Phone Number",
    "internet subscription": "Account Number / Username",
}

# Categories that require mandatory customer validation before payment
MANDATORY_VALIDATE_CATEGORIES = {"utility", "cable tv", "betting", "gaming"}


# ──────────────────────────────────────────────────────────────────────────────
# INTERNAL HELPERS
# ──────────────────────────────────────────────────────────────────────────────

def _get_access_token() -> str:
    """Fetch a short-lived VFD access token."""
    payload = {
        "consumerKey":    CONSUMER_KEY,
        "consumerSecret": CONSUMER_SECRET,
        "validityTime":   "-1",
    }
    resp = requests.post(AUTH_URL, json=payload, headers={"Content-Type": "application/json"}, timeout=30)
    data = resp.json()
    if data.get("status") == "00":
        return data["data"]["access_token"]
    raise RuntimeError(f"VFD auth failed: {data}")


def _wallet_headers() -> dict:
    return {"AccessToken": _get_access_token(), "Content-Type": "application/json"}


def _unique_ref() -> str:
    """Generate a unique, prefixed transaction reference."""
    ts  = datetime.utcnow().strftime("%Y%m%d%H%M%S")
    uid = uuid.uuid4().hex[:6].upper()
    return f"{WALLET_PREFIX}-{ts}-{uid}"


def _resolve_biller(biller_name: str) -> dict:
    """
    Silently resolve a human biller name → all internal biller parameters.

    Returns a dict with keys:
        billerId, divisionId, productId, paymentCode,
        isAmountFixed, fixedAmount, convenienceFee, category
    Raises ValueError if the biller cannot be matched.
    """
    # 1. Fetch all billers (no category filter → broadest match)
    resp = requests.get(f"{BILLS_BASE_URL}/billerlist", timeout=20)
    billers = resp.json().get("data", [])

    name_lower = biller_name.strip().lower()

    # Simple contains match – good enough for most billers.
    # For production, replace with fuzzy/embedding-based matching.
    matched = None
    for b in billers:
        if name_lower in b.get("name", "").lower() or name_lower in b.get("id", "").lower():
            matched = b
            break

    if not matched:
        raise ValueError(
            f"Biller '{biller_name}' not found. "
            "Please check the name and try again."
        )

    biller_id   = matched["id"]
    division_id = matched["division"]
    product_id  = matched["product"]
    category    = matched.get("category", "").lower()
    convenience = matched.get("convenienceFee", "0")

    # 2. Fetch biller items to get paymentCode
    items_resp = requests.get(
        f"{BILLS_BASE_URL}/billerItems",
        params={"billerId": biller_id, "divisionId": division_id, "productId": product_id},
        timeout=20,
    )
    items_data = items_resp.json().get("data", {})
    payment_items = items_data.get("paymentitems", [])

    if not payment_items:
        raise ValueError(f"No payment items found for biller '{biller_name}'.")

    item        = payment_items[0]           # Take first / primary item
    payment_code     = item.get("paymentCode", "")
    is_fixed    = item.get("isAmountFixed", "false").lower() == "true"
    fixed_amount = item.get("amount", "0") if is_fixed else None

    return {
        "billerId":       biller_id,
        "divisionId":     division_id,
        "productId":      product_id,
        "paymentCode":    payment_code,
        "isAmountFixed":  is_fixed,
        "fixedAmount":    fixed_amount,
        "convenienceFee": convenience,
        "category":       category,
    }


def _validate_customer(biller_info: dict, customer_id: str) -> bool:
    """
    Run mandatory customer validation for utility / cable TV billers.
    Returns True on success; raises ValueError on failure.
    """
    params = {
        "divisionId":  biller_info["divisionId"],
        "paymentItem": biller_info["paymentCode"],
        "customerId":  customer_id,
        "billerId":    biller_info["billerId"],
    }
    resp = requests.get(f"{BILLS_BASE_URL}/customervalidate", params=params, timeout=20)
    data = resp.json()
    if data.get("status") == "00":
        return True
    raise ValueError(
        f"Reference number validation failed: {data.get('message', 'Invalid reference')}. "
        "Please check your meter/smart card number and try again."
    )


def _get_customer_account(phone_number: str) -> dict:
    """
    Look up the VFD account number for the given phone number from local DB.
    Returns {"accountNumber": "...", "accountName": "..."}.
    """
    with SessionLocal() as session:
        sql = """
            SELECT account_number, full_name
            FROM banking_customer_profile
            WHERE phone_number = :phone
            LIMIT 1
        """
        row = session.execute(text(sql), {"phone": phone_number}).fetchone()
        if not row:
            raise ValueError(
                "No banking profile found for this phone number. "
                "Please complete account opening first."
            )
        return {"accountNumber": row[0], "accountName": row[1]}


def _verify_pin(phone_number: str, pin: str) -> bool:
    """
    Verify a 4-digit PIN against the stored (hashed) PIN for the customer.
    """
    hashed = hashlib.sha256(pin.encode()).hexdigest()
    with SessionLocal() as session:
        sql = """
            SELECT 1 FROM banking_customer_profile
            WHERE phone_number = :phone AND pin_hash = :pin_hash
            LIMIT 1
        """
        row = session.execute(text(sql), {"phone": phone_number, "pin_hash": hashed}).fetchone()
        return row is not None


def _increment_pin_attempts(phone_number: str) -> int:
    """Increment and return the failed PIN attempt counter."""
    with SessionLocal() as session:
        session.execute(
            text("""
                UPDATE banking_customer_profile
                SET failed_pin_attempts = COALESCE(failed_pin_attempts, 0) + 1
                WHERE phone_number = :phone
            """),
            {"phone": phone_number},
        )
        session.commit()
        row = session.execute(
            text("SELECT failed_pin_attempts FROM banking_customer_profile WHERE phone_number = :phone"),
            {"phone": phone_number},
        ).fetchone()
        return row[0] if row else 1


def _reset_pin_attempts(phone_number: str):
    with SessionLocal() as session:
        session.execute(
            text("UPDATE banking_customer_profile SET failed_pin_attempts = 0 WHERE phone_number = :phone"),
            {"phone": phone_number},
        )
        session.commit()


# ──────────────────────────────────────────────────────────────────────────────
# 1. ACCOUNT OPENING
# ──────────────────────────────────────────────────────────────────────────────

@tool
def create_vfd_account_tool(nin: str, date_of_birth: str, phone_number: str) -> str:
    """
    Open a new VFD bank account using the customer's NIN and date of birth.

    Args:
        nin:           11-digit National Identification Number.
        date_of_birth: Date of birth in YYYY-MM-DD format (e.g. 1994-04-05).
        phone_number:  Customer's registered phone number (used to link profile).

    Returns:
        A message containing the new account number, bank name, and a link
        for the customer to create their 4-digit PIN.
    """
    try:
        token = _get_access_token()
        url   = f"{WALLET_BASE_URL}/client/tiers/individual"
        params = {"nin": nin, "dateOfBirth": date_of_birth}
        headers = {"AccessToken": token, "Content-Type": "application/json"}

        resp = requests.post(url, params=params, json={}, headers=headers, timeout=30)
        data = resp.json()

        if data.get("status") != "00":
            return (
                f"Account opening was unsuccessful: {data.get('message', 'Unknown error')}. "
                "Please verify your NIN and date of birth and try again."
            )

        account_info = data.get("data", {})
        account_number = account_info.get("accountNumber") or account_info.get("account_number", "N/A")
        full_name      = account_info.get("fullName") or account_info.get("name", "")

        # Persist to local DB
        with SessionLocal() as session:
            session.execute(
                text("""
                    INSERT INTO banking_customer_profile
                        (phone_number, account_number, full_name, failed_pin_attempts)
                    VALUES (:phone, :acct, :name, 0)
                    ON CONFLICT (phone_number) DO UPDATE
                        SET account_number = EXCLUDED.account_number,
                            full_name      = EXCLUDED.full_name
                """),
                {"phone": phone_number, "acct": account_number, "name": full_name},
            )
            session.commit()

        create_pin_link = f"https://yourapp.com/banking/create-pin?phone={phone_number}"

        return (
            f"🎉 Account successfully created!\n\n"
            f"  Account Number : {account_number}\n"
            f"  Bank           : VFD Bank\n"
            f"  Account Name   : {full_name}\n\n"
            f"To complete your setup and access all banking services, please create your "
            f"4-digit PIN using the link below:\n👉 {create_pin_link}"
        )

    except Exception as exc:
        log_error(f"create_vfd_account_tool error: {exc}")
        return f"An error occurred during account opening: {exc}"


# ──────────────────────────────────────────────────────────────────────────────
# 2. FUND WALLET
# ──────────────────────────────────────────────────────────────────────────────

@tool
def fund_wallet_info_tool(phone_number: str) -> str:
    """
    Return the VFD bank account details the customer should use to fund their wallet.
    No PIN is required for this informational service.

    Args:
        phone_number: Customer's registered phone number.
    """
    try:
        profile = _get_customer_account(phone_number)
        return (
            f"To fund your wallet, make a transfer to the following account:\n\n"
            f"  Account Number : {profile['accountNumber']}\n"
            f"  Bank           : VFD Bank\n"
            f"  Account Name   : {profile['accountName']}\n\n"
            f"Available funding channels:\n"
            f"  • Mobile Banking App (any Nigerian bank)\n"
            f"  • USSD transfer\n"
            f"  • Bank transfer / Internet banking\n"
            f"  • POS or ATM deposit\n\n"
            f"Funds are usually credited within minutes. "
            f"Please contact support if your balance is not updated after 10 minutes."
        )
    except Exception as exc:
        return f"Unable to retrieve wallet details: {exc}"


# ──────────────────────────────────────────────────────────────────────────────
# 3. BALANCE ENQUIRY
# ──────────────────────────────────────────────────────────────────────────────

@tool
def balance_enquiry_tool(phone_number: str, pin: str) -> str:
    """
    Return the current wallet balance for the authenticated customer.
    PIN is required.

    Args:
        phone_number: Customer's registered phone number.
        pin:          4-digit numeric PIN.
    """
    try:
        # PIN verification
        if not _verify_pin(phone_number, pin):
            attempts = _increment_pin_attempts(phone_number)
            remaining = max(0, 5 - attempts)
            if remaining == 0:
                return (
                    "Your account has been locked due to too many incorrect PIN attempts. "
                    "Please use the 'Forgot PIN' option to reset your PIN."
                )
            return (
                f"Incorrect PIN. You have {remaining} attempt(s) remaining before your account is locked."
            )

        _reset_pin_attempts(phone_number)
        profile = _get_customer_account(phone_number)

        # Call VFD balance endpoint
        headers = _wallet_headers()
        resp = requests.get(
            f"{WALLET_BASE_URL}/account/enquiry",
            params={"accountNumber": profile["accountNumber"]},
            headers=headers,
            timeout=20,
        )
        data = resp.json()

        if data.get("status") != "00":
            return f"Balance enquiry failed: {data.get('message', 'Unknown error')}."

        balance = data.get("data", {}).get("balance", "N/A")
        return (
            f"💰 Account Balance\n\n"
            f"  Account : {profile['accountNumber']} (VFD Bank)\n"
            f"  Name    : {profile['accountName']}\n"
            f"  Balance : ₦{balance}"
        )

    except Exception as exc:
        log_error(f"balance_enquiry_tool error: {exc}")
        return f"An error occurred during balance enquiry: {exc}"


# ──────────────────────────────────────────────────────────────────────────────
# 4. AIRTIME PURCHASE
# ──────────────────────────────────────────────────────────────────────────────

@tool
def buy_airtime_tool(
    phone_number: str,
    network: str,
    amount: str,
    recipient_type: str = "self",
    beneficiary_phone: Optional[str] = None,
) -> str:
    """
    Purchase airtime for the customer (self) or a third party.

    Args:
        phone_number:     Customer's registered phone number.
        network:          Telecom network name (e.g. MTN, Airtel, Glo, 9mobile).
        amount:           Amount in Naira (e.g. "500").
        recipient_type:   "self" or "third_party".
        beneficiary_phone: Required only when recipient_type is "third_party".
    """
    try:
        target_phone = phone_number if recipient_type == "self" else beneficiary_phone
        if not target_phone:
            return "Please provide the beneficiary's phone number for a third-party airtime purchase."

        # Silently resolve network → biller params via bills API
        biller_info = _resolve_biller(network)

        # No customer validation needed for airtime
        reference = _unique_ref()
        payload = {
            "customerId":  target_phone,
            "amount":      amount,
            "division":    biller_info["divisionId"],
            "paymentItem": biller_info["paymentCode"],
            "productId":   biller_info["productId"],
            "billerId":    biller_info["billerId"],
            "reference":   reference,
            "phoneNumber": phone_number,
        }

        resp = requests.post(f"{BILLS_BASE_URL}/pay", json=payload, timeout=30)
        data = resp.json()

        if data.get("status") != "00":
            return f"Airtime purchase failed: {data.get('message', 'Unknown error')}. Please try again."

        label = "your number" if recipient_type == "self" else target_phone
        return (
            f"✅ Airtime Purchase Successful!\n\n"
            f"  Network    : {network.upper()}\n"
            f"  Amount     : ₦{amount}\n"
            f"  Recipient  : {label}\n"
            f"  Reference  : {reference}"
        )

    except Exception as exc:
        log_error(f"buy_airtime_tool error: {exc}")
        return f"An error occurred while purchasing airtime: {exc}"


# ──────────────────────────────────────────────────────────────────────────────
# 5. BILLS PAYMENT  (intelligent – customer only provides biller + amount + ref)
# ──────────────────────────────────────────────────────────────────────────────

@tool
def pay_bill_tool(
    phone_number: str,
    biller_name: str,
    reference_number: str,
    amount: str,
) -> str:
    """
    Pay a utility, cable TV, internet, or other bill.
    The customer provides only the biller name, their reference number
    (meter number / smart card number / account number), and the amount.
    All API orchestration is handled internally.

    Args:
        phone_number:     Customer's registered phone number.
        biller_name:      Human-readable biller name (e.g. DSTV, EKEDC, MTN).
        reference_number: Customer's reference – meter number, smart card, etc.
        amount:           Amount to pay in Naira.
    """
    try:
        # Step 1 – resolve biller silently
        biller_info = _resolve_biller(biller_name)
        category    = biller_info["category"]

        # Step 2 – mandatory validation for utility / cable TV
        if category in MANDATORY_VALIDATE_CATEGORIES:
            _validate_customer(biller_info, reference_number)

        # Step 3 – use fixed amount if biller dictates it
        pay_amount = biller_info["fixedAmount"] if biller_info["isAmountFixed"] else amount

        # Step 4 – build and fire the payment
        reference = _unique_ref()
        payload = {
            "customerId":  reference_number,
            "amount":      pay_amount,
            "division":    biller_info["divisionId"],
            "paymentItem": biller_info["paymentCode"],
            "productId":   biller_info["productId"],
            "billerId":    biller_info["billerId"],
            "reference":   reference,
            "phoneNumber": phone_number,
        }

        resp = requests.post(f"{BILLS_BASE_URL}/pay", json=payload, timeout=30)
        data = resp.json()

        if data.get("status") != "00":
            return f"Bill payment failed: {data.get('message', 'Unknown error')}. Please try again."

        # Step 5 – persist as a saved biller for future quick-pay
        _upsert_saved_biller(
            phone_number=phone_number,
            biller_name=biller_name,
            biller_info=biller_info,
            reference_number=reference_number,
        )

        ref_label = CATEGORY_REFERENCE_LABEL.get(category, "Reference")
        convenience = biller_info.get("convenienceFee", "0")
        fee_line = f"  Convenience Fee : ₦{convenience}\n" if convenience and convenience != "0" else ""

        return (
            f"✅ Bill Payment Successful!\n\n"
            f"  Biller     : {biller_name.upper()}\n"
            f"  {ref_label:<13}: {reference_number}\n"
            f"  Amount     : ₦{pay_amount}\n"
            f"{fee_line}"
            f"  Reference  : {reference}"
        )

    except Exception as exc:
        log_error(f"pay_bill_tool error: {exc}")
        return f"An error occurred during bill payment: {exc}"


# ──────────────────────────────────────────────────────────────────────────────
# 6. TRANSFER MONEY
# ──────────────────────────────────────────────────────────────────────────────

@tool
def transfer_money_tool(
    phone_number: str,
    beneficiary_account_number: str,
    beneficiary_bank: str,
    amount: str,
    pin: str,
    narration: Optional[str] = "",
) -> str:
    """
    Transfer funds from the customer's wallet to any Nigerian bank account.
    The agent collects beneficiary details, confirms the account name with the
    customer, then calls this tool to execute the transfer.

    Args:
        phone_number:               Sender's registered phone number.
        beneficiary_account_number: Recipient's account number.
        beneficiary_bank:           Recipient's bank name or code.
        amount:                     Transfer amount in Naira.
        pin:                        Sender's 4-digit PIN.
        narration:                  Optional transfer remark.
    """
    try:
        # ── PIN verification ──────────────────────────────────────────────────
        if not _verify_pin(phone_number, pin):
            attempts = _increment_pin_attempts(phone_number)
            remaining = max(0, 5 - attempts)
            if remaining == 0:
                return (
                    "Your account has been locked due to too many incorrect PIN attempts. "
                    "Please use the 'Forgot PIN' option to reset your PIN."
                )
            return f"Incorrect PIN. You have {remaining} attempt(s) remaining."

        _reset_pin_attempts(phone_number)
        headers = _wallet_headers()

        # ── Step 1: Sender account details ───────────────────────────────────
        sender = _get_customer_account(phone_number)
        sender_resp = requests.get(
            f"{WALLET_BASE_URL}/account/enquiry",
            params={"accountNumber": sender["accountNumber"]},
            headers=headers,
            timeout=20,
        )
        sender_data = sender_resp.json().get("data", {})
        from_account = sender_data.get("accountNumber", sender["accountNumber"])

        # ── Step 2: Bank list – resolve bank name → bank code ────────────────
        banks_resp = requests.get(f"{WALLET_BASE_URL}/bank", headers=headers, timeout=20)
        banks      = banks_resp.json().get("data", [])
        bank_code  = None
        bname_lower = beneficiary_bank.strip().lower()
        for bank in banks:
            if bname_lower in bank.get("name", "").lower() or bname_lower == bank.get("code", "").lower():
                bank_code = bank.get("code")
                break

        if not bank_code:
            return (
                f"Bank '{beneficiary_bank}' could not be found. "
                "Please check the bank name and try again."
            )

        # ── Step 3: Beneficiary enquiry ───────────────────────────────────────
        benef_resp = requests.get(
            f"{WALLET_BASE_URL}/transfer/recipient",
            params={
                "accountNo":     beneficiary_account_number,
                "bank":          bank_code,
                "transfer_type": "inter",
            },
            headers=headers,
            timeout=20,
        )
        benef_data = benef_resp.json()

        if benef_data.get("status") in ("104", 104):
            return (
                "Account not found. Please check the account number and bank, then try again."
            )
        if benef_data.get("status") in ("500", 500):
            return "A server error occurred while verifying the beneficiary. Please retry."

        benef_info   = benef_data.get("data", {})
        to_account   = benef_info.get("accountNumber", beneficiary_account_number)
        benef_name   = benef_info.get("accountName", "Unknown")

        # ── Step 4: SHA-512 signature ─────────────────────────────────────────
        sig_raw  = f"{from_account}{to_account}"
        signature = hashlib.sha512(sig_raw.encode()).hexdigest()

        # ── Step 5: Execute transfer ──────────────────────────────────────────
        reference = _unique_ref()
        transfer_payload = {
            "fromAccount":  from_account,
            "toAccount":    to_account,
            "amount":       amount,
            "narration":    narration or "Transfer",
            "reference":    reference,
            "bank":         bank_code,
            "signature":    signature,
            "transfer_type": "inter",
        }

        txn_resp = requests.post(
            f"{WALLET_BASE_URL}/transfer",
            json=transfer_payload,
            headers=headers,
            timeout=30,
        )
        txn_data = txn_resp.json()

        if txn_data.get("status") != "00":
            return (
                f"Transfer failed: {txn_data.get('message', 'Unknown error')}. "
                "Please try again or contact support."
            )

        txn_ref    = txn_data.get("data", {}).get("reference", reference)
        session_id = txn_data.get("data", {}).get("sessionId", "")

        # ── Step 6: TSQ verification ──────────────────────────────────────────
        tsq_resp = requests.get(
            f"{WALLET_BASE_URL}/transactions",
            params={"reference": txn_ref},
            headers=headers,
            timeout=20,
        )
        tsq_status = tsq_resp.json().get("data", {}).get("status", "pending")

        return (
            f"✅ Transfer Successful!\n\n"
            f"  To          : {benef_name}\n"
            f"  Bank        : {beneficiary_bank.upper()}\n"
            f"  Account     : {to_account}\n"
            f"  Amount      : ₦{amount}\n"
            f"  Narration   : {narration or 'Transfer'}\n"
            f"  Reference   : {txn_ref}\n"
            f"  Status      : {tsq_status}"
        )

    except Exception as exc:
        log_error(f"transfer_money_tool error: {exc}")
        return f"An error occurred during the transfer: {exc}"


@tool
def get_beneficiary_name_tool(
    beneficiary_account_number: str,
    beneficiary_bank: str,
    phone_number: str,
) -> str:
    """
    Look up and return a beneficiary's account name before the customer confirms
    a transfer.  Call this BEFORE transfer_money_tool so the agent can show the
    account name to the customer for confirmation.

    Args:
        beneficiary_account_number: Recipient's account number.
        beneficiary_bank:           Recipient's bank name.
        phone_number:               Sender's phone number (for auth token).
    """
    try:
        headers = _wallet_headers()

        # Resolve bank name → code
        banks_resp = requests.get(f"{WALLET_BASE_URL}/bank", headers=headers, timeout=20)
        banks = banks_resp.json().get("data", [])
        bank_code = None
        bname_lower = beneficiary_bank.strip().lower()
        for bank in banks:
            if bname_lower in bank.get("name", "").lower() or bname_lower == bank.get("code", "").lower():
                bank_code = bank.get("code")
                break

        if not bank_code:
            return f"Bank '{beneficiary_bank}' could not be found. Please check the name and try again."

        resp = requests.get(
            f"{WALLET_BASE_URL}/transfer/recipient",
            params={"accountNo": beneficiary_account_number, "bank": bank_code, "transfer_type": "inter"},
            headers=headers,
            timeout=20,
        )
        data = resp.json()

        if data.get("status") in ("104", 104):
            return "Account not found. Please verify the account number and bank."
        if data.get("status") in ("500", 500):
            return "Server error while verifying account. Please retry."

        info = data.get("data", {})
        return (
            f"Beneficiary Details:\n"
            f"  Account Name   : {info.get('accountName', 'N/A')}\n"
            f"  Account Number : {beneficiary_account_number}\n"
            f"  Bank           : {beneficiary_bank.upper()}\n\n"
            f"Is this correct? Please confirm to proceed with the transfer."
        )

    except Exception as exc:
        return f"Could not retrieve beneficiary details: {exc}"


# ──────────────────────────────────────────────────────────────────────────────
# 7. CHANGE PIN
# ──────────────────────────────────────────────────────────────────────────────

@tool
def change_pin_tool(phone_number: str, old_pin: str, new_pin: str, confirm_new_pin: str) -> str:
    """
    Change the customer's 4-digit banking PIN.

    Args:
        phone_number:    Customer's registered phone number.
        old_pin:         Current 4-digit PIN.
        new_pin:         New 4-digit PIN.
        confirm_new_pin: Confirmation of the new PIN (must match new_pin).
    """
    try:
        if len(new_pin) != 4 or not new_pin.isdigit():
            return "Your new PIN must be exactly 4 numeric digits."

        if new_pin != confirm_new_pin:
            return "Your new PIN and confirmation PIN do not match. Please try again."

        if not _verify_pin(phone_number, old_pin):
            attempts = _increment_pin_attempts(phone_number)
            remaining = max(0, 5 - attempts)
            return f"Incorrect current PIN. You have {remaining} attempt(s) remaining."

        new_hash = hashlib.sha256(new_pin.encode()).hexdigest()
        with SessionLocal() as session:
            session.execute(
                text("""
                    UPDATE banking_customer_profile
                    SET pin_hash = :ph, failed_pin_attempts = 0
                    WHERE phone_number = :phone
                """),
                {"ph": new_hash, "phone": phone_number},
            )
            session.commit()

        return "✅ Your PIN has been successfully changed. Please use your new PIN for future transactions."

    except Exception as exc:
        log_error(f"change_pin_tool error: {exc}")
        return f"An error occurred while changing your PIN: {exc}"


# ──────────────────────────────────────────────────────────────────────────────
# 8. FORGOT PIN
# ──────────────────────────────────────────────────────────────────────────────

@tool
def forgot_pin_tool(phone_number: str, nin: str, new_pin: str, confirm_new_pin: str) -> str:
    """
    Reset the customer's PIN after successful NIN + liveness verification.
    The agent must present the liveness check link to the customer and only
    call this tool once the liveness API has already returned success.

    Args:
        phone_number:    Customer's registered phone number.
        nin:             Customer's National Identification Number.
        new_pin:         New 4-digit PIN.
        confirm_new_pin: Confirmation of the new PIN.
    """
    try:
        if len(new_pin) != 4 or not new_pin.isdigit():
            return "Your new PIN must be exactly 4 numeric digits."

        if new_pin != confirm_new_pin:
            return "The PINs do not match. Please re-enter and confirm your new PIN."

        # Trigger liveness verification via the configured liveness API
        liveness_url    = os.getenv("LIVENESS_API_URL", "https://yourapp.com/api/liveness")
        liveness_payload = {"phoneNumber": phone_number, "nin": nin}
        liveness_resp   = requests.post(liveness_url, json=liveness_payload, timeout=30)
        liveness_data   = liveness_resp.json()

        if liveness_data.get("status") != "00":
            return (
                "Liveness verification failed. We could not confirm your identity. "
                "Please try again in a well-lit environment or contact support."
            )

        # Liveness passed – update PIN
        new_hash = hashlib.sha256(new_pin.encode()).hexdigest()
        with SessionLocal() as session:
            session.execute(
                text("""
                    UPDATE banking_customer_profile
                    SET pin_hash = :ph, failed_pin_attempts = 0
                    WHERE phone_number = :phone
                """),
                {"ph": new_hash, "phone": phone_number},
            )
            session.commit()

        return (
            "✅ Your PIN has been successfully reset. "
            "You can now log in and access all banking services with your new PIN."
        )

    except Exception as exc:
        log_error(f"forgot_pin_tool error: {exc}")
        return f"An error occurred during PIN reset: {exc}"


# ──────────────────────────────────────────────────────────────────────────────
# 9. SAVED BILLERS  (quick-pay)
# ──────────────────────────────────────────────────────────────────────────────

def _upsert_saved_biller(
    phone_number: str,
    biller_name: str,
    biller_info: dict,
    reference_number: str,
):
    """Persist a biller + reference after a successful payment."""
    try:
        with SessionLocal() as session:
            session.execute(
                text("""
                    INSERT INTO banking_saved_billers
                        (phone_number, biller_name, biller_id, division_id, product_id,
                         payment_code, category, reference_number, last_used)
                    VALUES
                        (:phone, :name, :bid, :did, :pid, :pc, :cat, :ref, NOW())
                    ON CONFLICT (phone_number, biller_id, reference_number)
                    DO UPDATE SET
                        last_used = NOW(),
                        biller_name = EXCLUDED.biller_name
                """),
                {
                    "phone": phone_number,
                    "name":  biller_name,
                    "bid":   biller_info["billerId"],
                    "did":   biller_info["divisionId"],
                    "pid":   biller_info["productId"],
                    "pc":    biller_info["paymentCode"],
                    "cat":   biller_info["category"],
                    "ref":   reference_number,
                },
            )
            session.commit()
    except Exception as exc:
        log_error(f"_upsert_saved_biller error: {exc}")


@tool
def get_saved_billers_tool(phone_number: str) -> str:
    """
    Return a list of the customer's previously paid billers for quick re-pay.

    Args:
        phone_number: Customer's registered phone number.
    """
    try:
        with SessionLocal() as session:
            rows = session.execute(
                text("""
                    SELECT biller_name, category, reference_number, last_used
                    FROM banking_saved_billers
                    WHERE phone_number = :phone
                    ORDER BY last_used DESC
                    LIMIT 10
                """),
                {"phone": phone_number},
            ).fetchall()

        if not rows:
            return "You have no saved billers yet. Complete a bill payment to save a biller for quick future access."

        lines = ["Here are your saved billers:\n"]
        for i, row in enumerate(rows, 1):
            cat       = row[1] or ""
            ref_label = CATEGORY_REFERENCE_LABEL.get(cat.lower(), "Reference")
            lines.append(
                f"  {i}. {row[0].upper()}  |  {ref_label}: {row[2]}  |  Last used: {row[3]}"
            )
        lines.append("\nReply with the number to pay again, or type a new biller name.")
        return "\n".join(lines)

    except Exception as exc:
        return f"Could not retrieve saved billers: {exc}"


@tool
def delete_saved_biller_tool(phone_number: str, biller_name: str, reference_number: str) -> str:
    """
    Remove a saved biller from the customer's quick-pay list.

    Args:
        phone_number:     Customer's registered phone number.
        biller_name:      Name of the biller to remove.
        reference_number: Reference (meter number / smart card) associated with it.
    """
    try:
        with SessionLocal() as session:
            session.execute(
                text("""
                    DELETE FROM banking_saved_billers
                    WHERE phone_number = :phone
                      AND LOWER(biller_name) = LOWER(:name)
                      AND reference_number = :ref
                """),
                {"phone": phone_number, "name": biller_name, "ref": reference_number},
            )
            session.commit()
        return f"✅ '{biller_name}' (ref: {reference_number}) has been removed from your saved billers."
    except Exception as exc:
        return f"Could not delete saved biller: {exc}"


# ──────────────────────────────────────────────────────────────────────────────
# 10. BANK LIST  (helper – agent uses this to present banks to the customer)
# ──────────────────────────────────────────────────────────────────────────────

@tool
def get_bank_list_tool(search: Optional[str] = None) -> str:
    """
    Return a list of Nigerian banks.  Pass an optional search term to filter.

    Args:
        search: Optional partial bank name to filter results (e.g. "access").
    """
    try:
        headers = _wallet_headers()
        resp    = requests.get(f"{WALLET_BASE_URL}/bank", headers=headers, timeout=20)
        banks   = resp.json().get("data", [])

        if search:
            banks = [b for b in banks if search.lower() in b.get("name", "").lower()]

        if not banks:
            return "No matching banks found. Please check the name and try again."

        names = [f"  • {b.get('name', 'N/A')}" for b in banks[:30]]
        return "Available banks:\n" + "\n".join(names)

    except Exception as exc:
        return f"Could not retrieve bank list: {exc}"


# ──────────────────────────────────────────────────────────────────────────────
# EXPORTED TOOL LIST  (import this in tools.py / chat_bot.py)
# ──────────────────────────────────────────────────────────────────────────────

banking_tools = [
    create_vfd_account_tool,
    fund_wallet_info_tool,
    balance_enquiry_tool,
    buy_airtime_tool,
    pay_bill_tool,
    transfer_money_tool,
    get_beneficiary_name_tool,
    change_pin_tool,
    forgot_pin_tool,
    get_saved_billers_tool,
    delete_saved_biller_tool,
    get_bank_list_tool,
]
