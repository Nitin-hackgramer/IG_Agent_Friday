import httpx
import logging
import os
from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger("uvicorn")

META_GRAPH_URL = "https://graph.instagram.com/v25.0/me/messages"
PAGE_ACCESS_TOKEN = os.getenv("PAGE_ACCESS_TOKEN")


async def send_instagram_dm(recipient_id: str, message_text: str) -> bool:
    """
    Sends an outbound text DM back to the user via Meta's Graph API.
    Replicates the 'Send Message' node functionality from n8n
    """
    headers = {"Content-Type": "application/json"}

    # This matches the Exact JSON payload n8n builds dynamically behind the scenes
    payload = {"recipient": {"id": recipient_id}, "message": {"text": message_text}}

    params = {"access_token": PAGE_ACCESS_TOKEN}

    async with httpx.AsyncClient() as client:
        try:
            response = await client.post(
                META_GRAPH_URL, json=payload, params=params, headers=headers
            )

            # If Meta responds with something other than 200, raise an Exception
            response.raise_for_status()
            logger.info(f"Successfully Sent message to user {recipient_id}")
            return True

        except httpx.HTTPStatusError as exc:
            # Try to parse Meta's JSON error body for a clearer message
            try:
                body = exc.response.json()
                error = body.get("error", {})
                code = error.get("code")
                subcode = error.get("error_subcode")
                msg = error.get("message")
            except Exception:
                code = exc.response.status_code
                subcode = None
                msg = exc.response.text

            # Common: recipient/user not found (subcode 2534014) — warn and skip
            if subcode == 2534014:
                logger.warning(
                    f"Meta API: recipient not found (subcode={subcode}) when sending to {recipient_id}: {msg}"
                )
            else:
                logger.error(f"Meta API Error: {code} - {msg}")
            return False

        except Exception as e:
            logger.error(f"Unexpected network error when calling Meta API: {str(e)}")
            return False
