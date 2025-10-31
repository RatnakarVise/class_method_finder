import os
import json
import logging
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from openai import OpenAI
from dotenv import load_dotenv

# ------------------- LOGGER SETUP -------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s"
)
logger = logging.getLogger("ABAP_LLM_API")

# ------------------- LOAD ENV -------------------
dotenv_path = os.path.join(os.path.dirname(__file__), ".env")
if load_dotenv(dotenv_path):
    logger.info(f"✅ .env loaded from: {dotenv_path}")
else:
    logger.warning(f"⚠️ Could not find .env at {dotenv_path}")

api_key = os.getenv("OPENAI_API_KEY")

if not api_key:
    logger.error("❌ OPENAI_API_KEY not found in environment variables.")
else:
    logger.info("✅ OPENAI_API_KEY found successfully.")

# ------------------- INIT APP + CLIENT -------------------
app = FastAPI(title="LLM-based ABAP Flat Class-Method Extractor API")

try:
    client = OpenAI(api_key=api_key)
    logger.info("✅ OpenAI client initialized successfully.")
except Exception as e:
    logger.exception(f"❌ Failed to initialize OpenAI client: {e}")
    client = None


# ------------------- INPUT MODEL -------------------
class ABAPInput(BaseModel):
    pgm_name: str
    inc_name: str
    code: str


# ------------------- LLM FUNCTION -------------------
def extract_flat_class_method_map_llm(pgm_name: str, inc_name: str, code: str):
    """Uses GPT-4.1 to extract class and method names from ABAP code."""
    if not client:
        raise HTTPException(status_code=500, detail="OpenAI client not initialized. Check API key.")

    system_prompt = """
You are an SAP ABAP code analysis expert.
Given ABAP source code, identify all CLASS definitions and their METHODS.
Return only structured JSON objects as a flat list.
Each record must contain:
- pgm_name
- inc_name
- class_name
- methods (method name only)

Rules:
- Include both METHODS and CLASS-METHODS.
- Ignore commented lines or commented-out code.
- Return clean ABAP names exactly as they appear in code.
- No explanations or text — only pure JSON.
Example output format:
[
  { "pgm_name": "ZR_FINANCE_AUTOMATION", "inc_name": "ZINCLUDE_PAYMENT_PROCESS", "class_name": "zcl_sales_order_manager", "methods": "create_sales_order" },
  { "pgm_name": "ZR_FINANCE_AUTOMATION", "inc_name": "ZINCLUDE_PAYMENT_PROCESS", "class_name": "zcl_sales_order_manager", "methods": "create_po" }
]
"""

    user_prompt = f"""
ABAP Program Name: {pgm_name}
Include Name: {inc_name}

ABAP Code:
{code}
"""

    try:
        logger.info("🚀 Sending request to GPT-4.1 ...")
        response = client.chat.completions.create(
            model="gpt-4.1",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0.1,
        )
        logger.info("✅ LLM response received.")

        raw_output = response.choices[0].message.content.strip()
        logger.debug(f"🔍 Raw LLM output:\n{raw_output}")

        # Extract JSON
        json_start = raw_output.find("[")
        json_end = raw_output.rfind("]") + 1
        json_data = raw_output[json_start:json_end]

        mappings = json.loads(json_data)
        logger.info(f"✅ Parsed {len(mappings)} mappings from LLM.")
        return mappings

    except Exception as e:
        logger.exception(f"❌ LLM processing failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ------------------- ENDPOINT -------------------
@app.post("/extract_abap_map")
def extract_abap_map(payload: ABAPInput):
    """Return flattened list of (pgm_name, inc_name, class_name, method_name)."""
    logger.info(f"📥 Received request for program {payload.pgm_name}")
    return extract_flat_class_method_map_llm(payload.pgm_name, payload.inc_name, payload.code)


@app.get("/")
def root():
    return {"message": "LLM-based ABAP Mapping API", "status": "ok"}
