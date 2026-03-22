import anthropic
import base64
import json
import re
import streamlit as st


def extract_rateview_data(image_bytes):
    """Send DAT RateView screenshot to Claude Vision API and extract structured data."""

    api_key = st.secrets.get("anthropic_api_key")
    if not api_key:
        return None, "No Anthropic API key configured. Add 'anthropic_api_key' to Streamlit secrets."

    client = anthropic.Anthropic(api_key=api_key)

    image_b64 = base64.standard_b64encode(image_bytes).decode("utf-8")

    # Determine media type
    if image_bytes[:4] == b'\x89PNG':
        media_type = "image/png"
    else:
        media_type = "image/jpeg"

    prompt = """Analyze this DAT RateView screenshot and extract ALL of the following data as JSON.

Extract from the rate panel:
- best_fit: the Best Fit rate (total dollar amount including fuel, e.g. 3091)
- range_low: the low end of the range (e.g. 2734)
- range_high: the high end of the range (e.g. 3482)
- rate_strength: the rate strength score (0-100)
- reports: number of reports
- companies: number of companies
- fuel_included: total fuel amount shown (e.g. 848)
- miles: miles from Quote Calculator if visible

Extract from Lane Trend table (if visible):
- lane_trend: array of objects with {date, low, mid, high} for each month
  Dates should be in format "Mon YY" (e.g. "Feb 26")
  Values should be numbers without $ signs

If any field is not visible in the screenshot, set it to null.

Return ONLY valid JSON, no other text. Example format:
{
  "best_fit": 3091,
  "range_low": 2734,
  "range_high": 3482,
  "rate_strength": 57,
  "reports": 7,
  "companies": 6,
  "fuel_included": 848,
  "miles": 1116,
  "lane_trend": [
    {"date": "Feb 26", "low": 2355, "mid": 2578, "high": 2879},
    {"date": "Jan 26", "low": 2433, "mid": 2623, "high": 3069}
  ]
}"""

    try:
        response = client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=2000,
            messages=[{
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": media_type,
                            "data": image_b64,
                        }
                    },
                    {"type": "text", "text": prompt}
                ]
            }]
        )

        response_text = response.content[0].text

        # Try to parse JSON from response
        # Handle case where response might have markdown code blocks
        json_match = re.search(r'```(?:json)?\s*(.*?)\s*```', response_text, re.DOTALL)
        if json_match:
            json_str = json_match.group(1)
        else:
            json_str = response_text.strip()

        data = json.loads(json_str)
        return data, None

    except json.JSONDecodeError as e:
        return None, f"Failed to parse response: {e}\nRaw: {response_text[:500]}"
    except anthropic.APIError as e:
        return None, f"API error: {str(e)}"
    except Exception as e:
        return None, f"Error: {str(e)}"
