import json
import boto3
import time
import logging
import re
from prompt import build_prompt

# ------------------- Logger Setup -------------------
logger = logging.getLogger()
logger.setLevel(logging.INFO)

# ------------------- Bedrock Client -------------------
bedrock_runtime = boto3.client("bedrock-runtime", region_name="ap-south-1")

# ------------------- Constants -------------------
MODEL_ID = "arn:aws:bedrock:ap-south-1:036160411876:inference-profile/apac.anthropic.claude-sonnet-4-20250514-v1:0"
MAX_RETRIES = 7
RETRY_DELAY_SECONDS = 3
MAX_TOKENS = 25000

# ------------------- Markdown to HTML -------------------
def format_markdown_to_html(md: str) -> str:
    logger.info("Formatting markdown content to HTML")
    md = re.sub(r'^### (.*)', r'<h3>\1</h3>', md, flags=re.MULTILINE)
    md = re.sub(r'^## (.*)', r'<h2>\1</h2>', md, flags=re.MULTILINE)
    md = re.sub(r'^# (.*)', r'<h1>\1</h1>', md, flags=re.MULTILINE)
    md = re.sub(r'\*\*(.*?)\*\*', r'<strong>\1</strong>', md)
    md = re.sub(r'\*(.*?)\*', r'<em>\1</em>', md)
    md = re.sub(r'`(.*?)`', r'<code>\1</code>', md)
    md = re.sub(r'\n\s*[-*]\s+(.*)', r'\n<li>\1</li>', md)
    if "<li>" in md:
        md = f"<ul>{md}</ul>"
    md = re.sub(r'\n{2,}', r'<br><br>', md)
    md = re.sub(r'\n', r'<br>', md)
    return md.strip()

# ------------------- HTML Template Wrapper -------------------
def wrap_with_template(content_html: str, title="Model Output") -> str:
    logger.info("Wrapping formatted HTML with Bootstrap template.")
    return f"""
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8" />
        <meta name="viewport" content="width=device-width, initial-scale=1.0" />
        <title>{title}</title>
        <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/css/bootstrap.min.css" rel="stylesheet" />
        <style>
            body {{
                padding: 2rem;
                background-color: #f8f9fa;
            }}
            .content {{
                background-color: white;
                padding: 2rem;
                border-radius: 0.5rem;
                box-shadow: 0 0 10px rgba(0,0,0,0.1);
            }}
            code {{
                background-color: #eee;
                padding: 2px 5px;
                border-radius: 4px;
                font-family: monospace;
            }}
        </style>
    </head>
    <body>
        <div class="container content">
            {content_html}
        </div>
    </body>
    </html>
    """

# ------------------- Lambda Handler -------------------
def lambda_handler(event, context):
    logger.info("Lambda function invoked.")
    logger.info("Received event: %s", json.dumps(event))

    try:
        body = json.loads(event.get('body', '{}'))
        user_input = body.get("question", "").strip()
        topic = body.get("topic", "General").strip()

        logger.info("Extracted user input: '%s'", user_input)
        logger.info("Topic specified: '%s'", topic)

        if not user_input:
            logger.warning("Missing 'question' key or value in request body.")
            return {
                "statusCode": 400,
                "headers": {
                    "Access-Control-Allow-Origin": "*",
                    "Content-Type": "text/html"
                },
                "body": "Missing 'question' in request"
            }

        combined_prompt = build_prompt(user_input, topic)
        logger.info("Constructed prompt for Bedrock model.")

        payload = {
            "anthropic_version": "bedrock-2023-05-31",
            "max_tokens": MAX_TOKENS,
            "messages": [{"role": "user", "content": combined_prompt}]
        }

        for attempt in range(1, MAX_RETRIES + 1):
            try:
                logger.info("Attempt #%d: Sending request to Bedrock...", attempt)

                response = bedrock_runtime.invoke_model(
                    modelId=MODEL_ID,
                    contentType="application/json",
                    accept="application/json",
                    body=json.dumps(payload).encode("utf-8")
                )

                raw_response = response["body"].read().decode("utf-8")
                logger.info("Raw response received: %s", raw_response)

                result = json.loads(raw_response)
                model_output = result.get("content", [])[0].get("text", "").strip()
                logger.info("Model returned content with length %d", len(model_output))

                formatted_html = format_markdown_to_html(model_output)
                full_html = wrap_with_template(formatted_html)
                logger.info("Final HTML body: %s", full_html)

                return {
                    "statusCode": 200,
                    "headers": {
                        "Access-Control-Allow-Origin": "*",
                        "Content-Type": "text/html"
                    },
                    "body": full_html
                }

            except Exception as retry_exception:
                logger.warning("Error in attempt #%d: %s", attempt, str(retry_exception))
                if attempt == MAX_RETRIES:
                    raise
                logger.info("Retrying in %d seconds...", RETRY_DELAY_SECONDS)
                time.sleep(RETRY_DELAY_SECONDS)

    except Exception as e:
        logger.error("Unhandled exception occurred: %s", str(e), exc_info=True)
        error_html = wrap_with_template(f"<h2>Error</h2><p>{str(e)}</p>")
        return {
            "statusCode": 500,
            "headers": {
                "Access-Control-Allow-Origin": "*",
                "Content-Type": "text/html"
            },
            "body": error_html
        }
