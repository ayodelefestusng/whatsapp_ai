import os
import sys
import logging

# Set up logging to console
logging.basicConfig(level=logging.INFO)

try:
    from chat_bot import ingest_pdf_for_tenant
    import os

    pdf_path = os.path.abspath("test_policy.pdf")
    if not os.path.exists(pdf_path):
        from reportlab.pdfgen import canvas
        c = canvas.Canvas(pdf_path)
        c.drawString(100, 750, "Test Policy Content")
        c.save()
        print(f"Created {pdf_path}")

    print(f"Calling ingest_pdf_for_tenant for {pdf_path}")
    result = ingest_pdf_for_tenant("DMC", pdf_path)
    print(f"Result: {result}")

except Exception as e:
    import traceback
    print(f"Error occurred: {e}")
    traceback.print_exc()
