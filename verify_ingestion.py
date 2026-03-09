from reportlab.pdfgen import canvas
import os
import requests
import json
import time

def create_test_pdf(filename, content):
    c = canvas.Canvas(filename)
    # Adding more text to ensure splitting happens if needed
    textobject = c.beginText(100, 750)
    textobject.setFont("Helvetica", 12)
    for line in content:
        textobject.textLine(line)
    c.drawText(textobject)
    c.save()

if __name__ == "__main__":
    pdf_path = os.path.abspath("test_policy.pdf")
    content = [
        "ATB Employee Policy Handbook",
        "Section 1: Annual Leave",
        "Employees are entitled to 20 days of annual leave per year.",
        "Section 2: Sick Leave",
        "Employees are entitled to 10 days of paid sick leave per year.",
        "Section 3: Remote Work",
        "Remote work is allowed up to 2 days per week with manager approval."
    ]
    create_test_pdf(pdf_path, content)
    print(f"Created {pdf_path}")
    
    # Wait a bit for server to be ready if it was just started
    time.sleep(2)
    
    url = "http://localhost:8000/load_pdf"
    payload = {
        "tenant_id": "DMC",
        "file_path": pdf_path
    }
    
    print(f"Calling {url} with {payload}")
    try:
        response = requests.post(url, json=payload)
        print(f"Status: {response.status_code}")
        print(f"Response: {json.dumps(response.json(), indent=2)}")
    except Exception as e:
        print(f"Error calling endpoint: {e}")
