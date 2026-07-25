import re

def parse_soap_sections(soap_text: str) -> dict:
    sections = ["SUBJECTIVE", "OBJECTIVE", "ASSESSMENT", "PLAN"]
    result = {}
    pattern = r"(" + "|".join(sections) + r"):\s*(.*?)(?=(?:" + "|".join(sections) + r"):|\Z)"
    matches = re.findall(pattern, soap_text, re.DOTALL)
    for section, content in matches:
        result[section.strip()] = content.strip()
    return result