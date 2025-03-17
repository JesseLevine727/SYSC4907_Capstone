import os
import json
import re
from fpdf import FPDF
from NurseAutoCharter import *

# ----------------------------
# PDF Generation Classes & Functions
# ----------------------------
class EHRPDF(FPDF):
    def header(self):
        self.set_font("Times", "B", 16)
        self.cell(0, 10, "Medical Patient Chart - Electronic Health Record (EHR)", ln=True, align="C")
        self.ln(5)
        self.set_draw_color(50, 50, 50)
        self.line(10, 25, 200, 25)
        self.ln(10)

    def footer(self):
        self.set_y(-15)
        self.set_font("Times", "I", 10)
        self.cell(0, 10, f"Page {self.page_no()}", align="C")

    def add_section_header(self, title):
        self.set_font("Times", "B", 14)
        self.cell(0, 8, title, ln=True)
        self.ln(2)
        self.set_draw_color(150, 150, 150)
        self.line(10, self.get_y(), 200, self.get_y())
        self.ln(5)

    def add_field(self, field_name, field_value):
        self.set_font("Times", "B", 12)
        self.cell(0, 7, f"{field_name}:", ln=True)
        self.set_font("Times", "", 12)
        field_value = self.convert_to_latin1(field_value)
        if isinstance(field_value, str):
            self.multi_cell(0, 7, field_value)
        elif isinstance(field_value, list):
            for item in field_value:
                self.cell(5)
                self.multi_cell(0, 7, f"- {item}")
        elif isinstance(field_value, dict):
            for key, value in field_value.items():
                self.multi_cell(0, 7, f"{key.capitalize()}: {value}")
        self.ln(3)

    def add_table(self, title, data_dict):
        self.add_section_header(title)
        self.set_font("Times", "B", 12)
        col_widths = [60, 65, 65]
        headers = ["Vital Sign", "Minimum", "Maximum"]
        for i in range(len(headers)):
            self.cell(col_widths[i], 8, headers[i], border=1, align="C")
        self.ln()
        self.set_font("Times", "", 12)
        for key, values in data_dict.items():
            self.cell(col_widths[0], 8, key.replace("_", " ").capitalize(), border=1, align="C")
            self.cell(col_widths[1], 8, str(values.get("minimum", "N/A")), border=1, align="C")
            self.cell(col_widths[2], 8, str(values.get("maximum", "N/A")), border=1, align="C")
            self.ln()
        self.ln(5)

    def convert_to_latin1(self, text):
        if isinstance(text, str):
            return text.encode("latin-1", "ignore").decode("latin-1")
        elif isinstance(text, list):
            return [self.convert_to_latin1(item) for item in text]
        elif isinstance(text, dict):
            return {key: self.convert_to_latin1(value) for key, value in text.items()}
        return text

def extract_vitals_from_string(vitals_string):
    matches = re.findall(r"([\w\s]+) ranged from (\d+\.?\d*) to (\d+\.?\d*)", vitals_string)
    structured_vitals = {}
    for match in matches:
        vital_name = match[0].strip().replace(" ", "_").lower()
        structured_vitals[vital_name] = {
            "minimum": match[1] + " units",
            "maximum": match[2] + " units"
        }
    return structured_vitals

def generate_patient_ehr(json_string, output_path):
    # Clean up JSON string if needed
    if json_string.startswith("```json"):
        json_string = json_string.replace("```json", "").replace("```", "").strip()
    try:
        patient_data = json.loads(json_string)
    except json.JSONDecodeError as e:
        print(f"Error parsing JSON: {e}")
        return

    pdf = EHRPDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()

    # Patient Info Section
    pdf.add_section_header("Patient Information")
    pdf.add_field("Name", patient_data.get("Name", "N/A"))
    pdf.add_field("Age", patient_data.get("Age", "N/A"))
    pdf.add_field("Sex", patient_data.get("Sex", "N/A"))
    pdf.add_field("Date of Admission", patient_data.get("Date_of_admission", "N/A"))
    pdf.add_field("Primary Care Provider", patient_data.get("Primary_care_provider", "N/A"))

    # Medical Details
    pdf.add_section_header("Medical History")
    pdf.add_field("Chief Complaint", patient_data.get("Complaint", "N/A"))
    pdf.add_field("Time Course", patient_data.get("Time_course", "N/A"))
    pdf.add_field("Symptom Severity", patient_data.get("Symptom_severity", "N/A"))
    pdf.add_field("Associated Symptoms", patient_data.get("Associated_symptoms", []))
    pdf.add_field("Exacerbating Factors", patient_data.get("Exacerbating_factors", []))
    pdf.add_field("Relieving Factors", patient_data.get("Relieving_factors", []))

    # Vital Signs Table
    vitals_data = patient_data.get("Vital_signs", {})
    if isinstance(vitals_data, str):
        vitals_data = extract_vitals_from_string(vitals_data)
    if isinstance(vitals_data, dict) and vitals_data:
        pdf.add_table("Vital Signs", vitals_data)

    # Interventions
    pdf.add_section_header("Interventions")
    interventions = patient_data.get("Interventions", [])
    formatted_interventions = []
    if all(isinstance(i, str) for i in interventions):
        formatted_interventions = interventions
    elif all(isinstance(i, dict) for i in interventions):
        formatted_interventions = [f"{i['type'].capitalize()}: {i['detail']}" for i in interventions]
    pdf.add_field("Actions Taken", formatted_interventions)

    # Progress Notes
    pdf.add_section_header("Progress Notes")
    pdf.add_field("Observations", patient_data.get("Progress_notes", "N/A"))

    # Follow-up Plan
    pdf.add_section_header("Follow-up Plan")
    pdf.add_field("Next Steps", patient_data.get("Follow_up_plan", "N/A"))

    pdf.output(output_path)
    print(f"Professional EHR PDF saved to {output_path}")

# ----------------------------
# Auto Chart Generation Function
# ----------------------------
def AutoGeneratePatientChart(filepath: str, session_id: str, output_pdf_path: str):
    """
    Generates a summary JSON from patient data and automatically creates a patient chart PDF.
    """
    # Generate the summary using the LLM pipeline (your NurseShiftSummary)
    summary_json = NurseShiftSummary(filepath, session_id)
    
    # Optionally, save the summary JSON to your DB (if desired)
    save_message(session_id, "ai", summary_json)
    
    # Validate that summary_json is proper JSON (or log error if not)
    try:
        json.loads(summary_json)
    except Exception as e:
        print(f"Error: Generated summary is not valid JSON: {e}")
        return f"Error: Generated summary is not valid JSON: {e}"
    
    # Generate the PDF chart from the summary JSON
    generate_patient_ehr(summary_json, output_pdf_path)
    
    return f"Patient chart generated and saved to {output_pdf_path}"

# Example usage:
if __name__ == "__main__":
    # Assume session_id and filepath are provided (could be via a UI, CLI, or configuration)
    session_id = "example_session_001"
    filepath = "Kate_Data.json"  # Replace with the actual path to your patient data JSON
    output_pdf_path = "Patient_Chart.pdf"
    
    result = AutoGeneratePatientChart(filepath, session_id, output_pdf_path)
    print(result)
