# query_pipeline.py
from models.retrieval_model import RetrievalModel
from utils.pinecone_utils import create_pinecone_index
from models.embedding_model import EmbeddingModel
from utils.prompt_utils import load_custom_prompt
from models.llm_model import LLMModel
from pydantic import BaseModel, Field
from typing import List, Optional
from datetime import date
import logging

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


class QueryRequest(BaseModel):
    question: str


class ScreeningHistory(BaseModel):
    last_screening_type: str
    last_screening_result: str
    date_of_last_screening: Optional[str]
    hpv_test_result: Optional[str]
    pap_smear_result: Optional[str]


class ClinicalFindings(BaseModel):
    presenting_symptoms: List[str]
    lesion_visible: bool
    lesion_description: Optional[str]
    cancer_stage: Optional[str]
    imaging_findings: Optional[str]


class PriorTreatment(BaseModel):
    cryotherapy: bool
    LEEP: bool
    radiation: bool
    chemotherapy: bool


class ClinicalRequest(BaseModel):
    patient_age: int
    parity: Optional[int]
    menopausal_status: Optional[str]
    hiv_status: Optional[str]
    art_adherence: Optional[str]
    screening_history: ScreeningHistory
    clinical_findings: ClinicalFindings
    comorbidities: Optional[List[str]] = []
    medications: Optional[List[str]] = []
    allergies: Optional[List[str]] = []
    prior_treatment: Optional[PriorTreatment]
    user_question: Optional[str]


def query_pipeline(user_question, api_token, prompt_file_path):
    """
    End-to-end pipeline for querying a vector database and generating a response using an LLM.
    Args:
        user_question (str): The question or query provided by the user.
        api_token (str): The API token used to authenticate with the LLM provider (e.g., OpenAI or HuggingFace).
        prompt_file_path (str): Path to the prompt template file with placeholders for context and question.
    Returns:
        str: The generated response from the language model.
    """

    custom_prompt = load_custom_prompt(prompt_file_path)

    embedding_model = EmbeddingModel()
    index_name = "oncology-index"
    index = create_pinecone_index(index_name)
    retriever = RetrievalModel(index, embedding_model)

    llm = LLMModel(model_name="gpt-4", token=api_token)

    docs = retriever.get_relevant_documents(user_question)

    context = "\n\n".join([doc["content"] for doc in docs])
    prompt = custom_prompt.replace("{input}", user_question) \
        .replace("{context}", context)

    # Print the prompt for debugging
    logger.info(f'Model prompt: {prompt}')

    response = llm.generate_response(prompt)

    logger.info('Response generation complete')

    return response


def format_clinical_data(clinical_data: ClinicalRequest) -> str:
    out = [
        f"Patient Age: {clinical_data.patient_age}",
        f"Parity: {clinical_data.parity}",
        f"Menopausal Status: {clinical_data.menopausal_status}",
        f"HIV Status: {clinical_data.hiv_status}",
        f"ART Adherence: {clinical_data.art_adherence}",
        "",
        "Screening History:"
    ]

    screening = clinical_data.screening_history
    out.extend([
        f"  - Last Screening Type: {screening.last_screening_type}",
        f"  - Last Screening Result: {screening.last_screening_result}",
        f"  - Date of Last Screening: {screening.date_of_last_screening}",
        f"  - HPV Test Result: {screening.hpv_test_result}",
        f"  - Pap Smear Result: {screening.pap_smear_result}"
    ])

    findings = clinical_data.clinical_findings
    out.extend([
        "",
        "Clinical Findings:",
        f"  - Presenting Symptoms: {', '.join(findings.presenting_symptoms)}",
        f"  - Lesion Visible: {findings.lesion_visible}",
        f"  - Lesion Description: {findings.lesion_description}",
        f"  - Cancer Stage: {findings.cancer_stage}",
        f"  - Imaging Findings: {findings.imaging_findings}"
    ])

    out.extend([
        "",
        f"Comorbidities: {', '.join(clinical_data.comorbidities) if clinical_data.comorbidities else 'None'}",
        f"Current Medications: {', '.join(clinical_data.medications) if clinical_data.medications else 'None'}",
        f"Allergies: {', '.join(clinical_data.allergies) if clinical_data.allergies else 'None'}",
    ])

    prior_tx = clinical_data.prior_treatment or {}
    out.extend([
        "",
        "Prior Treatment History:",
        f"  - Cryotherapy: {prior_tx.cryotherapy}",
        f"  - LEEP: {prior_tx.LEEP}",
        f"  - Radiation: {prior_tx.radiation}",
        f"  - Chemotherapy: {prior_tx.chemotherapy}"
    ])

    out.append("")
    out.append(f"Clinical Question: {clinical_data.user_question}")

    return "\n".join(out)


def query_clinical_decision_making(clinical_payload: ClinicalRequest, api_token):
    """
    Query function tailored for clinical decision-making.
    Args:
        clinical_payload: Structured clinical data in JSON containing patient information.
        api_token (str): The API token used to authenticate with the LLM provider.
    Returns:
        str: The generated response from the language model.
    """
    prompt_file_path = "data/prompts/clinical_prompt.txt"
    clinical_data = format_clinical_data(clinical_payload)

    return query_pipeline(clinical_data, api_token, prompt_file_path)


def query_public_health_monitoring(user_question, api_token):
    """
    Query function tailored for public health monitoring.
    Args:
        user_question (str): The question or query provided by the user.
        api_token (str): The API token used to authenticate with the LLM provider.
    Returns:
        str: The generated response from the language model.
    """
    prompt_file_path = "data/prompts/use_cases_prompt.txt"

    return query_pipeline(user_question, api_token, prompt_file_path)


def query_research(user_question, api_token):
    """
    Query function tailored for research purposes.
    Args:
        user_question (str): The question or query provided by the user.
        api_token (str): The API token used to authenticate with the LLM provider.
    Returns:
        str: The generated response from the language model.
    """
    prompt_file_path = "data/prompts/use_cases_prompt.txt"

    return query_pipeline(user_question, api_token, prompt_file_path)