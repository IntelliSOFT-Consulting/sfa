# query_pipeline.py
import logging
from typing import List, Optional, Literal
from datetime import date
from pydantic import BaseModel, Field

from models.retrieval_model import RetrievalModel
from utils.pinecone_utils import create_pinecone_index
from models.embedding_model import EmbeddingModel
from utils.prompt_utils import load_custom_prompt
from models.llm_model import LLMModel

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


class QueryRequest(BaseModel):
    question: str
class Meta(BaseModel):
    submitted_at: Optional[str] = None  # ISO-8601
    source_app_version: Optional[str] = None


class ClientFacility(BaseModel):
    date: Optional[date] = None
    county: Optional[str] = None
    sub_county: Optional[str] = None
    facility_name: Optional[str] = None
    service_provider_name: Optional[str] = None


class Residence(BaseModel):
    county: Optional[str] = None
    sub_county: Optional[str] = None
    ward: Optional[str] = None


class ClientIdentification(BaseModel):
    patient_id: Optional[str] = None
    full_name: Optional[str] = None
    age_years: Optional[int] = None
    phone_number: Optional[str] = None
    residence: Optional[Residence] = None


YNUNK = Literal["yes", "no", "unknown"]

class FamilyHistory(BaseModel):
    breast_cancer: Optional[YNUNK] = "unknown"
    hypertension: Optional[YNUNK] = "unknown"
    diabetes: Optional[YNUNK] = "unknown"
    mental_health_disorders: Optional[YNUNK] = "unknown"
    notes: Optional[str] = None


class DxTx(BaseModel):
    diagnosed: Optional[YNUNK] = "unknown"
    on_treatment: Optional[YNUNK] = "unknown"


class PersonalHistory(BaseModel):
    hypertension: Optional[DxTx] = None
    diabetes: Optional[DxTx] = None


class NcdRiskFactors(BaseModel):
    smoking: Optional[YNUNK] = "unknown"
    alcohol: Optional[YNUNK] = "unknown"


Menopause = Literal["pre-menopausal", "post-menopausal", "unknown"]

class Contraception(BaseModel):
    uses_contraception: Optional[YNUNK] = "unknown"
    method: Optional[str] = None


class ReproductiveHealth(BaseModel):
    gravida: Optional[int] = None
    parity: Optional[int] = None
    age_at_first_sex: Optional[int] = None
    contraception: Optional[Contraception] = None
    number_of_sex_partners: Optional[int] = None
    menopausal_status: Optional[Menopause] = "unknown"


class BPReading(BaseModel):
    systolic: Optional[float] = None
    diastolic: Optional[float] = None


class Measurements(BaseModel):
    weight_kg: Optional[float] = None
    height_cm: Optional[float] = None
    bmi: Optional[float] = None
    waist_circumference_cm: Optional[float] = None
    bp: Optional[dict] = Field(
        default_factory=lambda: {
            "reading_1": BPReading().dict(),
            "reading_2": BPReading().dict()
        }
    )


HivStatus = Literal["positive", "negative", "unknown"]
Adherence = Literal["good", "fair", "poor", "unknown"]

class HIV(BaseModel):
    status: Optional[HivStatus] = "unknown"
    on_art: Optional[YNUNK] = "unknown"
    art_start_date: Optional[date] = None
    adherence: Optional[Adherence] = "unknown"


DoneNA = Literal["done", "not_done", "not_applicable"]
PosNegSusUnk = Literal["positive", "negative", "suspected_cancer", "unknown"]

class HPVTesting(BaseModel):
    done: Optional[DoneNA] = "not_applicable"
    sample_date: Optional[date] = None
    self_sampling: Optional[YNUNK] = "unknown"
    result: Optional[PosNegSusUnk] = "unknown"
    action: Optional[List[str]] = None  # e.g., follow_up, referred, hpv_follow_up_1yr, routine_screen_3to5yrs


class VIATesting(BaseModel):
    done: Optional[DoneNA] = "not_applicable"
    result: Optional[PosNegSusUnk] = "unknown"
    action: Optional[List[str]] = None


class PapSmear(BaseModel):
    done: Optional[DoneNA] = "not_applicable"
    result: Optional[Literal["positive", "negative", "unknown"]] = "unknown"
    action: Optional[List[str]] = None


class TxBlock(BaseModel):
    status: Optional[DoneNA] = "not_applicable"
    single_visit_approach: Optional[YNUNK] = "unknown"
    if_not_done: Optional[Literal["referred", "postponed", None]] = None
    postponed_reason: Optional[str] = None


VisitType = Literal["hpv", "via", "pap", "mixed", "unknown"]

class PreCancerTreatment(BaseModel):
    cryotherapy: Optional[TxBlock] = None
    thermal_ablation: Optional[TxBlock] = None
    leep: Optional[TxBlock] = None


class CervicalScreening(BaseModel):
    type_of_visit: Optional[VisitType] = "unknown"
    hpv_testing: Optional[HPVTesting] = None
    via_testing: Optional[VIATesting] = None
    pap_smear: Optional[PapSmear] = None
    pre_cancer_treatment: Optional[PreCancerTreatment] = None


class BreastModality(BaseModel):
    done: Optional[Literal["yes", "no", "not_applicable"]] = "not_applicable"
    birads: Optional[Literal["0","1","2","3","4","5","6","7","8", None]] = None


class BreastAction(BaseModel):
    referred: Optional[str] = None
    follow_up: Optional[str] = None


class BreastScreening(BaseModel):
    cbe: Optional[Literal["normal", "abnormal", "not_applicable", "not_done"]] = "not_applicable"
    ultrasound: Optional[BreastModality] = None
    mammography: Optional[BreastModality] = None
    action: Optional[BreastAction] = None


class ClinicalFindings(BaseModel):
    presenting_symptoms: Optional[List[str]] = None
    lesion_visible: Optional[bool] = None
    lesion_description: Optional[str] = None
    cancer_stage: Optional[str] = None


class MedsAllergies(BaseModel):
    comorbidities: Optional[List[str]] = None
    current_medications: Optional[List[str]] = None
    allergies: Optional[List[str]] = None


class PriorTreatment(BaseModel):
    cryotherapy: Optional[bool] = None
    leep: Optional[bool] = None
    radiation: Optional[bool] = None
    chemotherapy: Optional[bool] = None


class LlmRequest(BaseModel):
    use_case: Literal["clinical_decision_support"] = "clinical_decision_support"
    user_question: Optional[str] = None


class ClinicalRequest(BaseModel):
    meta: Optional[Meta] = None
    client_facility: Optional[ClientFacility] = None
    client_identification: Optional[ClientIdentification] = None
    family_history: Optional[FamilyHistory] = None
    personal_history: Optional[PersonalHistory] = None
    ncd_risk_factors: Optional[NcdRiskFactors] = None
    reproductive_health: Optional[ReproductiveHealth] = None
    hiv: Optional[HIV] = None
    measurements: Optional[Measurements] = None
    cervical_screening: Optional[CervicalScreening] = None
    breast_screening: Optional[BreastScreening] = None
    clinical_findings: Optional[ClinicalFindings] = None
    medications_allergies: Optional[MedsAllergies] = None
    prior_treatment: Optional[PriorTreatment] = None
    llm_request: Optional[LlmRequest] = None

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


def _line(key: str, val) -> Optional[str]:
    """Format a single line, skipping empty/None."""
    if val is None or (isinstance(val, str) and val.strip() == ""):
        return None
    if isinstance(val, list):
        if len(val) == 0:
            return None
        return f"- {key}: {', '.join(map(str, val))}"
    if isinstance(val, bool):
        return f"- {key}: {'yes' if val else 'no'}"
    return f"- {key}: {val}"


def _block(title: str, lines: List[Optional[str]]) -> str:
    cleaned = [ln for ln in lines if ln]
    if not cleaned:
        return ""
    body = "\n".join(cleaned)
    return f"\n**{title}**\n{body}\n"


def format_clinical_data(payload: ClinicalRequest) -> str:
    """Create a compact, structured clinical brief for the LLM."""
    sections: List[str] = []

    # Identification
    ci = payload.client_identification
    rh = payload.reproductive_health
    id_lines = []
    if ci:
        id_lines += [
            _line("Patient ID", ci.patient_id),
            _line("Name", ci.full_name),
            _line("Age (years)", ci.age_years),
            _line("Phone", ci.phone_number),
        ]
        if ci.residence:
            id_lines += [
                _line("Residence County", ci.residence.county),
                _line("Residence Sub-county", ci.residence.sub_county),
                _line("Residence Ward", ci.residence.ward),
            ]
    if rh:
        id_lines += [
            _line("Parity", rh.parity),
            _line("Gravida", rh.gravida),
            _line("Menopausal status", rh.menopausal_status),
            _line("Contraception", (rh.contraception.method if rh.contraception else None)),
            _line("Number of sex partners", rh.number_of_sex_partners),
            _line("Age at first sex", rh.age_at_first_sex),
        ]
    sections.append(_block("Patient & Reproductive Profile", id_lines))

    # Facility
    cf = payload.client_facility
    if cf:
        sections.append(_block("Facility",
            [
                _line("Date", cf.date),
                _line("County", cf.county),
                _line("Sub-county", cf.sub_county),
                _line("Facility", cf.facility_name),
                _line("Provider", cf.service_provider_name),
            ]
        ))

    # HIV
    hiv = payload.hiv
    if hiv:
        sections.append(_block("HIV",
            [
                _line("Status", hiv.status),
                _line("On ART", hiv.on_art),
                _line("ART start date", hiv.art_start_date),
                _line("Adherence", hiv.adherence),
            ]
        ))

    # Measurements
    meas = payload.measurements
    if meas:
        bp1 = None
        bp2 = None
        if isinstance(meas.bp, dict):
            r1 = meas.bp.get("reading_1", {})
            r2 = meas.bp.get("reading_2", {})
            if r1.get("systolic") or r1.get("diastolic"):
                bp1 = f"{r1.get('systolic', '')}/{r1.get('diastolic', '')}"
            if r2.get("systolic") or r2.get("diastolic"):
                bp2 = f"{r2.get('systolic', '')}/{r2.get('diastolic', '')}"

        sections.append(_block("Measurements",
            [
                _line("Weight (kg)", meas.weight_kg),
                _line("Height (cm)", meas.height_cm),
                _line("BMI", meas.bmi),
                _line("Waist (cm)", meas.waist_circumference_cm),
                _line("BP reading 1", bp1),
                _line("BP reading 2", bp2),
            ]
        ))

    # NCD risks
    ncd = payload.ncd_risk_factors
    if ncd:
        sections.append(_block("NCD Risk Factors",
            [
                _line("Smoking", ncd.smoking),
                _line("Alcohol", ncd.alcohol),
            ]
        ))

    # Family & Personal history
    fh = payload.family_history
    if fh:
        sections.append(_block("Family History",
            [
                _line("Breast cancer", fh.breast_cancer),
                _line("Hypertension", fh.hypertension),
                _line("Diabetes", fh.diabetes),
                _line("Mental health disorders", fh.mental_health_disorders),
                _line("Notes", fh.notes),
            ]
        ))

    ph = payload.personal_history
    if ph:
        sections.append(_block("Personal History",
            [
                _line("Hypertension diagnosed", ph.hypertension.diagnosed if ph and ph.hypertension else None),
                _line("Hypertension on treatment", ph.hypertension.on_treatment if ph and ph.hypertension else None),
                _line("Diabetes diagnosed", ph.diabetes.diagnosed if ph and ph.diabetes else None),
                _line("Diabetes on treatment", ph.diabetes.on_treatment if ph and ph.diabetes else None),
            ]
        ))

    # Cervical screening
    cs = payload.cervical_screening
    if cs:
        lines = [_line("Type of visit", cs.type_of_visit)]
        if cs.hpv_testing:
            lines += [
                _line("HPV done", cs.hpv_testing.done),
                _line("HPV sample date", cs.hpv_testing.sample_date),
                _line("HPV self-sampling", cs.hpv_testing.self_sampling),
                _line("HPV result", cs.hpv_testing.result),
                _line("HPV action", cs.hpv_testing.action),
            ]
        if cs.via_testing:
            lines += [
                _line("VIA done", cs.via_testing.done),
                _line("VIA result", cs.via_testing.result),
                _line("VIA action", cs.via_testing.action),
            ]
        if cs.pap_smear:
            lines += [
                _line("Pap done", cs.pap_smear.done),
                _line("Pap result", cs.pap_smear.result),
                _line("Pap action", cs.pap_smear.action),
            ]
        if cs.pre_cancer_treatment:
            def tx_lines(name: str, tx: Optional[TxBlock]):
                if not tx: return []
                return [
                    _line(f"{name} status", tx.status),
                    _line(f"{name} single-visit", tx.single_visit_approach),
                    _line(f"{name} if_not_done", tx.if_not_done),
                    _line(f"{name} postponed_reason", tx.postponed_reason),
                ]
            lines += tx_lines("Cryotherapy", cs.pre_cancer_treatment.cryotherapy)
            lines += tx_lines("Thermal ablation", cs.pre_cancer_treatment.thermal_ablation)
            lines += tx_lines("LEEP", cs.pre_cancer_treatment.leep)
        sections.append(_block("Cervical Screening", lines))

    # Breast screening
    bs = payload.breast_screening
    if bs:
        lines = [_line("CBE", bs.cbe)]
        if bs.ultrasound:
            lines += [
                _line("Ultrasound done", bs.ultrasound.done),
                _line("Ultrasound BIRADS", bs.ultrasound.birads),
            ]
        if bs.mammography:
            lines += [
                _line("Mammography done", bs.mammography.done),
                _line("Mammography BIRADS", bs.mammography.birads),
            ]
        if bs.action:
            lines += [
                _line("Breast referral", bs.action.referred),
                _line("Breast follow-up", bs.action.follow_up),
            ]
        sections.append(_block("Breast Screening", lines))

    # Clinical findings
    cfnd = payload.clinical_findings
    if cfnd:
        sections.append(_block("Clinical Findings",
            [
                _line("Presenting symptoms", cfnd.presenting_symptoms or []),
                _line("Lesion visible", cfnd.lesion_visible),
                _line("Lesion description", cfnd.lesion_description),
                _line("Cancer stage", cfnd.cancer_stage)
            ]
        ))

    # Meds/Allergies & Prior Tx
    ma = payload.medications_allergies
    if ma:
        sections.append(_block("Comorbidities & Medications",
            [
                _line("Comorbidities", ma.comorbidities or []),
                _line("Current medications", ma.current_medications or []),
                _line("Allergies", ma.allergies or []),
            ]
        ))

    ptx = payload.prior_treatment
    if ptx:
        sections.append(_block("Prior Treatment",
            [
                _line("Cryotherapy", ptx.cryotherapy),
                _line("LEEP", ptx.leep),
                _line("Radiation", ptx.radiation),
                _line("Chemotherapy", ptx.chemotherapy),
            ]
        ))

    # Question
    uq = payload.llm_request.user_question if payload.llm_request else None
    sections.append(_block("Clinical Question", [_line("Question", uq)]))

    # Meta (last)
    mt = payload.meta
    if mt:
        sections.append(_block("Meta",
            [
                _line("Submitted at", mt.submitted_at),
                _line("App version", mt.source_app_version),
            ]
        ))

    # Join non-empty sections
    brief = "\n".join(s for s in sections if s and s.strip())
    return brief.strip()


def query_clinical_decision_making(clinical_payload: ClinicalRequest, api_token):

    print("Querying clinical decision-making model with payload:")

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