import os
import sqlite3
from fastapi import FastAPI, UploadFile, File
from fastapi import HTTPException, Depends
from fastapi.security import OAuth2PasswordBearer
from passlib.context import CryptContext
from .auth_module.database import engine
from .auth_module.models import User, Base

from build_pipeline.build_embeddings import process_and_store_data
from config import OPENAI_API_KEY, DB_PATH, BASE_DIR
from query_pipeline.query_model import query_pipeline, ClinicalRequest, QueryRequest, query_clinical_decision_making, \
    query_public_health_monitoring, query_research
from .auth_module import authenticate_user, create_access_token
from .auth_module.auth import register_user, get_current_user
from pydantic import BaseModel

class UserRequest(BaseModel):
    username: str
    password: str

ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 60

# Password hashing utility (optional if storing passwords)
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")

app = FastAPI()

@app.on_event("startup")
def startup_event():
    print(f"🔍 Using DB path: {DB_PATH}")
    Base.metadata.create_all(bind=engine)

@app.post("/register")
def register(request: UserRequest):
    try:
        register_user(request.username, request.password)
    except sqlite3.IntegrityError as e:
        raise HTTPException(status_code=400, detail= "Error creating use" + str(e))
    return {"message": "User registered successfully"}

@app.post("/login")
def login(request: UserRequest):
    user = authenticate_user(request.username,request.password)
    if not user:
        raise HTTPException(status_code=401, detail="Invalid credentials")
    token = create_access_token(data={"sub": user.username})
    return {"access_token": token, "token_type": "bearer"}

@app.post("/query")
async def query(request: QueryRequest):
    response = query_pipeline(
        user_question=request.question,
        api_token= OPENAI_API_KEY,
        prompt_file_path= os.path.join(BASE_DIR, 'data/prompts/use_cases_prompt.txt')
    )
    return {"response": response}

@app.post("/query_clinical_decision_making")
async def query_clinical_decision_making_endpoint(request: ClinicalRequest, user=Depends(get_current_user)):
    response = query_clinical_decision_making(
        clinical_payload=request,
        api_token= OPENAI_API_KEY,
    )
    return {"response": response}

@app.post("/query_public_health_monitoring")
async def query_public_health_monitoring_endpoint(request: QueryRequest, user=Depends(get_current_user)):
    response = query_public_health_monitoring(
        user_question=request.question,
        api_token= OPENAI_API_KEY,
    )
    return {"response": response}

@app.post("/query_research")
async def query_research_endpoint(request: QueryRequest, user=Depends(get_current_user)):
    response = query_research(
        user_question=request.question,
        api_token= OPENAI_API_KEY,
    )
    return {"response": response}

@app.post("/build_embeddings")
async def build(file: UploadFile = File(...), user=Depends(get_current_user)):
    filepath = f"data/{file.filename}"
    with open(filepath, "wb") as f:
        f.write(file.file.read())
    result = process_and_store_data()
    return {"message": "Embeddings built", "result": result}
