from typing import Optional, List
from fastapi import FastAPI, UploadFile, File, Form, Request
from fastapi.responses import JSONResponse, HTMLResponse, FileResponse
from fastapi.templating import Jinja2Templates
from langchain_community.document_loaders import Docx2txtLoader, PyPDFLoader
from langchain_community.llms import OpenAI
from langchain.chains.summarize import load_summarize_chain
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
import tensorflow as tf
import numpy as np
# import cv2
from gtts import gTTS
from langchain.schema import Document
from dotenv import load_dotenv
import os
import tempfile
from pathlib import Path




# 환경 변수 로드
load_dotenv()

# FastAPI 앱 생성
app = FastAPI()

# Static 파일 및 템플릿 경로 설정
BASE_DIR = Path(__file__).resolve().parent
# app.mount("/static", StaticFiles(directory=BASE_DIR / "static"), name="static")
templates = Jinja2Templates(directory=BASE_DIR / "templates")


@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    """메인 페이지: 버튼 두 개 제공"""
    return templates.TemplateResponse("home.html", {"request": request})


@app.get("/upload", response_class=HTMLResponse)
async def upload_page(request: Request):
    """이력서 업로드 페이지"""
    return templates.TemplateResponse("upload.html", {"request": request})


@app.post("/summarize")
async def summarize_resume(request: Request, file: UploadFile = File(...)):
    # 업로드된 파일을 임시 위치에 저장
    with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
        contents = await file.read()
        tmp_file.write(contents)
        tmp_file_name = tmp_file.name

    # 파일 형식에 따라 적절한 로더 선택
    if file.filename.endswith('.pdf'):
        loader = PyPDFLoader(tmp_file_name)
    elif file.filename.endswith('.docx'):
        loader = Docx2txtLoader(tmp_file_name)
    else:
        os.remove(tmp_file_name)
        return JSONResponse(content={"error": "지원하지 않는 파일 형식입니다."}, status_code=400)

    documents = loader.load()
    print(documents)
    # 임시 파일 삭제
    os.remove(tmp_file_name)

    # LLM 초기화 (API 키를 환경 변수에서 가져옴)
    openai_api_key = os.getenv("OPENAI_API_KEY")
    if not openai_api_key:
        return JSONResponse(content={"error": "API 키가 설정되지 않았습니다."}, status_code=500)

    llm = OpenAI(temperature=0.7, openai_api_key=openai_api_key)

    # 한 문장 요약을 위한 프롬프트 설정
    prompt_template = """다음 내용을 구직활동에 도움이 되는 핵심정보만 포함하여 한 문장으로 요약하세요:

{text}

요약:"""
    PROMPT = PromptTemplate(template=prompt_template, input_variables=["text"])

    # 요약 체인 로드
    chain = load_summarize_chain(llm, chain_type="stuff", prompt=PROMPT)

    # 요약 실행
    summary = chain.run(documents)

    # 결과를 프론트로 전달
    return templates.TemplateResponse("result.html", {"request": request, "summary": summary})


@app.get("/create", response_class=HTMLResponse)
async def create_page(request: Request):
    """이력서 작성 페이지"""
    return templates.TemplateResponse("create.html", {"request": request})


@app.post("/submit")
async def submit_resume(
    request: Request,
    name: Optional[str] = Form(None),
    age: Optional[int] = Form(None),
    contact: Optional[str] = Form(None),
    experience: Optional[str] = Form(None),
    education: Optional[str] = Form(None),
    certifications: Optional[str] = Form(None),
    skills: Optional[str] = Form(None),
    desired_job: Optional[str] = Form(None),
    desired_location: Optional[str] = Form(None)
):
    """작성된 이력서를 요약"""
    # 이력서 텍스트 생성
    resume_text = f"""
    이름: {name or "정보 없음"}
    나이: {age or "정보 없음"}
    연락처: {contact or "정보 없음"}
    주요 경력: {experience or "정보 없음"}
    학력: {education or "정보 없음"}
    자격증: {certifications or "정보 없음"}
    기술: {skills or "정보 없음"}
    희망 직무: {desired_job or "정보 없음"}
    희망 근무지: {desired_location or "정보 없음"}
    """

    # `Document` 객체로 변환
    document = Document(page_content=resume_text)

    # LLM 초기화
    openai_api_key = os.getenv("OPENAI_API_KEY")
    if not openai_api_key:
        return JSONResponse(content={"error": "API 키가 설정되지 않았습니다."}, status_code=500)

    llm = OpenAI(temperature=0.7, openai_api_key=openai_api_key)

    # 요약 프롬프트
    prompt_template = """다음 내용을 구직활동에 도움이 되는 핵심정보만 포함하여 한 문장으로 요약하세요:

{text}

요약:"""
    PROMPT = PromptTemplate(template=prompt_template, input_variables=["text"])
    chain = load_summarize_chain(llm, chain_type="stuff", prompt=PROMPT)

    # `chain.invoke`를 사용하여 요약 수행
    summary = chain.invoke([document])

    return templates.TemplateResponse("result.html", {"request": request, "summary": summary})


# @app.post("/ask_question")
# async def ask_question(job_description: str = Form(...)):
#     """OpenAI를 사용하여 맞춤형 면접 질문 생성"""
#     openai_api_key = os.getenv("OPENAI_API_KEY")
#     job_description = '60세 이상 우대, 서울시립도서관에서 도서 대출 보조, 서가 정리 및 이용자 안내를 담당할 시간제 근로자를 모집합니다.'

#     if not openai_api_key:
#         return JSONResponse(content={"error": "API 키가 설정되지 않았습니다."}, status_code=500)

@app.get("/ask_question")
async def ask_question():
    """
    LangChain을 사용하여 고정된 job_description을 기반으로 면접 질문 생성
    """
    openai_api_key = os.getenv("OPENAI_API_KEY")
    if not openai_api_key:
        return JSONResponse(content={"error": "API 키가 설정되지 않았습니다."}, status_code=500)

    try:
        # 고정된 Job Description
        job_description = "60세 이상 우대, 서울시립도서관에서 도서 대출 보조, 서가 정리 및 이용자 안내를 담당할 시간제 근로자를 모집합니다."

        # LLM 초기화
        llm = OpenAI(temperature=0.7, openai_api_key=openai_api_key)

        # 프롬프트 템플릿 정의
        prompt_template = PromptTemplate(
            input_variables=["job_description"],
            template="""
            다음 채용공고를 기반으로 구직자의 역량을 평가할 수 있는 면접 질문 3개를 한국어로 생성하세요:
            
            채용공고:
            {job_description}

            면접 질문:
            """
        )

        # LangChain LLMChain 구성
        chain = LLMChain(llm=llm, prompt=prompt_template)

        # LangChain을 사용하여 질문 생성
        generated_questions = chain.run({"job_description": job_description})
        question_text = generated_questions.strip()

        # TTS로 질문 음성 생성
        tts = gTTS(text=question_text, lang="ko")
        tts.save("question.mp3")

        return FileResponse("question.mp3")

    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)

# @app.post("/analyze_feedback")
# async def analyze_feedback(file: UploadFile = File(...)):
#     """Teachable Machine 모델을 사용하여 자세 분석 및 피드백 생성"""
#     try:
#         contents = await file.read()
#         nparr = np.frombuffer(contents, np.uint8)
#         img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
#         img = cv2.resize(img, (224, 224))
#         img = np.expand_dims(img, axis=0) / 255.0

#         predictions = model.predict(img)
#         label = class_names[np.argmax(predictions)]

#         feedback = {
#             "Confident": "Great! You look confident and professional.",
#             "Neutral": "Your posture is neutral. Try to show more enthusiasm.",
#             "Poor Posture": "Your posture needs improvement. Sit up straight and maintain eye contact."
#         }.get(label, "Unknown feedback.")

#         tts = gTTS(text=feedback, lang="en")
#         tts.save("feedback.mp3")

#         return FileResponse("feedback.mp3")

#     except Exception as e:
#         return JSONResponse(content={"error": str(e)}, status_code=500)