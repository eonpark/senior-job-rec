from typing import Optional
from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import JSONResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from langchain_community.document_loaders import Docx2txtLoader, PyPDFLoader
from langchain_community.llms import OpenAI
from langchain.chains.summarize import load_summarize_chain
from langchain.prompts import PromptTemplate
import os
import tempfile
from pathlib import Path
from fastapi import Request
from langchain.schema import Document
from dotenv import load_dotenv  # 추가된 부분
from module import hwp_loader  # 한글 data loader 추가

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
    tmp_file_name = None  # 초기화

    try:
        # 업로드된 파일을 임시 위치에 저장
        with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
            contents = await file.read()
            tmp_file.write(contents)
            tmp_file_name = tmp_file.name

        # 파일 형식에 따라 적절한 로더 선택
        if file.filename.endswith(".pdf"):
            loader = PyPDFLoader(tmp_file_name)
        elif file.filename.endswith(".docx"):
            loader = Docx2txtLoader(tmp_file_name)
        elif file.filename.endswith(".hwp"):  # 한글 파일 추가
            loader = hwp_loader.HWPLoader(tmp_file_name)
        else:
            return JSONResponse(
                content={"error": "지원하지 않는 파일 형식입니다."}, status_code=400
            )

        # 문서 내용 로드
        documents = loader.load()
        if not documents:
            raise ValueError("파일에서 텍스트를 추출할 수 없습니다.")

        # LLM 초기화 (API 키를 환경 변수에서 가져옴)
        openai_api_key = os.getenv("OPENAI_API_KEY")
        if not openai_api_key:
            return JSONResponse(
                content={"error": "OpenAI API 키가 설정되지 않았습니다."},
                status_code=500,
            )

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

        # 결과를 HTML로 전달
        return templates.TemplateResponse(
            "result.html", {"request": request, "summary": summary}
        )

    except Exception as e:  # 임시파일 존재로 인한 에러발생
        return JSONResponse(content={"error": str(e)}, status_code=500)
    finally:
        # 임시 파일 삭제
        if tmp_file_name and os.path.exists(tmp_file_name):
            try:
                os.remove(tmp_file_name)
            except Exception as e:
                print(f"임시 파일 삭제 중 오류 발생: {e}")


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
    desired_location: Optional[str] = Form(None),
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
        return JSONResponse(
            content={"error": "API 키가 설정되지 않았습니다."}, status_code=500
        )

    llm = OpenAI(temperature=0.7, openai_api_key=openai_api_key)

    # 요약 프롬프트
    prompt_template = """다음 내용을 구직활동에 도움이 되는 핵심정보만 포함하여 한 문장으로 요약하세요:

{text}

요약:"""
    PROMPT = PromptTemplate(template=prompt_template, input_variables=["text"])
    chain = load_summarize_chain(llm, chain_type="stuff", prompt=PROMPT)

    # `chain.invoke`를 사용하여 요약 수행
    summary = chain.invoke([document])

    return templates.TemplateResponse(
        "result.html", {"request": request, "summary": summary}
    )
