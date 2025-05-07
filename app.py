from fastapi import FastAPI
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from starlette.requests import Request
import threading
import time


from langchain_llm import get_answer               
from whisper_listener import listen_and_transcribe_

app = FastAPI()

templates = Jinja2Templates(directory="templates")
app.mount("/static", StaticFiles(directory="static"), name="static")

# Shared data structure for frontend polling
latest = {
    "question": "",
    "answer": "",
    "status": "idle"
}

@app.get("/")
def index(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.get("/get_latest_answer")
def get_latest_answer():
    return JSONResponse(content=latest)

def background_listener():
    global latest
    while True:
        print("🎧 Listening for user speech...")
        spoken_text = listen_and_transcribe_()

        if spoken_text and len(spoken_text.strip()) >= 10 and not spoken_text.strip().startswith(".") and spoken_text != latest["question"]:
            print(f"📝 Transcribed: {spoken_text}")
            latest["question"] = spoken_text
            latest["status"] = "answering"
            latest["answer"] = get_answer(spoken_text)
            latest["status"] = "ready"
        else:
            print("⚠️ Short or empty input, ignored")

        time.sleep(0.1)

# Background transcription thread (real-time)
threading.Thread(target=background_listener, daemon=True).start()
