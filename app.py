from fastapi import FastAPI,Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.templating import Jinja2Templates
from src.pipeline.predict_pipeline import CustomData,PredictPipeline

app=FastAPI()
jinja=Jinja2Templates(directory="templates")

origins = [
    "http://localhost:8080",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins, 
    allow_credentials=True,
    allow_methods=["*"], 
    allow_headers=["*"], 
)

@app.get("/")
def index(request: Request):
    return jinja.TemplateResponse("index.html", {"request": request})

@app.get("/predict")
def make_prediction(request:Request):
    return jinja.TemplateResponse("home.html",{"request":request})

@app.post("/result")
async def show_result(request:Request):
    inp=await request.form()
    custom_data=CustomData(
    gender=inp.get("gender"),
    race_ethnicity=inp.get("race_ethnicity"),
    parental_level_of_education=inp.get("parental_level_of_education"),
    lunch=inp.get("lunch"),
    test_preparation_course=inp.get("test_preparation_course"),
    reading_score=int(inp.get("reading_score")),
    writing_score=int(inp.get("writing_score"))
    )
    df=custom_data.get_data_as_data_frame()
    predict_pipeline=PredictPipeline()
    result=predict_pipeline.predict(df)[0]
    return {"result":result}