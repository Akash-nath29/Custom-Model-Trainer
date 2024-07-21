from fastapi import FastAPI, File, UploadFile, Body
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
from utils.custom_training import CustomTraining

app = FastAPI()

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/train_random_forest_regressor")
async def train_random_forest_regressor(file: UploadFile = File(...), input_column_name: str = Body(...), output_column_name: str = Body(...)):
    input_column_name = [input_column_name]
    file_content = await file.read()
    with open('new_data.csv', 'wb') as f:
        f.write(file_content)
    
    custom_training = CustomTraining('new_data.csv', input_column_name, output_column_name)
    score = custom_training.train_random_forest_regressor()
        
    return {"message": "Model trained successfully!", "score": score}

@app.post('/download_model')
async def download_model():
    return FileResponse('model.pkl', media_type='application/octet-stream', filename='model.pkl')