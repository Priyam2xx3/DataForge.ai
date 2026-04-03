import os
import uuid
import pandas as pd
from fastapi import FastAPI, UploadFile, File, Depends, HTTPException, Form
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy.orm import Session
from fastapi.responses import FileResponse
from typing import List, Optional
from pydantic import BaseModel

import database
import models
import data_engine

models.Base.metadata.create_all(bind=database.engine)

app = FastAPI(title="DataForge AI API")

# Setup CORS to allow frontend communication
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

os.makedirs("./tmp_data", exist_ok=True)

@app.get("/")
async def serve_ui():
    return FileResponse(
        "../frontend/index.html",
        headers={
            "Cache-Control": "no-cache, no-store, must-revalidate",
            "Pragma": "no-cache",
            "Expires": "0"
        }
    )

class TransformRequest(BaseModel):
    file_id: str
    command: str

@app.post("/upload")
async def upload_file(file: UploadFile = File(...), db: Session = Depends(database.get_db)):
    file_id = str(uuid.uuid4())
    ext = file.filename.split('.')[-1].lower()
    
    filepath = f"./tmp_data/{file_id}.{ext}"
    
    with open(filepath, "wb") as buffer:
        buffer.write(await file.read())
        
    try:
        if ext == 'csv':
            # Handle empty values during read for speed
            df = pd.read_csv(filepath, keep_default_na=True, na_values=['', 'nan', 'NAN', 'null'])
        elif ext in ['xls', 'xlsx']:
            df = pd.read_excel(filepath)
            df.replace({"": pd.NA}, inplace=True)
        elif ext == 'json':
            df = pd.read_json(filepath)
        else:
            raise HTTPException(status_code=400, detail="Unsupported file format")
            
        # Save a clean parquet version for fast state loading
        parquet_path = f"./tmp_data/{file_id}.parquet"
        df.to_parquet(parquet_path)
        
        # Save to DB
        db_file = models.DatasetMetadata(
            filename=file.filename,
            filepath=parquet_path,
            total_rows=len(df),
            total_cols=len(df.columns)
        )
        db.add(db_file)
        db.commit()
        db.refresh(db_file)
        
        preview = data_engine.get_preview_stats(df, include_expensive=False)
        return {"file_id": file_id, "filename": file.filename, "preview": preview}
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing file: {str(e)}")


@app.get("/preview/{file_id}")
async def get_preview(file_id: str):
    filepath = f"./tmp_data/{file_id}.parquet"
    if not os.path.exists(filepath):
        raise HTTPException(status_code=404, detail="File not found")
        
    df = pd.read_parquet(filepath)
    preview = data_engine.get_preview_stats(df, include_expensive=False)
    return {"preview": preview}


@app.get("/analyze/{file_id}")
async def analyze_data(file_id: str):
    filepath = f"./tmp_data/{file_id}.parquet"
    if not os.path.exists(filepath):
        raise HTTPException(status_code=404, detail="File not found")
        
    df = pd.read_parquet(filepath)
    recommendations = data_engine.analyze_dataset(df)
    preview = data_engine.get_preview_stats(df, include_expensive=True)
    return {"recommendations": recommendations, "preview": preview}


@app.post("/transform")
async def transform_data(request: TransformRequest):
    filepath = f"./tmp_data/{request.file_id}.parquet"
    if not os.path.exists(filepath):
        raise HTTPException(status_code=404, detail="File not found")
        
    df = pd.read_parquet(filepath)
    
    # Apply transformation via AI mapped logic
    df, message = data_engine.apply_nlp_transformation(df, request.command)
    
    # Save back the state
    df.to_parquet(filepath)
    
    # Return updated preview
    preview = data_engine.get_preview_stats(df)
    return {"message": message, "preview": preview}


@app.post("/chart")
async def create_chart(request: TransformRequest):
    """Generate a dark-themed chart image from a natural language prompt."""
    filepath = f"./tmp_data/{request.file_id}.parquet"
    if not os.path.exists(filepath):
        raise HTTPException(status_code=404, detail="File not found")

    df = pd.read_parquet(filepath)
    result = data_engine.generate_chart(df, request.command)

    if "error" in result:
        raise HTTPException(status_code=400, detail=result["error"])

    return result  # {"image_b64": "...", "title": "...", "description": "..."}


@app.get("/export/{file_id}")
async def export_data(file_id: str):
    filepath = f"./tmp_data/{file_id}.parquet"
    if not os.path.exists(filepath):
        raise HTTPException(status_code=404, detail="File not found")
        
    df = pd.read_parquet(filepath)
    export_path = f"./tmp_data/{file_id}_export.csv"
    df.to_csv(export_path, index=False)
    
    return FileResponse(path=export_path, filename=f"cleaned_data.csv", media_type='text/csv')

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)
