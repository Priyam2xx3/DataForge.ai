import pandas as pd
import numpy as np
import io
import os
import json
import google.generativeai as genai
from dotenv import load_dotenv

load_dotenv(override=True)

# Configure Gemini
api_key = os.getenv("GEMINI_API_KEY")
if api_key:
    genai.configure(api_key=api_key)

model = genai.GenerativeModel('gemini-2.5-flash')

def get_preview_stats(df: pd.DataFrame):
    """Returns a dictionary containing a sample of the data and some statistics."""
    # Data stats
    total_rows = len(df)
    total_cols = len(df.columns)
    null_counts = df.isnull().sum().to_dict()
    duplicates = int(df.duplicated().sum())

    # Replace NaNs with None for JSON serialization
    preview_df = df.head(50).replace({np.nan: None})
    
    return {
        "rows": total_rows,
        "cols": total_cols,
        "nulls": sum(null_counts.values()),
        "duplicates": duplicates,
        "columns": list(df.columns),
        "data": preview_df.to_dict(orient="records"),
        "col_types": {col: str(df[col].dtype) for col in df.columns}
    }

def analyze_dataset(df: pd.DataFrame):
    """Gives AI-powered recommendations for fixing the dataset."""
    if not api_key:
        return [{"title": "API Key Missing", "description": "Configure GEMINI_API_KEY to see AI recommendations.", "severity": "medium", "icon": "⚠️", "action": "None"}]

    sample = df.head(5).to_dict(orient="records")
    stats = {
        "rows": len(df),
        "cols": list(df.columns),
        "null_counts": df.isnull().sum().to_dict(),
        "duplicates": int(df.duplicated().sum())
    }

    prompt = f"""
    You are a Data Quality Expert. Analyze this dataset and recommend fixes.
    Stats: {json.dumps(stats)}
    Sample Data: {json.dumps(sample, default=str)}
    
    Return a JSON array of up to 5 recommendations. Each recommendation must have:
    - title (string)
    - description (string)
    - severity (high|medium|low)
    - action (string: plain english command suitable for NLP engine)
    - icon (string: emoji)
    
    Return ONLY a raw JSON array. DO NOT format as markdown. 
    """

    try:
        response = model.generate_content(prompt)
        text = response.text.strip().removeprefix('```json').removesuffix('```').strip()
        data = json.loads(text)
        return data
    except Exception as e:
        print(f"Error in analyze: {e}")
        return [{"title": "API Error", "description": str(e), "severity": "high", "icon": "❌", "action": "None"}]

def apply_nlp_transformation(df: pd.DataFrame, command: str) -> tuple[pd.DataFrame, str]:
    """
    Parses a user command and transforms the DataFrame accordingly.
    Instead of executing arbitrary code, the AI defines parameters for safe operations.
    """
    # Extremely basic safe operations we can support via structured response
    prompt = f"""
    You are a Data Transformation Engine.
    The user wants to apply a transformation to a pandas DataFrame.
    Columns: {list(df.columns)}
    Command: {command}
    
    You must map the user command to one of the following structured JSON actions:
    1. {{"op": "drop_duplicates"}}
    2. {{"op": "drop_empty_rows"}}
    3. {{"op": "fill_nulls", "column": "col_name", "strategy": "mean|median|mode|zero|forward|backward"}}
    4. {{"op": "fill_nulls_all", "strategy": "zero|empty_string"}}
    5. {{"op": "rename_columns", "mapping": {{"old_name": "new_name"}}}}
    6. {{"op": "drop_column", "column": "col_name"}}
    7. {{"op": "trim_whitespace"}}
    8. {{"op": "to_numeric", "column": "col_name"}}
    9. {{"op": "to_datetime", "column": "col_name"}}
    10. {{"op": "lowercase", "column": "col_name"}}
    11. {{"op": "uppercase", "column": "col_name"}}
    
    Return ONLY a single valid JSON object representing the operation. No markdown.
    """
    
    try:
        response = model.generate_content(prompt)
        text = response.text.strip().removeprefix('```json').removesuffix('```').strip()
        action = json.loads(text)
        
        op = action.get("op")
        msg = "Transformation applied."
        
        if op == "drop_duplicates":
            before = len(df)
            df.drop_duplicates(inplace=True)
            msg = f"Dropped {before - len(df)} duplicate rows."
            
        elif op == "drop_empty_rows":
            before = len(df)
            df.dropna(how="all", inplace=True)
            msg = f"Dropped {before - len(df)} empty rows."
            
        elif op == "fill_nulls":
            col = action.get("column")
            strategy = action.get("strategy")
            if col in df.columns:
                nulls = df[col].isnull().sum()
                if strategy == "mean":
                    val = df[col].mean()
                elif strategy == "median":
                    val = df[col].median()
                elif strategy == "mode":
                    val = df[col].mode()[0]
                elif strategy == "zero":
                    val = 0
                else:
                    val = 0
                df[col] = df[col].fillna(val)
                msg = f"Filled {nulls} nulls in {col} with {strategy} ({val})."
        
        elif op == "fill_nulls_all":
            # For simplicity
            df.fillna("", inplace=True)
            msg = "Filled all nulls."
            
        elif op == "rename_columns":
            mapping = action.get("mapping", {})
            df.rename(columns=mapping, inplace=True)
            msg = f"Renamed columns: {mapping}."
            
        elif op == "drop_column":
            col = action.get("column")
            if col in df.columns:
                df.drop(columns=[col], inplace=True)
                msg = f"Dropped column {col}."
                
        elif op == "trim_whitespace":
            str_cols = df.select_dtypes(include=['object']).columns
            for col in str_cols:
                df[col] = df[col].astype(str).str.strip()
            msg = "Trimmed whitespace from string columns."
            
        elif op == "to_numeric":
            col = action.get("column")
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
                msg = f"Converted {col} to numeric."
                
        elif op == "lowercase":
            col = action.get("column")
            if col in df.columns:
                df[col] = df[col].astype(str).str.lower()
                msg = f"Lowercased column {col}."
                
        elif op == "uppercase":
            col = action.get("column")
            if col in df.columns:
                df[col] = df[col].astype(str).str.upper()
                msg = f"Uppercased column {col}."
        else:
            msg = f"Unrecognized operation: {op}"

        return df, msg
        
    except Exception as e:
        print(f"Error executing NLP transform: {e}")
        return df, f"Error: Unable to process command. ({e})"
