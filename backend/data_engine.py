import base64
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend — safe for server use
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import io
import os
import json
import google.generativeai as genai
from dotenv import load_dotenv

load_dotenv(override=True)

model = genai.GenerativeModel('gemini-2.5-flash')

# ── Chart Theme Palette (matches UI dark theme) ──────────────
DARK_BG  = "#0f0f12"
CARD_BG  = "#1a1a22"
GRID     = "#252530"
TEXT_PRI = "#ede8e0"
TEXT_SEC = "#9a9080"
PALETTE  = ["#f97316","#fbbf24","#ef4444","#22c55e","#fb923c","#fde68a","#86efac","#fca5a5"]

def _apply_dark_style(fig, ax):
    """Apply dark analytics theme to a matplotlib figure."""
    fig.patch.set_facecolor(DARK_BG)
    ax.set_facecolor(CARD_BG)
    ax.tick_params(colors=TEXT_SEC, labelsize=9)
    ax.xaxis.label.set_color(TEXT_SEC)
    ax.yaxis.label.set_color(TEXT_SEC)
    ax.title.set_color(TEXT_PRI)
    for spine in ax.spines.values():
        spine.set_edgecolor(GRID)
    ax.grid(True, color=GRID, linewidth=0.6, linestyle='--', alpha=0.7)
    ax.set_axisbelow(True)

def _fig_to_b64(fig) -> str:
    """Convert a matplotlib figure to a Base64-encoded PNG string."""
    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=100, bbox_inches='tight',
                facecolor=DARK_BG, edgecolor='none')
    plt.close(fig)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode('utf-8')

def _local_parse_chart(df: pd.DataFrame, command: str) -> dict | None:
    """
    Fast local keyword parser — no API call, returns spec dict or None if unsure.
    Handles ~90% of common chart requests instantly.
    """
    cmd = command.lower()
    cols = list(df.columns)
    num_cols = list(df.select_dtypes(include='number').columns)
    cat_cols = list(df.select_dtypes(exclude='number').columns)

    # Detect chart type
    if any(k in cmd for k in ['barh', 'horizontal bar']):
        ct = 'barh'
    elif any(k in cmd for k in ['bar chart', 'bar graph', 'bar plot', '  bar ']):
        ct = 'bar'
    elif any(k in cmd for k in ['area chart', 'area graph', 'area plot']):
        ct = 'area'
    elif any(k in cmd for k in ['line chart', 'line graph', 'line plot', 'trend', 'over time']):
        ct = 'line'
    elif any(k in cmd for k in ['scatter plot', 'scatter chart', 'scatter']):
        ct = 'scatter'
    elif any(k in cmd for k in ['pie chart', 'pie graph', 'pie']):
        ct = 'pie'
    elif any(k in cmd for k in ['histogram', 'distribution', 'freq']):
        ct = 'histogram'
    elif any(k in cmd for k in ['box plot', 'box chart', 'boxplot', 'box']):
        ct = 'box'
    elif 'bar' in cmd:
        ct = 'bar'
    else:
        return None  # Can't determine — fall back to Gemini

    # Extract top_n
    import re
    top_n = None
    m = re.search(r'top\s+(\d+)', cmd)
    if m: top_n = int(m.group(1))

    # Find mentioned columns (case-insensitive match)
    mentioned = [c for c in cols if c.lower() in cmd]

    # Assign x/y based on chart type and mentioned columns
    x_col, y_col = None, None

    if ct == 'scatter':
        if len(mentioned) >= 2:
            x_col, y_col = mentioned[0], mentioned[1]
        elif len(num_cols) >= 2:
            x_col, y_col = num_cols[0], num_cols[1]
        if not x_col or not y_col: return None

    elif ct == 'pie':
        # Default to first categorical and first numeric
        for c in mentioned:
            if c in cat_cols and not x_col: x_col = c
            elif c in num_cols and not y_col: y_col = c
        if not x_col: x_col = cat_cols[0] if cat_cols else (cols[0] if cols else None)
        if not y_col: y_col = num_cols[0] if num_cols else None
        if not x_col or not y_col: return None

    elif ct in ('histogram', 'box'):
        # Only y matters for these
        if mentioned:
            y_col = [c for c in mentioned if c in num_cols] or [num_cols[0]] if num_cols else None
        else:
            y_col = num_cols[:4] if ct == 'box' else (num_cols[0] if num_cols else None)
        if not y_col: return None

    else: # Bar, Line, Area
        if len(mentioned) >= 2:
            # Assume first is X, rest are Y
            x_col = mentioned[0]
            y_col = mentioned[1] if len(mentioned) == 2 else [c for c in mentioned[1:] if c in num_cols]
        elif len(mentioned) == 1:
            if mentioned[0] in num_cols:
                x_col = cat_cols[0] if cat_cols else (cols[0] if cols != [mentioned[0]] else None)
                y_col = mentioned[0]
            else:
                x_col = mentioned[0]
                y_col = num_cols[0] if num_cols else None
        else:
            x_col = cat_cols[0] if cat_cols else (cols[0] if cols else None)
            y_col = num_cols[0] if num_cols else None
        
        if not y_col: return None

    title = f"{ct.capitalize()} Chart: {command}"
    return {
        "chart_type": ct, "x": x_col, "y": y_col,
        "title": title, "xlabel": str(x_col or ""), "ylabel": str(y_col if isinstance(y_col, str) else "Values"),
        "top_n": top_n or 20, # Default to top 20 for speed
        "description": f"Instant chart generated locally for '{command}'"
    }


def generate_chart(df: pd.DataFrame, command: str) -> dict:
    """
    Parse the user's NL chart request. Tries fast local parsing first,
    only calls Gemini API as fallback. Returns Base64 PNG or error.
    """
    # ── 1. Try instant local parse (no API call) ──────────────
    spec = _local_parse_chart(df, command)

    # ── 2. Fallback to Gemini only if local parse failed ──────
    if spec is None:
        load_dotenv(override=True)
        key = os.getenv("GEMINI_API_KEY")
        if not key:
            return {"error": "GEMINI_API_KEY not set."}
        genai.configure(api_key=key)

        prompt = f"""
    You are a Data Visualization AI Agent.
    Available columns: {list(df.columns)}
    Column dtypes: { {col: str(df[col].dtype) for col in df.columns} }
    User command: {command}

    Return ONLY a JSON object. No markdown. No explanation.
    Supported chart types: bar, barh, line, scatter, pie, histogram, area, box

    {{
      "chart_type": "<type>",
      "x": "<column or null>",
      "y": "<column or list of columns or null>",
      "title": "<descriptive title>",
      "xlabel": "<label or null>",
      "ylabel": "<label or null>",
      "top_n": <int or null>,
      "description": "<one sentence description>"
    }}

    Rules:
    - pie: x=label column, y=numeric column
    - histogram/box: x=null, y=numeric column
    - scatter: x and y must both be numeric columns
    - Only use columns that actually exist in the dataframe
    """
        try:
            res = model.generate_content(prompt)
            text = res.text.strip().removeprefix('```json').removesuffix('```').strip()
            spec = json.loads(text)
        except Exception as e:
            return {"error": f"AI parsing failed: {e}"}

    ct     = spec.get("chart_type", "bar")
    x_col  = spec.get("x")
    y_col  = spec.get("y")
    title  = spec.get("title", "Chart")
    xlabel = spec.get("xlabel") or x_col or ""
    ylabel = spec.get("ylabel") or (y_col if isinstance(y_col, str) else "")
    top_n  = spec.get("top_n")
    desc   = spec.get("description", "")

    try:
        d = df.copy()
        for col in d.columns:
            try: d[col] = pd.to_numeric(d[col])
            except: pass

        if top_n and isinstance(top_n, int):
            if isinstance(y_col, str) and y_col in d.columns:
                d = d.nlargest(top_n, y_col)
            else:
                d = d.head(top_n)

        fig, ax = plt.subplots(figsize=(9, 5))
        _apply_dark_style(fig, ax)

        if ct == "bar":
            cols = [y_col] if isinstance(y_col, str) else (y_col or [])
            xv = d[x_col].astype(str) if x_col else d.index.astype(str)
            for i, col in enumerate(cols):
                ax.bar(range(len(xv)), d[col], color=PALETTE[i % len(PALETTE)],
                       width=0.6, label=col, alpha=0.9, edgecolor=DARK_BG, linewidth=0.4)
            ax.set_xticks(range(len(xv)))
            ax.set_xticklabels(xv, rotation=35, ha='right', fontsize=8)
            if len(cols) > 1:
                ax.legend(facecolor=CARD_BG, edgecolor=GRID, labelcolor=TEXT_SEC, fontsize=8)

        elif ct == "barh":
            xv = d[x_col].astype(str) if x_col else d.index.astype(str)
            ax.barh(xv, d[y_col], color=PALETTE[0], alpha=0.9, edgecolor=DARK_BG, linewidth=0.4)

        elif ct in ("line", "area"):
            cols = [y_col] if isinstance(y_col, str) else (y_col or [])
            xv = d[x_col] if x_col else d.index
            for i, col in enumerate(cols):
                ax.plot(xv, d[col], color=PALETTE[i % len(PALETTE)], linewidth=2.2,
                        marker='o', markersize=4, label=col, alpha=0.9)
                if ct == "area":
                    ax.fill_between(xv, d[col], alpha=0.15, color=PALETTE[i % len(PALETTE)])
            if len(cols) > 1:
                ax.legend(facecolor=CARD_BG, edgecolor=GRID, labelcolor=TEXT_SEC, fontsize=8)

        elif ct == "scatter":
            ax.scatter(d[x_col], d[y_col], color=PALETTE[0], alpha=0.7,
                       s=40, edgecolors=DARK_BG, linewidths=0.4)

        elif ct == "pie":
            vals   = d[y_col].dropna()
            labels = d[x_col].astype(str) if x_col else vals.index.astype(str)
            wedges, texts, autotexts = ax.pie(
                vals, labels=labels, autopct='%1.1f%%',
                colors=PALETTE[:len(vals)], startangle=140,
                textprops={'color': TEXT_SEC, 'fontsize': 8},
                wedgeprops={'edgecolor': DARK_BG, 'linewidth': 1.2})
            for at in autotexts: at.set_color(TEXT_PRI)
            ax.set_facecolor(DARK_BG)

        elif ct == "histogram":
            ax.hist(d[y_col].dropna(), bins=25, color=PALETTE[0], alpha=0.85,
                    edgecolor=DARK_BG, linewidth=0.4)

        elif ct == "box":
            cols = [y_col] if isinstance(y_col, str) else (y_col or list(d.select_dtypes('number').columns))
            bp = ax.boxplot([d[c].dropna() for c in cols], patch_artist=True,
                            labels=cols, medianprops={'color': PALETTE[1], 'linewidth': 2})
            for patch, color in zip(bp['boxes'], PALETTE):
                patch.set_facecolor(color)
                patch.set_alpha(0.7)
            for el in ['whiskers', 'fliers', 'caps']:
                for item in bp[el]: item.set(color=TEXT_SEC, linewidth=1.2)

        ax.set_title(title, fontsize=13, fontweight='bold', pad=14, color=TEXT_PRI)
        ax.set_xlabel(xlabel, fontsize=9, color=TEXT_SEC)
        ax.set_ylabel(ylabel, fontsize=9, color=TEXT_SEC)
        return {"image_b64": _fig_to_b64(fig), "title": title, "description": desc}

    except Exception as e:
        return {"error": f"Chart rendering failed: {e}"}


# ─────────────────────────────────────────────────────────────
# Existing functions below
# ─────────────────────────────────────────────────────────────

def get_preview_stats(df: pd.DataFrame, include_expensive=True):
    """Returns a dictionary containing a sample of the data and some statistics."""
    total_rows = len(df)
    total_cols = len(df.columns)
    
    # Nulls are O(N), usually fast enough
    null_counts = df.isnull().sum().to_dict()
    
    # Duplicates are expensive — skip if include_expensive=False
    # or if dataset is massive (> 100k rows) and we're not in full analysis mode
    duplicates = 0
    if include_expensive:
        duplicates = int(df.duplicated().sum())
    else:
        # Fast preview mode
        duplicates = "N/A (Run Analysis)"

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
    load_dotenv(override=True)
    current_key = os.getenv("GEMINI_API_KEY")
    if not current_key:
        return [{"title": "API Key Missing", "description": "Configure GEMINI_API_KEY to see AI recommendations.", "severity": "medium", "icon": "⚠️", "action": "None"}]
    else:
        genai.configure(api_key=current_key)

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
    prompt = f"""
    You are an Intelligent Data Cleaning and Transformation Agent designed for production-grade data analysis workflows.
    The user wants to apply a transformation to a pandas DataFrame.
    Columns: {list(df.columns)}
    Command: {command}

    ========================
    CORE PRINCIPLES
    ===============
    1. NEVER overwrite existing data unless the user explicitly uses words like "overwrite", "replace", or "update".
    2. ALWAYS preserve original data for auditability and reproducibility.
    3. VALIDATE all operations before execution to prevent runtime failures.
    4. NEVER crash — always return structured, user-friendly responses.
    5. LOG every transformation step clearly.

    ========================
    DATA VALIDATION RULES
    =====================
    Before executing any operation check datatype compatibility. If mismatch: return structured error.

    ========================
    OUTPUT FORMAT
    =============
    For error:
    {{
    "status": "error",
    "column": "<column_name>",
    "issue": "<description>",
    "suggestion": "<actionable fix>"
    }}

    For success, you MUST also inject a 'backend_op' dictionary so the Python engine can actually execute it.
    {{
    "status": "success",
    "action": "<operation performed>",
    "new_column": "<column_name if created>",
    "column": "<column_name if overwritten>",
    "details": "<clear explanation>",
    "backend_op": {{
        "op": "<drop_duplicates|fill_nulls|math_scalar|math_columns|aggregate|rename_columns>",
        "col": "<primary column>",
        "val": "<val for scalar math or fill>",
        "col2": "<secondary column for column math>",
        "operator": "<+|-|*|/|%>",
        "agg_type": "<mean|sum|min|max|median>",
        "condition": {{
            "col": "<optional column to filter on>",
            "operator": "< > | < | == | >= | <= | != >",
            "val": "<optional filter value>"
        }}
    }}
    }}
    
    Return ONLY a single valid JSON object representing the operation. No markdown.
    """
    
    load_dotenv(override=True)
    current_key = os.getenv("GEMINI_API_KEY")
    if current_key:
        genai.configure(api_key=current_key)
    
    try:
        response = model.generate_content(prompt)
        text = response.text.strip().removeprefix('```json').removesuffix('```').strip()
        action = json.loads(text)
        
        if action.get("status") == "error":
            return df, f"Error on {action.get('column')}: {action.get('issue')}. {action.get('suggestion')}"

        backend_op = action.get("backend_op", {})
        op = backend_op.get("op")
        msg = action.get("details", "Transformation applied.")
        target_col = action.get("new_column") or action.get("column")
        source_col = backend_op.get("col")
        
        # Safely parse conditional logic row masks
        mask = slice(None)
        cond = backend_op.get("condition")
        if cond and cond.get("col") in df.columns:
            c_col = cond.get("col")
            c_op = cond.get("operator")
            c_val = cond.get("val")
            c_series = pd.to_numeric(df[c_col], errors="coerce") if isinstance(c_val, (int, float)) else df[c_col]
            try: c_val = float(c_val) if isinstance(c_val, (int, float)) else float(c_val) if str(c_val).replace('.','',1).isdigit() else c_val
            except: pass
            if c_op == ">": mask = c_series > c_val
            elif c_op == "<": mask = c_series < c_val
            elif c_op == "==": mask = c_series == c_val
            elif c_op == ">=": mask = c_series >= c_val
            elif c_op == "<=": mask = c_series <= c_val
            elif c_op == "!=": mask = c_series != c_val

        if op == "drop_duplicates":
            df.drop_duplicates(inplace=True)
            
        elif op == "drop_empty_rows":
            df.dropna(how="all", inplace=True)
            
        elif op == "fill_nulls":
            if source_col in df.columns:
                df.loc[mask, source_col] = df.loc[mask, source_col].fillna(backend_op.get("val", 0))
                
        elif op == "math_scalar":
            if source_col in df.columns and target_col:
                val = float(backend_op.get("val", 0))
                operator = backend_op.get("operator", "+")
                source_data = pd.to_numeric(df.loc[mask, source_col], errors="coerce")
                if operator == "+": df.loc[mask, target_col] = source_data + val
                elif operator == "-": df.loc[mask, target_col] = source_data - val
                elif operator == "*": df.loc[mask, target_col] = source_data * val
                elif operator == "/": df.loc[mask, target_col] = source_data / val
                elif operator == "%": df.loc[mask, target_col] = source_data % val
                
        elif op == "math_columns":
            col2 = backend_op.get("col2")
            if source_col in df.columns and col2 in df.columns and target_col:
                operator = backend_op.get("operator", "+")
                c1 = pd.to_numeric(df.loc[mask, source_col], errors="coerce")
                c2 = pd.to_numeric(df.loc[mask, col2], errors="coerce")
                if operator == "+": df.loc[mask, target_col] = c1 + c2
                elif operator == "-": df.loc[mask, target_col] = c1 - c2
                elif operator == "*": df.loc[mask, target_col] = c1 * c2
                elif operator == "/": df.loc[mask, target_col] = c1 / c2
                elif operator == "%": df.loc[mask, target_col] = c1 % c2
                
        elif op == "aggregate":
            if source_col in df.columns:
                agg_type = backend_op.get("agg_type", "mean")
                c1 = pd.to_numeric(df[source_col], errors="coerce")
                if agg_type == "mean": val = c1.mean()
                elif agg_type == "sum": val = c1.sum()
                elif agg_type == "min": val = c1.min()
                elif agg_type == "max": val = c1.max()
                elif agg_type == "median": val = c1.median()
                else: val = "N/A"
                msg = f"The {agg_type} of {source_col} is {val}."

        return df, msg
        
    except Exception as e:
        print(f"Error executing NLP transform: {e}")
        return df, f"Error: Unable to process command. ({e})"
