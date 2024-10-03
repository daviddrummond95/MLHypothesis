import os
import yaml
from fastapi import FastAPI, Request, Form, Body, HTTPException
from fastapi.responses import JSONResponse, HTMLResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from utils.hypothesis.save_hypothesis import get_all_hypotheses, get_pipeline_results, get_hypothesis, get_hypothesis_results
from utils.hypothesis.validate_hypothesis import validate_hypothesis
import markdown2  # Add this import
import re  # Add this import
from utils.hypothesis.save_causals import get_causal_results, get_report_from_database
from utils.hypothesis.test_hypothesis import test_multiple_hypotheses
from models.causal import perform_linear_regression, perform_random_forest
import json

app = FastAPI()

# Set up Jinja2 templates
templates = Jinja2Templates(directory="ui/templates")

def preprocess_markdown(text):
    if not text:
        return ""
    # Replace "\n" followed by a number with two newlines and the number
    processed_text = re.sub(r'\n(\d)', r'\n\n\1', text)
    # Ensure all other newlines are preserved
    processed_text = processed_text.replace('\n', '  \n')
    return processed_text

# Add markdown filter to Jinja2 environment
templates.env.filters["markdown"] = lambda text: markdown2.markdown(preprocess_markdown(text), extras=["break-on-newline"])

# Database path
DB_PATH = "hypotheses.db"

# Load model configuration
with open("configs/pred_model.yaml", "r") as f:
    model_config = yaml.safe_load(f)

@app.get("/")
async def home(request: Request):
    pipeline_results = get_pipeline_results(DB_PATH)
    hypotheses = get_all_hypotheses(DB_PATH)
    
    # Filter out feature importances of 0
    filtered_feature_importance = [
        fi for fi in pipeline_results['feature_importance']
        if fi['importance'] > 0
    ]
    
    # Sort feature importance in descending order
    filtered_feature_importance.sort(key=lambda x: x['importance'], reverse=True)
    
    # Update pipeline_results with filtered data
    pipeline_results['feature_importance'] = filtered_feature_importance

    # Get number of features
    num_features = len(pipeline_results['feature_importance'])

    return templates.TemplateResponse("home.html", {
        "request": request,
        "pipeline_results": pipeline_results,
        "hypotheses": hypotheses,
        "model_config": model_config,
        "num_features": num_features
    })

@app.get("/hypothesis/{hypothesis_id}")
async def view_hypothesis(request: Request, hypothesis_id: int):
    hypothesis = get_hypothesis(DB_PATH, hypothesis_id)
    if hypothesis is None:
        raise HTTPException(status_code=404, detail="Hypothesis not found")

    features = hypothesis['relevant_features']
    if isinstance(features, str):
        features = [feature.strip() for feature in features.split(',')]
    
    # Initialize result containers
    hypothesis_test_results = {}
    lr_coefficients = {}
    rf_importances = {}
    causal_results = {}

    for feature in features:
        # Fetch all results for this feature
        results = get_hypothesis_results(DB_PATH, feature)
        hypothesis_test_results[feature] = results.get('hypothesis_test', 'N/A')
        lr_coefficients[feature] = results.get('linear_regression', 'N/A')
        rf_importances[feature] = results.get('random_forest', 'N/A')
        
        # Fetch causal results
        feature_causal_results = get_causal_results(DB_PATH, feature)
        if feature in feature_causal_results:
            causal_results[feature] = json.dumps(feature_causal_results[feature], indent=2)
        else:
            causal_results[feature] = 'N/A'

    # Fetch the generated report
    report = get_report_from_database(DB_PATH, hypothesis_id)

    return templates.TemplateResponse("hypothesis.html", {
        "request": request,
        "hypothesis": hypothesis,
        "hypothesis_test_results": hypothesis_test_results,
        "lr_coefficients": lr_coefficients,
        "rf_importances": rf_importances,
        "causal_results": causal_results,
        "generated_report": report  # Add this line
    })

@app.post("/approve_hypothesis/{hypothesis_id}")
async def approve_hypothesis(hypothesis_id: int, request: Request):
    try:
        # Try to get form data
        form_data = await request.form()
        reason = form_data.get("reason", "")
    except:
        # If form data fails, try to get JSON data
        json_data = await request.json()
        reason = json_data.get("reason", "")

    validate_hypothesis(DB_PATH, hypothesis_id, None, "Approved", reason)
    hypothesis = get_hypothesis(DB_PATH, hypothesis_id)
    return templates.TemplateResponse("hypothesis.html", {"request": request, "hypothesis": hypothesis})

@app.post("/decline_hypothesis/{hypothesis_id}")
async def decline_hypothesis(hypothesis_id: int, request: Request):
    validate_hypothesis(DB_PATH, hypothesis_id, None, "Declined", "")
    hypothesis = get_hypothesis(DB_PATH, hypothesis_id)
    return templates.TemplateResponse("hypothesis.html", {"request": request, "hypothesis": hypothesis})

@app.get("/all_hypotheses")
def all_hypotheses():
    hypotheses = get_all_hypotheses(DB_PATH)
    return hypotheses

@app.post("/submit_hypothesis")
async def submit_hypothesis(request: Request):
    form_data = await request.form()
    hypothesis = {
        "statement": form_data.get("statement"),
        "rationale": form_data.get("rationale"),
        "relevant_features": form_data.get("relevant_features"),
        "expected_effect": form_data.get("expected_effect"),
        "confidence_level": float(form_data.get("confidence_level"))
    }
    # Here you would save the hypothesis to the database
    return {"status": "success", "message": "Hypothesis submitted successfully."}

@app.get("/hypotheses")
async def all_hypotheses(request: Request):
    hypotheses = get_all_hypotheses(DB_PATH)
    return templates.TemplateResponse("hypotheses.html", {
        "request": request,
        "hypotheses": hypotheses
    })

def load_and_prepare_data(db_path):
    # Implement this function to load and prepare the data
    # It should return X_train and y_train
    pass

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)