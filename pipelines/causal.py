import logging
from langchain_openai import ChatOpenAI
from concurrent.futures import ThreadPoolExecutor
import re
import os
from models.predictive import calculate_feature_importance
from models.morphing import morph_profile, enforce_constraints, calculate_similarity
from models.synth_data_gen import generate_synthetic_data
from utils.hypothesis.save_hypothesis import get_hypothesis_results
from utils.hypothesis.save_causals import get_causal_results
import json

logger = logging.getLogger(__name__)

# Set up OpenAI API
model = ChatOpenAI(model='gpt-4')

# Documentation storage
DOCUMENTATION_FILE = 'documentation/methodology_storage.json'

def read_file_content(file_path):
    try:
        with open(file_path, 'r') as file:
            return file.read()
    except Exception as e:
        logger.error(f"Error reading file {file_path}: {str(e)}")
        return "Error: Could not read file content"

def load_documentation():
    if os.path.exists(DOCUMENTATION_FILE):
        with open(DOCUMENTATION_FILE, 'r') as f:
            return json.load(f)
    return {}

def save_documentation(documentation):
    os.makedirs(os.path.dirname(DOCUMENTATION_FILE), exist_ok=True)
    with open(DOCUMENTATION_FILE, 'w') as f:
        json.dump(documentation, f, indent=2)

def generate_report_section(section_type, component, hypothesis_data, pipeline_results, additional_data, domain_files, data_context_files):
    documentation = load_documentation()
    methodology_key = f"{component}_methodology"
    
    domain_content = "\n".join([f"File: {file}\nContent:\n{read_file_content(file)}\n" for file in domain_files])
    data_context_content = "\n".join([f"File: {file}\nContent:\n{read_file_content(file)}\n" for file in data_context_files])
    
    system_prompt = """You are a medical research assistant tasked with writing scientific reports on hypotheses related to HIV treatment outcomes. You will be provided with:

    1. The section type you need to write (introduction, methodology, results, or conclusion)
    2. The component of the pipeline you're focusing on
    3. Hypothesis data including the statement, rationale, relevant features, expected effect, and confidence level
    4. Pipeline results including hypothesis testing, linear regression, random forest, and causal inference results
    5. Additional data specific to each pipeline component
    6. The actual code used for the component (when available)
    7. Domain-specific information
    8. Data context information

    Your task is to generate a comprehensive, scientifically sound section for the report based on this information. Use a formal, academic tone and ensure your writing is clear and concise. When discussing the methodology, refer to the provided code to explain the process accurately. Incorporate relevant domain knowledge and data context where appropriate.

    If a methodology for this component already exists in the documentation, use it as a base and adapt it to the current hypothesis if necessary.

    Response Format:
    <section>
    Your generated section content here. This should be a fully formed, paragraph-style text suitable for inclusion in a scientific report.
    </section>
    """

    human_prompt = f"""
    Section Type: {section_type}
    Component: {component}
    Hypothesis: {hypothesis_data['statement']}
    Rationale: {hypothesis_data['rationale']}
    Relevant Features: {', '.join(hypothesis_data['relevant_features'])}
    Expected Effect: {hypothesis_data['expected_effect']}
    Confidence Level: {hypothesis_data['confidence_level']}

    Pipeline Results:
    Hypothesis Testing: {pipeline_results.get('hypothesis_test', 'N/A')}
    Linear Regression: {pipeline_results.get('linear_regression', 'N/A')}
    Random Forest: {pipeline_results.get('random_forest', 'N/A')}
    Causal Inference: {pipeline_results.get('causal_inference', 'N/A')}

    Additional Data:
    {additional_data['data']}

    Component Code:
    {additional_data['code']}

    Domain Information:
    {domain_content}

    Data Context:
    {data_context_content}

    Existing Methodology:
    {documentation.get(methodology_key, 'No existing methodology found.')}

    Please generate the {section_type} section for the report on this hypothesis, focusing on the {component} component. Incorporate details from the provided code when explaining the methodology. Use the domain information and data context to enrich your explanation where relevant. If an existing methodology is provided, use it as a base and adapt it to the current hypothesis if necessary.
    """

    response = model.invoke([("system", system_prompt), ("human", human_prompt)]).content
    
    section_content = re.search(r'<section>(.*?)</section>', response, re.DOTALL)
    content = section_content.group(1).strip() if section_content else None
    
    if content and section_type == 'methodology':
        documentation[methodology_key] = content
        save_documentation(documentation)
    
    return content

def generate_full_report(hypothesis_data, pipeline_results, domain_files, data_context_files):
    components = ['Data Analysis', 'Hypothesis Testing', 'Causal Inference', 'Conclusion']
    sections = ['introduction', 'methodology', 'results', 'conclusion']
    
    report = {}
    
    with ThreadPoolExecutor(max_workers=5) as executor:
        for component in components:
            report[component] = {}
            additional_data = get_additional_data(component, hypothesis_data, pipeline_results)
            futures = {executor.submit(generate_report_section, section, component, hypothesis_data, pipeline_results, additional_data, domain_files, data_context_files): section for section in sections}
            for future in futures:
                section = futures[future]
                report[component][section] = future.result()
    
    return report

def get_additional_data(component, hypothesis_data, pipeline_results):
    additional_data = {'data': '', 'code': ''}
    
    if component == 'Data Analysis':
        additional_data['data'] = f"""
        Relevant Features: {', '.join(hypothesis_data['relevant_features'])}
        """
        additional_data['code'] = read_file_content('pipelines/data_prep.py')
    elif component == 'Hypothesis Testing':
        additional_data['data'] = f"""
        Hypothesis Test Results: {pipeline_results.get('hypothesis_test', 'N/A')}
        """
        additional_data['code'] = read_file_content('utils/hypothesis/test_hypothesis.py')
    elif component == 'Causal Inference':
        additional_data['data'] = f"""
        Causal Inference Results: {pipeline_results.get('causal_inference', 'N/A')}
        Linear Regression Results: {pipeline_results.get('linear_regression', 'N/A')}
        Random Forest Results: {pipeline_results.get('random_forest', 'N/A')}
        """
        additional_data['code'] = read_file_content('models/causal.py')
    elif component == 'Conclusion':
        additional_data['data'] = f"""
        Overall Results Summary:
        Hypothesis Test: {pipeline_results.get('hypothesis_test', 'N/A')}
        Causal Inference: {pipeline_results.get('causal_inference', 'N/A')}
        """
        # No specific code for conclusion
    
    return additional_data

def format_report(report):
    formatted_report = ""
    for component, sections in report.items():
        formatted_report += f"# {component}\n\n"
        for section, content in sections.items():
            formatted_report += f"## {section.capitalize()}\n\n{content}\n\n"
    return formatted_report

def generate_hypothesis_report(hypothesis_data, pipeline_results, domain_files=[], data_context_files=[]):
    """
    Generate a full report for a given hypothesis using data from the paper pipeline.
    
    Args:
        hypothesis_data (dict): Data about the hypothesis including statement, rationale, etc.
        pipeline_results (dict): Results from the paper pipeline including hypothesis testing, regression, etc.
        domain_files (list): List of file paths containing domain-specific information.
        data_context_files (list): List of file paths containing data context information.
    
    Returns:
        str: Formatted report for the hypothesis
    """
    try:
        report = generate_full_report(hypothesis_data, pipeline_results, domain_files, data_context_files)
        formatted_report = format_report(report)
        return formatted_report
    except Exception as e:
        logger.error(f"An error occurred while generating the hypothesis report: {str(e)}")
        return None

