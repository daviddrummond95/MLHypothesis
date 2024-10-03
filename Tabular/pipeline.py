# Get Data
import pandas as pd
import numpy as np
import featuretools as ft
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from models.synth_data_gen import train_vae, generate_synthetic_data
from models.morphing import morph_profile, enforce_constraints, calculate_similarity
from pipelines.hypothesis_gen import generate_and_analyze_hypotheses
import logging
from causalml.inference.tree import UpliftTreeClassifier, uplift_tree_plot
from causalml.inference.meta import XGBTRegressor
from causalml.inference.meta import BaseXRegressor
from causalml.metrics import plot_gain
from IPython.display import Image
import matplotlib.pyplot as plt

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


########################################################
# SET TARGET
target = "Attrition_Yes"
target_definition = f"The target variable '{target}' represents employee attrition (Yes/No)."
########################################################

data = pd.read_csv('/Users/daviddrummond/MLHypothesis/Tabular/WA_Fn-UseC_-HR-Employee-Attrition.csv')
print(data.head())

# Preprocess Data
def preprocess_data(df):
    # Identify numeric and categorical columns
    numeric_features = df.select_dtypes(include=['int64', 'float64']).columns
    categorical_features = df.select_dtypes(include=['object']).columns

    # Create preprocessing pipelines
    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])

    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ])

    # Combine preprocessing steps
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features)
        ])

    # Fit and transform the data
    preprocessed_data = preprocessor.fit_transform(df)
    
    # Convert to DataFrame and set column names
    feature_names = (numeric_features.tolist() + 
                     preprocessor.named_transformers_['cat']
                     .named_steps['onehot']
                     .get_feature_names_out(categorical_features).tolist())
    
    preprocessed_df = pd.DataFrame(preprocessed_data, columns=feature_names, index=df.index)
    
    return preprocessed_df, preprocessor

# Engineer Features
def engineer_features(df):
    # Create an EntitySet
    es = ft.EntitySet(id="hr_data")
    es = es.add_dataframe(
        dataframe_name="employees",
        dataframe=df,
        index="EmployeeNumber",  # Assuming EmployeeNumber is a unique identifier
        time_index="YearsAtCompany"  # Using YearsAtCompany as a proxy for time
    )

    # Define feature primitives
    agg_primitives = ["count", "mean", "max", "min", "std"]
    trans_primitives = ["year", "month", "day", "weekday", "is_weekend"]

    # Run deep feature synthesis
    feature_matrix, feature_defs = ft.dfs(
        entityset=es,
        target_dataframe_name="employees",
        agg_primitives=agg_primitives,
        trans_primitives=trans_primitives,
        max_depth=2,
        features_only=False,
        verbose=True
    )

    # Encode categorical features
    feature_matrix_enc, features_enc = ft.encode_features(feature_matrix, feature_defs)

    return feature_matrix_enc, features_enc

# Apply preprocessing and feature engineering
preprocessed_data, preprocessor = preprocess_data(data)
engineered_data, engineered_features = engineer_features(preprocessed_data)

print("Preprocessed data shape:", preprocessed_data.shape)
print("Engineered data shape:", engineered_data.shape)
print("Number of engineered features:", len(engineered_features))

# Split the data
train_data, test_data = train_test_split(engineered_data, test_size=0.2, random_state=42)

# Synthetic Data Generation
logging.info("Starting synthetic data generation")
try:
    vae, encoder, decoder, scaler, columns = train_vae(train_data)
    synthetic_df = generate_synthetic_data(decoder, len(train_data), 10, scaler, columns)
    if synthetic_df is not None:
        logging.info(f"Generated synthetic data shape: {synthetic_df.shape}")
    else:
        logging.warning("Failed to generate synthetic data")
except Exception as e:
    logging.error(f"Error in synthetic data generation: {str(e)}")
    synthetic_df = None

# Morphing Data
logging.info("Starting data morphing process")
original_profile = train_data.iloc[0].to_dict()
morphed_profile = morph_profile(original_profile, direction="increase")
morphed_profile = enforce_constraints(morphed_profile)
similarity = calculate_similarity(original_profile, morphed_profile)
logging.info(f"Similarity between original and morphed profile: {similarity}")

# Hypothesis Generation and Causal Inference
logging.info("Starting hypothesis generation, analysis, and causal inference")
if synthetic_df is not None:
    hypotheses = generate_and_analyze_hypotheses(synthetic_df, train_data, target_definition)
    logging.info(f"Generated and analyzed {len(hypotheses)} hypotheses")
    
    # Log a few sample hypotheses
    for i, hypothesis in enumerate(hypotheses[:5]):  # Log first 5 hypotheses
        logging.info(f"Hypothesis {i + 1}:")
        logging.info(f"  Statement: {hypothesis['statement']}")
        logging.info(f"  Rationale: {hypothesis['rationale']}")
        logging.info(f"  LLM Explanation: {hypothesis['llm_explanation']}")
        logging.info(f"  LLM Validity: {hypothesis['llm_validity']}")
    
    # Causal Inference using Uplift Tree and XGBoost for valid hypotheses
    valid_hypotheses = [h for h in hypotheses if h['llm_validity']]
    logging.info(f"Number of valid hypotheses: {len(valid_hypotheses)}")

    for i, hypothesis in enumerate(valid_hypotheses):
        logging.info(f"Performing causal inference for Hypothesis {i + 1}")
        relevant_features = hypothesis['relevant_features']
        
        # Prepare full feature set for uplift modeling
        all_features = train_data.columns.tolist()
        all_features.remove(target)  # Remove the target variable from features
        X_full = train_data[all_features]

        # Ensure all features are numeric
        X_full = X_full.select_dtypes(include=[np.number])

        # Prepare data for causal inference
        treatment = train_data[target].astype(int)  # Convert treatment to integer
        y = train_data['Attrition_Yes'].astype(int)  # Ensure outcome is integer
        
        try:
            # Try Uplift Tree first
            uplift_model = UpliftTreeClassifier(max_depth=5, min_samples_leaf=200, min_samples_treatment=50,
                                                n_reg=100, evaluationFunction='KL', control_name=0)
            uplift_model.fit(X_full, treatment=treatment, y=y)
            
            # Generate Uplift Tree Visualization
            graph = uplift_tree_plot(uplift_model.fitted_uplift_tree, X_full.columns.tolist())
            img = Image(graph.create_png())
            
            # Save the image
            img_path = f"uplift_tree_hypothesis_{i+1}.png"
            with open(img_path, "wb") as f:
                f.write(img.data)
            logging.info(f"Uplift Tree visualization saved as {img_path}")
            
            # Feature Importances
            feature_importances = pd.Series(uplift_model.feature_importances_, index=X_full.columns).sort_values(ascending=False)
            
        except Exception as e:
            logging.warning(f"Uplift Tree failed for Hypothesis {i + 1}. Trying XGBoost. Error: {str(e)}")
            
            # Use XGBoost as a fallback
            xgb_model = BaseXRegressor(learner=XGBTRegressor(random_state=42))
            xgb_model.fit(X=X_full, treatment=treatment, y=y)
            
            # Generate XGBoost feature importances
            feature_importances = pd.Series(xgb_model.feature_importances_, index=X_full.columns).sort_values(ascending=False)
            
            # Generate Uplift Curve
            plot_gain(xgb_model, X_full, treatment, y)
            plt.title(f"Uplift Curve for Hypothesis {i+1}")
            img_path = f"uplift_curve_hypothesis_{i+1}.png"
            plt.savefig(img_path)
            plt.close()
            logging.info(f"Uplift Curve saved as {img_path}")
        
        # Highlight relevant features
        plt.figure(figsize=(12, 8))
        colors = ['red' if feature in relevant_features else 'blue' for feature in feature_importances.index]
        feature_importances.plot(kind='barh', color=colors)
        plt.title(f"Feature Importances for Hypothesis {i+1}\n(Hypothesis-relevant features in red)")
        plt.tight_layout()
        imp_path = f"feature_importances_hypothesis_{i+1}.png"
        plt.savefig(imp_path)
        plt.close()
        logging.info(f"Feature importances plot saved as {imp_path}")
        
        # Update hypothesis with causal inference results
        hypothesis['uplift_model_path'] = img_path
        hypothesis['feature_importances_path'] = imp_path
        hypothesis['feature_importances'] = feature_importances.to_dict()
        
        # Add analysis of hypothesis-relevant features
        relevant_importances = feature_importances[feature_importances.index.isin(relevant_features)]
        hypothesis['relevant_feature_analysis'] = relevant_importances.to_dict()
        
        logging.info(f"Causal inference completed for Hypothesis {i + 1}")

# Reporting
logging.info("Pipeline execution complete")
print("Pipeline execution complete")
