"""
Author: XOMehdi
Created on: November 6, 2024
Description:
    This script implements the backend API for the AI FlavorFix project. 
    The API leverages the k-Nearest Neighbors (kNN) algorithm to suggest 
    recipes based on user-input ingredients, returning recipes 
    with similar ingredients and detailed instructions for preparation.

Features:
    - Ingredient-based recipe recommendations using a pre-trained kNN model.
    - Cross-origin resource sharing (CORS) setup for frontend integration.
    - Endpoints for recipe recommendations, supported ingredients, and health checks.

Dependencies:
    - FastAPI
    - Scikit-learn
    - Pandas
    - Numpy
    - Pydantic
    - CORS Middleware for FastAPI

Instructions:
    - Ensure the pre-trained kNN model (`knn_model.pkl`) and MultiLabelBinarizer (`multilabel_binarizer.pkl`) 
      are placed in the `models/` directory.
    - Load recipe data from `data/processed_recipes.csv`.
"""


from fastapi import FastAPI, HTTPException, Depends
from pydantic import BaseModel
from typing import List
import numpy as np
import pandas as pd
import pickle
from fastapi.middleware.cors import CORSMiddleware

# Initialize FastAPI app with metadata for API documentation
app = FastAPI(
    title="AI FlavorFix API",
    description="Recipe recommendation API based on available ingredients",
    version="1.0.0"
)

# Add CORS middleware to allow cross-origin requests, helpful for frontend communication
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Define Pydantic models for request validation and response structure
class IngredientsInput(BaseModel):
    """Schema for the ingredients input provided by the user."""
    ingredients: List[str]


class RecipeResponse(BaseModel):
    """Schema for individual recipe details in the recommendation response."""
    recipe_id: int  # Unique identifier for each recipe (using row index as ID)
    title: str  # Recipe title/name
    ingredients: List[str]  # Ingredients list in the recipe
    instructions: str  # Recipe preparation instructions
    cooking_time: float  # Total cooking time in minutes
    cuisine: str
    similarity_score: float  # Similarity score based on input ingredients
    url: str


class RecipeList(BaseModel):
    """Schema for list of recipe recommendations."""
    recipes: List[RecipeResponse]  # List of recommended recipes


# Global variables for model, binarizer, and data
model = None
mlb = None
scaler = None
recipe_data = None


def load_model():
    """Load the pre-trained kNN model, MultiLabelBinarizer, and recipe dataset.

    Returns:
        bool: True if all components are loaded successfully, False otherwise.
    """
    global model, mlb, scaler, recipe_data

    try:
        # Load the pre-trained knn model
        with open('model/knn_model.pkl', 'rb') as f:
            model = pickle.load(f)

        # Load the MultiLabelBinarizer for ingredient encoding
        with open('model/multilabel_binarizer.pkl', 'rb') as f:
            mlb = pickle.load(f)

        with open('model/scaler.pkl', 'rb') as f:
            scaler = pickle.load(f)

        # Load recipe data from CSV file
        recipe_data = pd.read_csv('data/processed_recipes.csv')

        return True
    except Exception as e:
        print(f"Error loading model: {str(e)}")
        return False


@app.on_event("startup")
async def startup_event():
    """Event handler to initialize the model and data when the API starts."""
    if not load_model():
        raise Exception("Failed to load model and data")


def get_feature_vector(ingredients: List[str]) -> np.ndarray:
    """Convert user input ingredients to binary vector for model compatibility.

    Args:
        ingredients (List[str]): List of ingredients input by the user.

    Returns:
        np.ndarray: Binary vector representing the ingredients and their feature weights.
    """
    # Load feature weights
    with open('model/feature_weights.pkl', 'rb') as f:
        weights = pickle.load(f)

    # Convert ingredients list into binary vector using the trained MultiLabelBinarizer
    ingredient_vector = mlb.transform([ingredients])

    # Create additional features
    additional_features = scaler.transform([[
        len(ingredients),
        recipe_data['cooking_time'].mean(),
        len(ingredients) * recipe_data['cooking_time'].mean()
    ]])

    # Combine vectors with weights
    return np.hstack((
        weights['ingredient_weight'] * ingredient_vector.toarray(),
        weights['additional_features_weight'] * additional_features
    ))


def find_similar_recipes(feature_vector: np.ndarray, n_neighbors: int = 5) -> List[RecipeResponse]:
    """Use the kNN model to find recipes similar to the input ingredients.

    Args:
        ingredient_vector (np.ndarray): Binary vector of user input ingredients.
        n_neighbors (int): Number of similar recipes to retrieve.

    Returns:
        List[RecipeResponse]: List of recommended recipes based on similarity score.
    """
    distances, indices = model.kneighbors(
        feature_vector, n_neighbors=n_neighbors)

    recommendations = []
    for i in range(len(indices[0])):
        recipe_idx = indices[0][i]
        recipe = recipe_data.iloc[recipe_idx]

        # Append recipe data to recommendations with similarity score calculated from distance
        recommendations.append(
            RecipeResponse(
                # Using row index as unique recipe ID
                recipe_id=int(recipe_idx),

                title=recipe['title'],
                ingredients=eval(recipe['ingredients']),
                instructions=recipe['instructions'],
                cooking_time=float(recipe['cooking_time']),
                cuisine=recipe['cuisine'],

                # Similarity as inverse of distance
                similarity_score=float(1 - distances[0][i]),
                url=recipe['url']
            )
        )

    return recommendations


@app.post("/recommendations/", response_model=RecipeList)
async def get_recommendations(input_data: IngredientsInput):
    """Endpoint to get recipe recommendations based on input ingredients.

    Args:
        input_data (IngredientsInput): User-provided list of ingredients.

    Returns:
        RecipeList: List of recipe recommendations.
    """
    try:
        # Convert ingredients to feature vector
        feature_vector = get_feature_vector(input_data.ingredients)

        # Get similar recipes using the kNN model
        recommendations = find_similar_recipes(feature_vector)

        return RecipeList(recipes=recommendations)

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/health/")
async def health_check():
    """Health check endpoint to verify API and model are functioning."""
    return {"status": "healthy", "model_loaded": model is not None}


@app.get("/ingredients/")
async def get_supported_ingredients():
    """Endpoint to retrieve list of all supported ingredients for recommendations."""
    return {"ingredients": list(mlb.classes_)}
