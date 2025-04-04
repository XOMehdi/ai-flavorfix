from flask import Flask, render_template, request, jsonify
import pandas as pd
import joblib
import numpy as np
import requests

app = Flask(__name__)

# Load unique ingredients from CSV
try:
    unique_ingredients = pd.read_csv(
        'model/unique_ingredients.csv', header=None)[0].str.lower().tolist()
except Exception as e:
    print(f"Error loading unique ingredients: {e}")
    unique_ingredients = []

# Load your recipe dataset
try:
    recipes_df = pd.read_csv('model/data.csv')  # Adjust the path as necessary
except Exception as e:
    print(f"Error loading recipes dataset: {e}")
    recipes_df = pd.DataFrame()


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/search_recipes', methods=['POST'])
def search_recipes():
    query = request.json.get('query', '').lower()

    # Filter recipes based on the query
    matching_recipes = recipes_df[recipes_df['TranslatedRecipeName'].str.contains(
        query, case=False, na=False)]

    recipe_list = matching_recipes[['TranslatedRecipeName', 'TotalTimeInMins', 'Cuisine',
                                    'image-url', 'TranslatedIngredients', 'TranslatedInstructions']].to_dict(orient='records')

    return jsonify(recipe_list)


@app.route('/ingredients')
def ingredients_page():
    return render_template('ingredients.html')


@app.route('/recipes')
def recipes_page():
    return render_template('recipes.html')


@app.route('/search_ingredients', methods=['POST'])
def search_ingredients():
    query = request.json.get('query', '').lower()
    results = [ing for ing in unique_ingredients if query in ing]
    return jsonify(results)


@app.route('/get_recipe', methods=['POST'])
def get_recipe():
    ingredients = request.json.get('ingredients', [])

    api_url = 'http://127.0.0.1:8000/recommendations/'

    try:
        # Make a POST request to the API
        response = requests.post(api_url, json={"ingredients": ingredients})

        # Raise an error if the response status code is not 200
        response.raise_for_status()

        # Extract JSON data from the response
        recommendations = response.json()
        return jsonify(recommendations)

    except requests.exceptions.RequestException as e:
        return jsonify({"error": str(e)}), 500


if __name__ == '__main__':
    app.run(debug=True)
