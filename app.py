from flask import Flask, render_template, request, send_file, jsonify, redirect, url_for
import pickle
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from flask_cors import CORS
from io import BytesIO
import base64
import pickle
import json
import os
import openai

app = Flask(__name__)
CORS(app)

openai.api_key = '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'

# Load user_main_category list from JSON file
with open('categories.json', 'r') as json_file:
    user_main_categories = json.load(json_file)

# Load County models and transformers
with open('models\county\county_column_transformer.pkl', 'rb') as county_column_transformer_file:
    county_column_transformer = pickle.load(county_column_transformer_file)
with open('models\county\county_label_encoder.pkl', 'rb') as county_label_encoder_file:
    county_label_encoder = pickle.load(county_label_encoder_file)
with open('models\county\county_scalar.pkl', 'rb') as county_scalar_file:
    county_scalar = pickle.load(county_scalar_file)
with open('models\county\county_model.pkl', 'rb') as county_model_file:
    county_model = pickle.load(county_model_file)

# Load Gender models and transformers
with open('models\gender\gender_column_transformer.pkl', 'rb') as gender_column_transformer_file:
    gender_column_transformer = pickle.load(gender_column_transformer_file)
with open('models\gender\gender_label_encoder.pkl', 'rb') as gender_label_encoder_file:
    gender_label_encoder = pickle.load(gender_label_encoder_file)
with open('models\gender\gender_scalar.pkl', 'rb') as gender_scalar_file:
    gender_scalar = pickle.load(gender_scalar_file)
with open('models\gender\gender_model.pkl', 'rb') as gender_model_file:
    gender_model = pickle.load(gender_model_file)


# Load Age Group models and transformers
with open('models\Age_group\Age_column_transformer.pkl', 'rb') as age_column_transformer_file:
    age_column_transformer = pickle.load(age_column_transformer_file)
with open('models\Age_group\Age_label_encoder.pkl', 'rb') as age_label_encoder_file:
    age_label_encoder = pickle.load(age_label_encoder_file)
with open('models\Age_group\Age_scalar.pkl', 'rb') as age_scalar_file:
    age_scalar = pickle.load(age_scalar_file)
with open('models\Age_group\Age_model.pkl', 'rb') as age_model_file:
    age_model = pickle.load(age_model_file)

# Load Demand models and transformers
with open('models\demand\demand_encoder.pkl', 'rb') as demand_encoder_file:
    demand_encoder = pickle.load(demand_encoder_file)
with open('models\demand\demand_model.pkl', 'rb') as demand_model_file:
    demand_model = pickle.load(demand_model_file)

# Load Time models and transformers
with open('models\Time\Time_encoder.pkl', 'rb') as time_encoder_file:
    time_encoder = pickle.load(time_encoder_file)
with open('models\Time\Time_model.pkl', 'rb') as time_model_file:
    time_model = pickle.load(time_model_file)


# Create a mapping dictionary
county_category_mapping = dict(zip(county_label_encoder.transform(county_label_encoder.classes_), county_label_encoder.classes_))
gender_category_mapping = dict(zip(gender_label_encoder.transform(gender_label_encoder.classes_), gender_label_encoder.classes_))
age_category_mapping = dict(zip(age_label_encoder.transform(age_label_encoder.classes_), age_label_encoder.classes_))


# Route to serve JSON file
@app.route('/get_dropdown_options')
def get_dropdown_options():
    return send_file('dropdownOptions.json', mimetype='application/json')

@app.route('/get_dropdown_dates')
def get_dropdown_dates():
    return send_file('dropdownDates.json', mimetype='application/json')

@app.route('/about', methods=['GET', 'POST'])
def about():
    if request.method == 'POST':
        main_category = request.form['main_category']
        sub_category = request.form['sub_category']
        
        #predict country
        county_data = [[main_category, sub_category]]
        county_data = county_column_transformer.transform(county_data)
        county_data[:, 16:] = county_scalar.transform(county_data[:, 16:])
        
        county_predicted_probs = county_model.predict_proba(county_data)
        top_county_index = np.argmax(county_predicted_probs[0])
        top_county_category = county_category_mapping[top_county_index]
        top3_county_indices = np.argsort(county_predicted_probs[0])[::-1][:3]
        top3_county_categories = [county_category_mapping[idx] for idx in top3_county_indices]

        #predict gender
        gender_data = [[main_category, sub_category,top_county_category]]
        gender_data = gender_column_transformer.transform(gender_data)
        gender_data[:, 16:] = gender_scalar.transform(gender_data[:, 16:])

        gender_predicted_probs = gender_model.predict_proba(gender_data)
        top_gender_index = np.argmax(gender_predicted_probs[0])
        top_gender_category = gender_category_mapping[top_gender_index]

        #predict age
        age_data = [[main_category, sub_category,top_county_category,top_gender_category]]
        age_data = age_column_transformer.transform(age_data)
        age_data[:, 16:] = age_scalar.transform(age_data[:, 16:])

        age_predicted_probs = age_model.predict_proba(age_data)
        top_age_index = np.argmax(age_predicted_probs[0])
        top_age_category = age_category_mapping[top_age_index]

        #predict time
        
        encoded_main_category = time_encoder["main_category"].transform([main_category])[0]
        encoded_sub_category = time_encoder["sub_category"].transform([sub_category])[0]

        date_range = pd.date_range(start="2023-09-01", end="2024-08-01", freq="M")
        user_input_data = pd.DataFrame({
            "main_category": [encoded_main_category] * len(date_range),
            "sub_category": [encoded_sub_category] * len(date_range),
            "Date_year": date_range.year,
            "Date_month": date_range.month
        })

        predicted_counts = time_model.predict(user_input_data)

        chart_data = {
            "labels": ["September-2023", "October-2023", "November-2023", "December-2023", "January-2024", "February-2024", "March-2024", "April-2024", "May-2024", "June-2024", "July-2024"],
            "data": predicted_counts.tolist()
        }

        chart_data_json = json.dumps(chart_data)


        return render_template('about.html', top3_country_categories=top3_county_categories, top_gender_category=top_gender_category,top_age_category=top_age_category, top_county_category=top_county_category, main_category=main_category, sub_category=sub_category, chart_data_json=chart_data_json)
    

    return render_template('about.html')

@app.route('/', methods=['GET', 'POST'])
def index():

    return render_template('index.html')


@app.route('/demand_predict', methods=['GET', 'POST'])
def demand_predict():
    prediction_results = []

    if request.method == 'POST':
        user_date_year = int(request.form['main_category'])
        user_date_month = int(request.form['sub_category'])

        for user_main_category in user_main_categories:
            # Encode user input main_category using the same encoder
            encoded_main_category = demand_encoder.transform([user_main_category])

            # Create a DataFrame with user input
            user_input_df = pd.DataFrame({
                'main_category': encoded_main_category,
                'Date_year': [user_date_year],
                'Date_month': [user_date_month]
            })

            # Make predictions using the trained regressor
            predicted_count = float(demand_model.predict(user_input_df)[0])  # Convert to float

            # Store prediction results in a dictionary
            prediction_results.append({
                'main_category': user_main_category,
                'predicted_count': predicted_count
            })
    
    ################################################################################################

    prediction_results_line = []

    user_date_year = 2023
    target_months = pd.date_range(start='2023-09-01', end='2024-08-01', freq='M').strftime('%Y-%m').tolist()

        # Dictionary to hold predicted counts for each main category over the months
    predicted_counts_over_time = {main_category: [] for main_category in user_main_categories}

    for user_main_category in user_main_categories:
            # Encode user input main_category using the same encoder
        encoded_main_category = demand_encoder.transform([user_main_category])

        for target_month in target_months:
            year, month = map(int, target_month.split('-'))

                # Create a DataFrame with user input
            user_input_df = pd.DataFrame({
                    'main_category': encoded_main_category,
                    'Date_year': [year],
                    'Date_month': [month]
            })

                # Make predictions using the trained regressor
            predicted_count = float(demand_model.predict(user_input_df)[0])  # Convert to float
            predicted_counts_over_time[user_main_category].append(predicted_count)

        # Prepare data for Chart.js
    chart_data = {
            'labels': target_months,
            'datasets': [
                {
                    'label': main_category,
                    'data': predicted_counts_over_time[main_category],
                    'borderColor': 'rgba(75, 192, 192, 1)',
                    'fill': False,
                    'lineTension': 0.4
                }
                for main_category in user_main_categories
            ]
        }

    return render_template('demand_predict.html', prediction_results=json.dumps(prediction_results), chart_data=chart_data)

@app.route('/service', methods=['GET', 'POST'])
def service():

    return render_template('service.html')

@app.route('/idea', methods=['GET', 'POST'])
def idea():

    return render_template('idea.html')


@app.route('/testidea', methods=['GET', 'POST'])
def testidea():
    if request.method == 'POST':
        passion_interests = request.form.getlist('passionInterests')
        skills_expertise = request.form.getlist('skillsExpertise')
        offering_type = request.form.get('offeringType')
        business_type = request.form.get('businessType')

        # Process the received data as needed
        # ...
        response_data = {
                    'passionInterests': passion_interests,
                    'skillsExpertise': skills_expertise,
                    'offeringType': offering_type,
                    'businessType': business_type
                }
        print(response_data)

        promptidea = f"I have passion and interests in {passion_interests} areas and I have skills and expertise in {skills_expertise}. I want to create 5 business ideas to start a new business for a {offering_type} and it sholud offer {business_type}. Give ideas in point form."

        print(promptidea)

    responseidea = openai.Completion.create(
        model="text-davinci-003",
        prompt=promptidea,
        temperature=1,
        max_tokens=256,
        top_p=1,
        frequency_penalty=0,
        presence_penalty=0
    )

    print(responseidea)

    #return render_template('idea.html',responseidea=responseidea)
    return responseidea

    
    
@app.route('/createplan', methods=['GET', 'POST'])
def createplan():
    if request.method == 'POST':
        
        product_name = request.form.get('sub_category')
        country = request.form.get('top3_country_category')
        gender = request.form.get('top_gender_category')
        age_group = request.form.get('top_age_category')

        print(product_name)

        prompt = f"Write a marketing plan specifically {product_name} sales for {country} for {gender}, age group of {age_group}. Generate the answer specifically for the given demographics. Give plan in paragraphs but break into these headlines. Executive Summary, Market Evaluation, Goals & Objectives, Strategic Initiatives, Timeline & Budget , Evaluation & Control. Each headline should contain only 1 paragraph. Always use this format: \n\n for headlines and \n for content for each headline."

# Print the formatted prompt
    print(prompt)

    response = openai.Completion.create(
        model="text-davinci-003",
        prompt=prompt,
        temperature=1,
        max_tokens=256,
        top_p=1,
        frequency_penalty=0,
        presence_penalty=0
    )

    print(response)

    response_text = response.choices[0].text
    sections = response_text.split("\n\n")

    # Create a dictionary to store sections
    marketing_plan_sections = {}
    current_section = None

    for section in sections:
        section_parts = section.split("\n", 1)
        if len(section_parts) == 2:
            current_section = section_parts[0].strip()
            marketing_plan_sections[current_section] = section_parts[1].strip().split("\n\n")

    return render_template('service.html', sections=marketing_plan_sections)

if __name__ == '__main__':
    app.run(host='0.0.0.0',port=3000)

    
