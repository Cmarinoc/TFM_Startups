# Dependencies

import os
from flask import Flask, jsonify, request
from flask_restful import Api, Resource
from model import train_model
import joblib
import pandas as pd
import numpy as np



# Your API definition
app = Flask(__name__)
api = Api(app)


train_model() #comentar esta linea si ya está entrenado y tiene todas las categoricas guardadas
    
    
model = joblib.load('model.model')

df_only_cat=joblib.load('categorical_vars.pkl')


for var in df_only_cat: #cargo las tablas con la proporción asociada a las categóricas

	vars()[var]=joblib.load('temp_files/category_encoding/'+var+'.pkl')



class MakePrediction(Resource):

	@staticmethod
	def post():
	
		json_ = request.get_json() #datos de entrada: serían las nuevas observaciones en formato json (solo columnas requeridas, y en su formato adecuado)
		print(json_)
		
		
		posted_data = request.get_json()
		
		
		industry_trend_investing=posted_data['industry_trend_investing']
		number_cofounders=posted_data['number_cofounders']
		renowned_professional_circle=posted_data['renowned_professional_circle']
		team_size_senior_lead=posted_data['team_size_senior_lead']
		
		#a las que son categóricas se les tiene que aplicar el encoding con las tablas importadas
		 
		relevance_experience=posted_data['relevance_experience_venture']
		relevance_experience=relevance_experience_venture[relevance_experience]
		
		relevance_education=posted_data['relevance_education_venture']
		relevance_education=relevance_education_venture[relevance_education] 
		
		focus_structured_unstructured=posted_data['focus_structured_unstructured_data']
		focus_structured_unstructured=focus_structured_unstructured_data[focus_structured_unstructured]
		
		local_or_global=posted_data['local_or_global_player']
		local_or_global=local_or_global_player[local_or_global] 
			
		big_data=posted_data['big_data_business']
		big_data=big_data_business[big_data] 
		
		top_management=posted_data['top_management_similarity']
		top_management=top_management_similarity[top_management] 
		
		degree_tier1_tier2=posted_data['degree_tier1_tier2_university']
		degree_tier1_tier2=degree_tier1_tier2_university[degree_tier1_tier2] 
		
		number_sales_support=posted_data['number_sales_support_material']
		number_sales_support=number_sales_support_material[number_sales_support] 
	
		B2C_B2B=posted_data['B2C_or_B2B']
		B2C_B2B=B2C_or_B2B[B2C_B2B] 
		
		catering_product_service_verticals= posted_data['catering_product_service_across_verticals']
		catering_product_service_verticals= catering_product_service_across_verticals[catering_product_service_verticals]
				
		top_forums_talking_company=posted_data['top_forums_talking_about_company']
		top_forums_talking_company=top_forums_talking_about_company[top_forums_talking_company]
		
		gartner_hype_cycle=posted_data['gartner_hype_cycle_stage']
		gartner_hype_cycle=gartner_hype_cycle_stage[gartner_hype_cycle]
		
		aggregator_or_market=posted_data['aggregator_or_market_place']
		aggregator_or_market=aggregator_or_market_place[aggregator_or_market]
		
		focus_private_or_public=posted_data['focus_private_or_public_data']
		focus_private_or_public=focus_private_or_public_data[focus_private_or_public]
		
		focus_funct=posted_data['focus_functions']
		focus_funct=focus_functions[focus_funct]
			
		employee_benefits_salary=posted_data['employee_benefits_salary_structures']
		employee_benefits_salary=employee_benefits_salary_structures[employee_benefits_salary]
		
		specialization_highest=posted_data['specialization_highest_education']
		specialization_highest=specialization_highest_education[specialization_highest]
		
		consulting_exp=posted_data['consulting_experience']
		consulting_exp=consulting_experience[consulting_exp]
		
		predictive_analytics=posted_data['predictive_analytics_business']
		predictive_analytics=predictive_analytics_business[predictive_analytics]
		
		barriers_entry=posted_data['barriers_entry_competitors']
		barriers_entry=barriers_entry_competitors[barriers_entry]
		
		#falta imputar NA y reducir categóricas para la parte de predicciones (focus_functions y specialization_highest_education)
		
		
		prediction = model.predict_proba([[relevance_experience,
		relevance_education,
		focus_structured_unstructured,
		local_or_global,
		big_data,
		top_management,
		renowned_professional_circle,
		degree_tier1_tier2,
		team_size_senior_lead,
		number_sales_support,
		B2C_B2B,
		catering_product_service_verticals,
		top_forums_talking_company,
		gartner_hype_cycle,
		aggregator_or_market,
		focus_private_or_public,
		focus_funct,
		employee_benefits_salary,
		specialization_highest,
		consulting_exp,
		number_cofounders,
		predictive_analytics,
		industry_trend_investing,
		barriers_entry]])[:,1]
		
		
		
		return jsonify({'Success score': str(prediction)})




api.add_resource(MakePrediction, '/predict')


if __name__ == '__main__':
    app.run(debug=True)














