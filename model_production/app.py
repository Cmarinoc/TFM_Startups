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

if not os.path.isfile('model.model'):
    train_model()
    
    
model = joblib.load('model.model')

class MakePrediction(Resource):

	@staticmethod
	def post():
	
		json_ = request.get_json() #datos de entrada: serían las nuevas observaciones en formato json (solo columnas requeridas, y en su formato adecuado)
		print(json_)
		
		
		posted_data = request.get_json()
		
		
		relevance_experience_venture=posted_data['relevance_experience_venture']
		relevance_education_venture=posted_data['relevance_education_venture']
		focus_structured_unstructured_data=posted_data['focus_structured_unstructured_data']
		local_or_global_player=posted_data['local_or_global_player']
		big_data_business=posted_data['big_data_business']
		top_management_similarity=posted_data['top_management_similarity']
		renowned_professional_circle=posted_data['renowned_professional_circle']
		degree_tier1_tier2_university=posted_data['degree_tier1_tier2_university']
		team_size_senior_lead=posted_data['team_size_senior_lead']
		number_sales_support_material=posted_data['number_sales_support_material']
		B2C_or_B2B=posted_data['B2C_or_B2B']
		catering_product_service_across_verticals= posted_data['catering_product_service_across_verticals']
		top_forums_talking_about_company=posted_data['top_forums_talking_about_company']
		gartner_hype_cycle_stage=posted_data['gartner_hype_cycle_stage']
		aggregator_or_market_place=posted_data['aggregator_or_market_place']
		focus_private_or_public_data=posted_data['focus_private_or_public_data']
		focus_functions=posted_data['focus_functions']
		employee_benefits_salary_structures=posted_data['employee_benefits_salary_structures']
		specialization_highest_education=posted_data['specialization_highest_education']
		consulting_experience=posted_data['consulting_experience']
		number_cofounders=posted_data['number_cofounders']
		predictive_analytics_business=posted_data['predictive_analytics_business']
		industry_trend_investing=posted_data['industry_trend_investing']
		barriers_entry_competitors=posted_data['barriers_entry_competitors']
		
		
		'''
		
		df=pd.DataFrame(json_) ???
		
		#Imputamos NA:
		cols_numeric=train_final.select_dtypes(include=["float64"]).columns
		
		for col in cols_numeric:
		
		train_final[col] = train_final[col].fillna(train_final[col].mean())
		
		
		cols_object=train_final.select_dtypes(include=["object"]).columns
		
		for col in cols_object:
		
		train_final[col] = train_final[col].fillna('No Info')
		
		
		#encoding categoricals
		
		
		# Seleccionamos sólo variables de tipo "object"
		df_only_cat = train_final.select_dtypes(include=[object])
		
		# Sustituimos categorías por la proporción de Success
		for var in df_only_cat.columns:
		prop = train_final.groupby(var)["status"].sum()/len(train_final)
		df_only_cat[var] = df_only_cat[var].map(prop)
		
		# Concatenamos con el resto de variables
		train_final = pd.concat([train_final.drop(df_only_cat.columns, axis=1), df_only_cat], axis=1)
		
				  
		'''
	
		
		
		prediction = model.predict_proba([[relevance_experience_venture,
		relevance_education_venture,
		focus_structured_unstructured_data,
		local_or_global_player,
		big_data_business,
		top_management_similarity,
		renowned_professional_circle,
		degree_tier1_tier2_university,
		team_size_senior_lead,
		number_sales_support_material,
		B2C_or_B2B,
		catering_product_service_across_verticals,
		top_forums_talking_about_company,
		gartner_hype_cycle_stage,
		aggregator_or_market_place,
		focus_private_or_public_data,
		focus_functions,
		employee_benefits_salary_structures,
		specialization_highest_education,
		consulting_experience,
		number_cofounders,
		predictive_analytics_business,
		industry_trend_investing,
		barriers_entry_competitors]])[:,1]
		
		
		
		return jsonify({'prediction': str(prediction)})




api.add_resource(MakePrediction, '/predict')


if __name__ == '__main__':
    app.run(debug=True)














