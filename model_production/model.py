# Import dependencies
import pandas as pd
import numpy as np
import os
import pycats 
from pycats import cat_lump
import joblib
from sklearn.ensemble import GradientBoostingClassifier

def train_model():

	current_path = os.getcwd() #obtener la ruta donde se guarda este cuaderno
	dataset_path =current_path+'/startups_data.csv' #obtener la ruta donde se guarda el conjunto de datos de startups
	
	df=pd.read_csv(dataset_path, na_values=['none','None'])
	
	
	df.rename(columns={'Dependent-Company Status': 'status',
		'Specialization of highest education':'specialization_highest_education',
		'Focus on private or public data?' :'focus_private_or_public_data',
		'Focus on structured or unstructured data' :'focus_structured_unstructured_data',
		'Barriers of entry for the competitors':'barriers_entry_competitors',
		'Industry trend in investing' :'industry_trend_investing',
		'Predictive Analytics business':'predictive_analytics_business',
		'Big Data Business':'big_data_business',
		'Local or global player':'local_or_global_player',
		'Top management similarity':'top_management_similarity',
		'Renowned in professional circle':'renowned_professional_circle',
		'Degree from a Tier 1 or Tier 2 university?':'degree_tier1_tier2_university',
		'Number of  Sales Support material':'number_sales_support_material',
		'B2C or B2B venture?':'B2C_or_B2B',
		'Catering to product/service across verticals':'catering_product_service_across_verticals',
		"Top forums like 'Tech crunch' or 'Venture beat' talking about the company/model - How much is it being talked about?":'top_forums_talking_about_company',
		'Number of Co-founders':'number_cofounders',
		'Gartner hype cycle stage':'gartner_hype_cycle_stage',
		'Is the company an aggregator/market place? e.g. Bluekai':'aggregator_or_market_place',
		'Employee benefits and salary structures':'employee_benefits_salary_structures',
		'Consulting experience?':'consulting_experience',
		'Relevance of experience to venture':'relevance_experience_venture',
		'Relevance of education to venture':'relevance_education_venture',
		'Team size Senior leadership':'team_size_senior_lead',
		'Focus functions of company': 'focus_functions'},inplace=True) 
	
	
	
	features_final_model=['relevance_experience_venture',
	 'relevance_education_venture',
	 'focus_structured_unstructured_data',
	 'local_or_global_player',
	 'big_data_business',
	 'top_management_similarity',
	 'renowned_professional_circle',
	 'degree_tier1_tier2_university',
	 'team_size_senior_lead',
	 'number_sales_support_material',
	 'B2C_or_B2B',
	 'catering_product_service_across_verticals',
	 'top_forums_talking_about_company',
	 'gartner_hype_cycle_stage',
	 'aggregator_or_market_place',
	 'focus_private_or_public_data',
	 'focus_functions',
	 'employee_benefits_salary_structures',
	 'specialization_highest_education',
	 'consulting_experience',
	 'number_cofounders',
	 'predictive_analytics_business',
	 'industry_trend_investing',
	 'barriers_entry_competitors']
	
	var_dep=['status']
	
	
	train_final=df[features_final_model+var_dep]
	
	
	
	train_final['renowned_professional_circle'] = train_final['renowned_professional_circle'].map(lambda item : np.nan if item == 'No Info' else item)
	
	#Transformar tipo de datos
	
	transform_to_float=['renowned_professional_circle', 'team_size_senior_lead', 'number_cofounders']
	
	for col in transform_to_float:
	
		train_final[col]=train_final[col].astype(float)
		
	
	cols_numeric=train_final.select_dtypes(include=["float64"]).columns
	
	for col in cols_numeric:
		
		train_final[col] = train_final[col].fillna(train_final[col].mean())
		
		
	cols_object=train_final.select_dtypes(include=["object"]).columns
	
	for col in cols_object:
	
		train_final[col] = train_final[col].fillna('No Info')
	
	
	train_final['focus_private_or_public_data'].replace(['no'],'No Info',inplace=True)
	train_final['focus_structured_unstructured_data'].replace(['no'],'No Info',inplace=True)
	train_final['local_or_global_player'].replace(['global','GLOBAL','GLObaL'],'Global',inplace=True)
	train_final['local_or_global_player'].replace(['local','LOCAL','local  '],'Local',inplace=True)
	
	
	vars_to_relevel=['focus_functions', 'specialization_highest_education']
	
	for var in vars_to_relevel:
	
		train_final[var] = cat_lump(train_final[var].astype('category'), 29) 
	
		
	for col in vars_to_relevel: #volvemos a convertirlas a tipo object (para la función cat_lump era necesario transformarlas a 'category')
	
		train_final[col]=train_final[col].astype(object)
		
		
		
	train_final['status'].replace(['Success'], 1, inplace = True)
	train_final['status'].replace(['Failed'], 0, inplace = True)
	train_final['status'] = train_final['status'].astype(float)
	
	
	
	# Seleccionamos sólo variables de tipo "object"
	df_only_cat = train_final.select_dtypes(include=[object])
	
	
	# Sustituimos categorías por la proporción de Success
	for var in df_only_cat.columns:
		vars()[var] = train_final.groupby(var)["status"].sum()/len(train_final)
		df_only_cat[var] = df_only_cat[var].map(vars()[var])
	
	# Concatenamos con el resto de variables
	train_final = pd.concat([train_final.drop(df_only_cat.columns, axis=1), df_only_cat], axis=1)
	
	model = GradientBoostingClassifier(learning_rate=0.1, max_features='log2', min_samples_split=2, n_estimators=500, subsample=0.8)  
	
	model_fit = model.fit(train_final[features_final_model], train_final['status'])
	
	
	# Save your model
	
	joblib.dump(model_fit, 'model.model')
	print("Model dumped!")


	# Save categorical variables
	
	joblib.dump(df_only_cat.columns.tolist(), 'categorical_vars.pkl')
	print("Model dumped!")
	
	#Save tables with categorical encoding
	
	for var in df_only_cat.columns:
	
		joblib.dump(vars()[var], 'temp_files/category_encoding/'+var+'.pkl')
		

