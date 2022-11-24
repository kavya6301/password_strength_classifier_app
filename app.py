# Core Pkgs
import streamlit as st 
import altair as alt
import plotly.express as px 

# EDA Pkgs
import pandas as pd 
import numpy as np 
from datetime import datetime

# Utils
import joblib 
import pickle
pipe_lr = joblib.load(open("password_strength_classification.pkl","rb"))
#pipe_lr = pickle.load(open("password_strength_classification.pkl","rb"))

# Track Utils
from track_utils import create_page_visited_table,add_page_visited_details,view_all_page_visited_details,add_prediction_details,view_all_prediction_details,create_strengthclf_table

# Fxn
def predict_strength(docx):
	results = pipe_lr.predict([docx])
	return results[0]

def get_prediction_proba(docx):
	results = pipe_lr.predict_proba([docx])
	return results

emoji_dict = {"strong":"🤗", "fair":"😐", "weak":"😔"}


# Main Application
def main():
	st.title("Password Strength App")
	menu = ["Home","Monitor","About"]
	choice = st.sidebar.selectbox("Menu",menu)
	create_page_visited_table()
	create_strengthclf_table()
	if choice == "Home":
		add_page_visited_details("Home",datetime.now())
		st.subheader("Home-Password Strength")

		with st.form(key='password_form'):
			raw_text = st.text_area("Type Here")
			submit_text = st.form_submit_button(label='Submit')

		if submit_text:
			col1,col2  = st.beta_columns(2)

			# Apply Fxn Here
			prediction = predict_strength(raw_text)
			probability = get_prediction_proba(raw_text)
			
			add_prediction_details(raw_text,prediction,np.max(probability),datetime.now())

			with col1:
				st.success("Original Text")
				st.write(raw_text)

				st.success("Prediction")
				emoji_icon = emoji_dict[prediction]
				st.write("{}:{}".format(prediction,emoji_icon))
				st.write("Confidence:{}".format(np.max(probability)))



			with col2:
				st.success("Prediction Probability")
				# st.write(probability)
				proba_df = pd.DataFrame(probability,columns=pipe_lr.classes_)
				# st.write(proba_df.T)
				proba_df_clean = proba_df.T.reset_index()
				proba_df_clean.columns = ["Strength","probability"]

				fig = alt.Chart(proba_df_clean).mark_bar().encode(x='Strength',y='probability',color='Strength')
				st.altair_chart(fig,use_container_width=True)



	elif choice == "Monitor":
		add_page_visited_details("Monitor",datetime.now())
		st.subheader("Monitor App")

		with st.beta_expander("Page Metrics"):
			page_visited_details = pd.DataFrame(view_all_page_visited_details(),columns=['Pagename','Time_of_Visit'])
			st.dataframe(page_visited_details)	

			pg_count = page_visited_details['Pagename'].value_counts().rename_axis('Pagename').reset_index(name='Counts')
			c = alt.Chart(pg_count).mark_bar().encode(x='Pagename',y='Counts',color='Pagename')
			st.altair_chart(c,use_container_width=True)	

			p = px.pie(pg_count,values='Counts',names='Pagename')
			st.plotly_chart(p,use_container_width=True)

		with st.beta_expander('Password Strength Classifier Metrics'):
			df_strength = pd.DataFrame(view_all_prediction_details(),columns=['Rawtext','Prediction','Probability','Time_of_Visit'])
			st.dataframe(df_strength)

			prediction_count = df_strength['Prediction'].value_counts().rename_axis('Prediction').reset_index(name='Counts')
			pc = alt.Chart(prediction_count).mark_bar().encode(x='Prediction',y='Counts',color='Prediction')
			st.altair_chart(pc,use_container_width=True)	



	else:
		st.subheader("About")
		add_page_visited_details("About",datetime.now())





if __name__ == '__main__':
	main()