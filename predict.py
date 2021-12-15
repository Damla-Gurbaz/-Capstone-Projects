import streamlit as st
import pickle
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import plotly.figure_factory as ff
import matplotlib.pyplot as plt
# Visualization packages
import matplotlib.pyplot as plt
import seaborn as sns
plt.rcParams["axes.grid"] = False
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.graph_objects as go


st.sidebar.image('employess-2.jpg', use_column_width=True)

st.sidebar.title('Employee Information')

html_temp = """
<div style="background-color:LightCoral;padding:10px">
<h2 style="color:white;text-align:center;">Employee Churn Prediction</h2>
</div><br><br>"""

st.markdown(html_temp,unsafe_allow_html=True)


st.image('employess.jpg', use_column_width=True)

st.write("[Photo by Shridhar Gupta](https://unsplash.com/photos/dZxQn4VEv2M)")
#Photo by <a href="https://unsplash.com/photos/dZxQn4VEv2M">Shridhar Gupta</a> on <a href="https://unsplash.com/photos/dZxQn4VEv2M">Unsplash</a>


data = pd.read_csv('dumm-scal-df.csv')


html_temp1 = """
<div style="background-color:LightGray;padding:8px">
<h2 style="color:black;text-align:center;">Data Overview</h2>
</div><br><br>"""

st.markdown(html_temp1,unsafe_allow_html=True)




#Tablo1
st.subheader("🔍How does the promotion status affect employee churn?")

fig1 = px.histogram(data,
                x='promotion_last_5years',
                color="left",
                barmode='group',
                color_discrete_sequence=px.colors.qualitative.Set2)

st.plotly_chart(fig1)


#Tablo2
st.subheader("🔍How does years of experience affect employee churn?")

fig2 = px.histogram(data,
                x='time_spend_company',
                color="left",
                barmode='group',
                color_discrete_sequence=px.colors.qualitative.Set2)


st.plotly_chart(fig2)


#Tablo3
st.subheader("🔍How does workload affect employee churn?")
fig3 = px.histogram(data,
                x='average_montly_hours',
                color="left",
                barmode='group',
                color_discrete_sequence=px.colors.qualitative.Set2)


st.plotly_chart(fig3)


#Tablo4
st.subheader('🔍How does the salary level affect employee churn?')
fig4 = px.histogram(data,
                x='salary',
                color="left",
                barmode='group',
                color_discrete_sequence=px.colors.qualitative.Set2)


st.plotly_chart(fig4)



#Prediction

st.title('Prediction Time...')
selection = st.selectbox("Select Your Model", ["KNN", "Gradient Boosting", "Random Forest"])

if selection =="KNN":
	#st.write("You selected", selection, "model👇🏻")
	model = pickle.load(open('knn_grid_model', 'rb'))
elif selection =="Gradient Boosting":
	#st.write("You selected", selection, "model👇🏻")
	model = pickle.load(open('gb_grid', 'rb'))
else:
	#st.write("You selected", selection, "model👇🏻")
	model = pickle.load(open('rf_grid_model', 'rb'))


#Sidebar

satisfaction_level = st.sidebar.slider(label="Satisfaction Level", min_value=0.0, max_value=1.0, step=0.01)
last_evaluation = st.sidebar.slider(label="Last Evaluation", min_value=0.0, max_value=1.0, step=0.01)
number_project = st.sidebar.number_input(label="Number of Projects", min_value=1, max_value=200)
average_monthly_hours = st.sidebar.number_input("Average Monthly Hours", min_value=10, max_value=2000)
time_spend_company = st.sidebar.slider("Time Spend in Company", min_value=0, max_value=30, step=1)
work_accident = st.sidebar.radio("Work Accident", (1, 0))
promotion_last_5years = st.sidebar.radio("Promotion in Last 5 Years", (1, 0))
department = st.sidebar.selectbox("Department", ['sales', 'technical', 'support', 'IT', 'product_mng', 'marketing', 'RandD', 'accounting', 'hr', 'management'])
salary = st.sidebar.selectbox("Salary", ['0(low)', '1(medium)', '2(high)'])



coll_dict = {'satisfaction_level':satisfaction_level, 'last_evaluation':last_evaluation, 'number_project':number_project, 'average_montly_hours':average_monthly_hours,\
			'time_spend_company':time_spend_company, 'Work_accident':work_accident, 'promotion_last_5years':promotion_last_5years,\
			'Departments ': department, 'salary':salary}
columns = ['satisfaction_level', 'last_evaluation', 'number_project', 'average_montly_hours',\
		   'time_spend_company', 'Work_accident', 'promotion_last_5years','salary', 'Departments _RandD',\
		   'Departments _accounting', 'Departments _hr', 'Departments _management',\
		   'Departments _marketing', 'Departments _product_mng', 'Departments _sales',\
		   'Departments _support', 'Departments _technical']


df_coll = pd.DataFrame.from_dict([coll_dict])
X = pd.get_dummies(df_coll,drop_first=True).reindex(columns=columns, fill_value=0)


scalerfile = 'scaler.sav'
scaler = pickle.load(open(scalerfile, 'rb'))

X_transformed = scaler.transform(X)


prediction = model.predict(X_transformed)

st.subheader('📌Employee Information')
st.table(df_coll)
st.subheader('Click PREDICT if configuration is OK...')
if st.button('PREDICT'):
	if prediction[0]==0:
		st.success(f'Employee will STAY ')
	elif prediction[0]==1:
		st.error(f'Employee will CHURN ')



st.image('image.jpg', use_column_width=True)












