import streamlit as st
import pandas as pd
import plotly.express as px

@st.cache_data
def load_data():
    df = pd.read_csv("pages/clothing.csv")
    return df

df = load_data()

st.title("Customer Review Analysis")

st.sidebar.title("Filters")
min_age = st.sidebar.slider("Minimum Age", int(df['Age'].min()), int(df['Age'].max()), int(df['Age'].min()))
max_age = st.sidebar.slider("Maximum Age", int(df['Age'].min()), int(df['Age'].max()), int(df['Age'].max()))
min_rating = st.sidebar.slider("Minimum Rating", 1, 5, 1)
max_rating = st.sidebar.slider("Maximum Rating", 1, 5, 5)

division_options = df['Division Name'].unique()
selected_division = st.sidebar.selectbox("Select Division", division_options)

department_options = df[df['Division Name'] == selected_division]['Department Name'].unique()
selected_department = st.sidebar.selectbox("Select Department", department_options)

class_options = df[(df['Division Name'] == selected_division) &
                   (df['Department Name'] == selected_department)]['Class Name'].unique()
selected_class = st.sidebar.selectbox("Select Class", class_options)

filtered_df = df[(df['Age'] >= min_age) & (df['Age'] <= max_age) &
                 (df['Rating'] >= min_rating) & (df['Rating'] <= max_rating) &
                 (df['Division Name'] == selected_division) &
                 (df['Department Name'] == selected_department) &
                 (df['Class Name'] == selected_class)]

st.write("Filtered Data:")
st.write(filtered_df)

fig = px.scatter_3d(filtered_df, x='Age', y='Rating', z='Positive Feedback Count', color='Rating',
                    title='Relationship between Age, Rating, and Positive Feedback Count')
fig.update_layout(margin=dict(l=0, r=0, b=0, t=30))
st.plotly_chart(fig)








































