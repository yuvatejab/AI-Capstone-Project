import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import cosine_similarity

st.set_page_config(layout="wide", page_title="Tourism Insights & Recommender")

@st.cache_data
def load_data():
    user = pd.read_csv("Part 2/user.csv")
    ratings = pd.read_csv("Part 2/tourism_rating.csv")
    places = pd.read_excel("Part 2/tourism_with_id.xlsx")

    user['User_Id'] = user['User_Id'].astype(int)
    ratings['User_Id'] = ratings['User_Id'].astype(int)
    ratings = ratings[ratings['Place_Id'].apply(lambda x: str(x).isdigit())]
    ratings['Place_Id'] = ratings['Place_Id'].astype(int)
    places['Place_Id'] = places['Place_Id'].astype(int)
    
    # Create User_City column from Location to avoid conflicts
    user['User_City'] = user['Location'].str.split(',').str[0].str.strip()

    return user, ratings, places

user, ratings, places = load_data()

st.title("🏞 Tourism Explorer & Recommender")

tab1, tab2, tab3, tab4 = st.tabs(["Users", "Places", "Top Picks", "Recommendations"])

with tab1:
    st.header("Who's Visiting?")
    fig, ax = plt.subplots()
    sns.histplot(data=user, x='Age', bins=15, kde=True, ax=ax)
    st.pyplot(fig)

    top = user['User_City'].value_counts().head(10)
    st.bar_chart(top)

with tab2:
    st.header("Where Are They Going?")
    st.write("Categories available:")
    st.write(places['Category'].unique())

    trend = places.groupby('City')['Category'].agg(lambda x: x.mode()[0])
    st.write("Each city's standout category:")
    st.dataframe(trend.reset_index())

    nature_spots = places[places['Category'].str.lower().str.contains("alam")]
    st.write("Best cities for nature lovers:")
    st.write(pd.Series(nature_spots['City']).unique().tolist())

with tab3:
    st.header("Loved by Tourists")
    data = ratings.merge(user, on='User_Id').merge(places, on='Place_Id')

    top_spots = data.groupby('Place_Name')['Place_Ratings'].mean().sort_values(ascending=False).head(10)  # type: ignore
    st.write("Most appreciated spots:")
    st.dataframe(top_spots)

    fav_cities = data.groupby('User_City')['Place_Ratings'].mean().sort_values(ascending=False).head(5)  # type: ignore
    st.write("Cities that never disappoint:")
    st.dataframe(fav_cities)

    fav_types = data.groupby('Category')['Place_Ratings'].mean().sort_values(ascending=False)  # type: ignore
    st.write("Types of places people enjoy the most:")
    st.dataframe(fav_types)

with tab4:
    st.header("Let's Recommend You a Place")
    data = ratings.merge(user, on='User_Id').merge(places, on='Place_Id')
    pivot = data.pivot_table(index='User_Id', columns='Place_Name', values='Place_Ratings').fillna(0)
    sim_matrix = cosine_similarity(pivot.T)
    sim_frame = pd.DataFrame(sim_matrix, index=pivot.columns, columns=pivot.columns)

    place = st.selectbox("Pick a place you've enjoyed:", sorted(sim_frame.columns))
    if place:
        st.subheader(f"If you liked **{place}**, you might also enjoy:")
        results = sim_frame[place].sort_values(ascending=False)[1:6]  # type: ignore
        for name, score in results.items():  # type: ignore
            st.markdown(f"• **{name}** (match score: {score:.2f})")
