import streamlit as st

st.set_page_config(
    page_title="Extreme Weather Forecasting and Its Impacts in Togo Using Machine Learning: WeatherAI",
    page_icon="👋",
)

st.title("Home Page")
st.image("app/WeatherAI-Extreme.jpg", caption='', use_column_width=True)

st.header("The Problem")
st.write("""We want to help address damages caused by (extreme) rain and dryness.
Climate change is making the rains and drynesses more extreme in Togo. Many things are easily destroyed/disrupted during the rainy period. Accessing clean water during dryness periods is a challenge in some parts of the country.
Some problems:
– Homes are destroyed
– Life habits are perturbed: being late at rendez-vous/works/school, inability to open stores
– Transit becomes difficult, as it is usual to have water up to the adult waist, making the roads impracticable. 
The government and the officials, as well as NGOs, are doing many things to address the situation, but the results we are seeing during extreme weather periods show that citizens are still victims of the situation.""")

st.header("Want to know more?")
st.markdown("* [Omdena Page](https://omdena.com/chapter-challenges/extreme-weather-forecasting-in-togo-using-machine-learning-weatherai/)")

st.sidebar.success("Select a page above.")
