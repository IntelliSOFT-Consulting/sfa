import streamlit as st
import lida
from openai import OpenAI
from dotenv import load_dotenv
from lida import Manager, TextGenerationConfig, llm
from lida.datamodel import Goal
import os
import pandas as pd

# make data dir if it doesn't exist
os.makedirs("data", exist_ok=True)

st.set_page_config(
    page_title="Cancer Registry Visualization",
    page_icon="📊",
    layout="wide",
)

hide_default_format = """
       <style>
       #MainMenu {visibility: hidden; }
       footer {visibility: hidden;}
       </style>
       """
st.markdown(hide_default_format, unsafe_allow_html=True)

st.write("# Using LLMs to better understand structured cancer data 📊")

st.sidebar.write("## Setup")

# Step 1 - Get OpenAI API key
load_dotenv()  # Load environment variables from .env file
openai_key = os.getenv("OPENAI_API_KEY")
client = OpenAI(api_key=openai_key) if openai_key else None

if not openai_key:
    openai_key = st.sidebar.text_input("Enter OpenAI API key:")
    if openai_key:
        display_key = openai_key[:2] + "*" * (len(openai_key) - 5) + openai_key[-3:]
        st.sidebar.write(f"Current key: {display_key}")
    else:
        st.sidebar.write("Please enter OpenAI API key.")
else:
    display_key = openai_key[:2] + "*" * (len(openai_key) - 5) + openai_key[-3:]
    st.sidebar.write(f"OpenAI API key loaded from environment variable: {display_key}")

# Step 2 - Select a dataset and summarization method
if openai_key:
    # Initialize selected_dataset to None
    selected_dataset = None
    # select model from gpt-4 , gpt-3.5-turbo, gpt-3.5-turbo-16k
    # st.sidebar.write("## Text Generation Model")
    # models = ["gpt-4", "gpt-3.5-turbo", "gpt-3.5-turbo-16k"]
    selected_model = "gpt-4"
    # select temperature on a scale of 0.0 to 1.0
    # st.sidebar.write("## Text Generation Temperature")
    temperature = 0.5
    # set use_cache in sidebar
    use_cache = True

    left, right = st.columns(2)
    with left:
        # Handle dataset selection and upload
        st.write("## Choose a dataset")

        datasets = [
            {"label": "Select a dataset", "url": None},
            {"label": "Stocks", "url": os.path.join("data", "stocks.csv")},
            {"label": "Cervical Cancer Risk Factors", "url": os.path.join("data", "risk_factors_cervical_cancer.csv")},
        ]

        selected_dataset_label = st.selectbox('Choose a dataset', options=[dataset["label"] for dataset in datasets],
                                              index=0)

        upload_own_data = st.checkbox("Upload your own data")

        if upload_own_data:
            uploaded_file = st.sidebar.file_uploader("Choose a CSV or JSON file", type=["csv", "json"])

            if uploaded_file is not None:
                # Get the original file name and extension
                file_name, file_extension = os.path.splitext(uploaded_file.name)

                # Load the data depending on the file type
                if file_extension.lower() == ".csv":
                    data = pd.read_csv(uploaded_file)
                elif file_extension.lower() == ".json":
                    data = pd.read_json(uploaded_file)

                # Save the data using the original file name in the data dir
                uploaded_file_path = os.path.join("data", uploaded_file.name)
                data.to_csv(uploaded_file_path, index=False)

                selected_dataset = uploaded_file_path
            datasets.append({"label": file_name, "url": uploaded_file_path})

            # st.sidebar.write("Uploaded file path: ", uploaded_file_path)
        else:
            selected_dataset = datasets[[dataset["label"] for dataset in datasets].index(selected_dataset_label)]["url"]

        if not selected_dataset:
            st.info("To continue, select one of our pre-loaded datasets or upload your own.")

    with right:
        selected_method = "llm"  # Default method
        # Step 3 - Generate data summary
        lida = Manager(text_gen=llm("openai", api_key=openai_key))
        textgen_config = TextGenerationConfig(n=1, temperature=temperature, model=selected_model,
                                              use_cache=use_cache)
        st.write("## Dataset description")
        # **** lida.summarize *****

        summary = None
        df_fields = pd.DataFrame()
        if openai_key and selected_dataset and selected_method:
            summary = lida.summarize(selected_dataset, summary_method=selected_method, textgen_config=textgen_config)
            if "dataset_description" in summary:
                st.write(summary["dataset_description"])
            if "fields" in summary:
                df_fields = pd.DataFrame(
                    [{"column": f["column"], **{k: str(v) for k, v in f["properties"].items() if k != "samples"}}
                        for f in summary["fields"]])

    if df_fields is not None and not df_fields.empty:
        st.dataframe(df_fields)

    # Step 4 - Generate goals
    if summary is not None:

        # **** lida.goals *****

        st.write("### Analysis Areas")

        num_goals = st.slider(
            "How many analysis areas would you like to generate?You can also add your own analysis objective.",
            min_value=1,
            max_value=10,
            value=4)
        own_goal = st.checkbox("Add Your Own Goal")

        goals = lida.goals(summary, n=num_goals, textgen_config=textgen_config,
                           persona="A clinical data scientist interested in deeply understanding cancer data to improve patient outcomes. I want all things to come back to cervical cancer.")

        default_goal = goals[0].question
        goal_questions = [goal.question for goal in goals]

        if own_goal:
            user_goal = st.text_input("Describe Your Goal")

            if user_goal:
                new_goal = Goal(question=user_goal, visualization=user_goal, rationale="")
                goals.append(new_goal)
                goal_questions.append(new_goal.question)

        selected_goal = st.selectbox('Choose an analysis to conduct', options=goal_questions, index=0)
        selected_goal_index = goal_questions.index(selected_goal)
        selected_goal_object = goals[selected_goal_index]

        st.markdown(f"**Goal Selected**: {selected_goal_object.question}")
        st.markdown(f"**Rationale**: {selected_goal_object.rationale}")

        # Step 5 - Generate visualizations
        if selected_goal_object:
            # st.sidebar.write("## Visualization Library")
            visualization_libraries = ["seaborn", "matplotlib", "plotly"]

            selected_library = "seaborn"

            # Update the visualization generation call to use the selected library.
            st.write("## Visualizations")

            # **** lida.visualize *****
            visualizations = lida.visualize(
                summary=summary,
                goal=selected_goal_object,
                textgen_config=textgen_config,
                library=selected_library)

            selected_viz = visualizations[0]

            if selected_viz.raster:
                from PIL import Image
                import io
                import base64

                imgdata = base64.b64decode(selected_viz.raster)
                img = Image.open(io.BytesIO(imgdata))
                img.save(fp="data/visualization.png", format='PNG')
                st.image(img, caption=selected_goal_object.visualization + ". " + selected_goal_object.rationale,
                         use_column_width=True)

            # st.write("### Visualization Code")
            # st.code(selected_viz.code)

            # *** lida.explain ***

            st.write("## Explanation")
            explanation = lida.explain(
                code=selected_viz.code,
                textgen_config=textgen_config,
                library=selected_library
            )

            for section_group in explanation:
                for item in section_group:
                    with st.expander(f"Category: {item['section']}"):
                        st.markdown("**Explanation:**")
                        st.markdown(item['explanation'])

            st.write("### Conclusion")
            with open("data/visualization.png", "rb") as img_file:
                image_bytes = img_file.read()

            messages = [
                {"role": "system", "content": "You are a data visualization expert. Review the image and description "
                                              "provided, and write a concise summary of what the visualization shows. "
                                              "Focus on trends, patterns, and insights that can be inferred."},
                {"role": "user", "content": [
                    {"type": "image_url",
                     "image_url": {"url": f"data:image/png;base64,{base64.b64encode(image_bytes).decode()}"}},
                    {"type": "text",
                     "text": f"Chart description: {selected_goal_object.visualization + selected_goal_object.rationale}"}
                ]}
            ]

            response = client.chat.completions.create(model="gpt-4o", messages=messages, temperature=0.3,
                                                      max_tokens=300)

            st.write(response.choices[0].message.content)
