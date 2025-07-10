import streamlit as st
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
)

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
    st.sidebar.write("## Text Generation Model")
    models = ["gpt-4", "gpt-3.5-turbo", "gpt-3.5-turbo-16k"]
    selected_model = st.sidebar.selectbox(
        'Choose a model',
        options=models,
        index=0
    )


    # select temperature on a scale of 0.0 to 1.0
    # st.sidebar.write("## Text Generation Temperature")
    temperature = st.sidebar.slider(
        "Temperature",
        min_value=0.0,
        max_value=1.0,
        value=0.0)

    # set use_cache in sidebar
    use_cache = st.sidebar.checkbox("Use cache", value=True)

    # Handle dataset selection and upload
    st.sidebar.write("## Data Summarization")
    st.sidebar.write("### Choose a dataset")

    datasets = [
        {"label": "Select a dataset", "url": None},
        {"label": "Stocks", "url": os.path.join("data", "stocks.csv")},
        {"label": "Cervical Cancer Risk Factors", "url": os.path.join("data", "risk_factors_cervical_cancer.csv")},
    ]

    selected_dataset_label = st.sidebar.selectbox(
        'Choose a dataset',
        options=[dataset["label"] for dataset in datasets],
        index=0
    )

    upload_own_data = st.sidebar.checkbox("Upload your own data")

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
        selected_dataset = datasets[[dataset["label"]
                                     for dataset in datasets].index(selected_dataset_label)]["url"]

    if not selected_dataset:
        st.info("To continue, select a dataset from the sidebar on the left or upload your own.")

    st.sidebar.write("### Choose a summarization method")
    # summarization_methods = ["default", "llm", "columns"]
    summarization_methods = [
        {"label": "llm",
         "description":
             "Uses the LLM to generate annotate the default summary, adding details such as semantic types for columns and dataset description"},
        {"label": "default",
         "description": "Uses dataset column statistics and column names as the summary"},

        {"label": "columns", "description": "Uses the dataset column names as the summary"}]

    # selected_method = st.sidebar.selectbox("Choose a method", options=summarization_methods)
    selected_method_label = st.sidebar.selectbox(
        'Choose a method',
        options=[method["label"] for method in summarization_methods],
        index=0
    )

    selected_method = summarization_methods[[
        method["label"] for method in summarization_methods].index(selected_method_label)]["label"]

    # add description of selected method in very small font to sidebar
    selected_summary_method_description = summarization_methods[[
        method["label"] for method in summarization_methods].index(selected_method_label)]["description"]

    if selected_method:
        st.sidebar.markdown(
            f"<span> {selected_summary_method_description} </span>",
            unsafe_allow_html=True)

# Step 3 - Generate data summary
if openai_key and selected_dataset and selected_method:
    lida = Manager(text_gen=llm("openai", api_key=openai_key))
    textgen_config = TextGenerationConfig(
        n=1,
        temperature=temperature,
        model=selected_model,
        use_cache=use_cache)

    st.write("## Summary")
    # **** lida.summarize *****
    summary = lida.summarize(
        selected_dataset,
        summary_method=selected_method,
        textgen_config=textgen_config)

    if "dataset_description" in summary:
        st.write(summary["dataset_description"])

    if "fields" in summary:
        fields = summary["fields"]
        nfields = []
        for field in fields:
            flatted_fields = {}
            flatted_fields["column"] = field["column"]
            # flatted_fields["dtype"] = field["dtype"]
            for row in field["properties"].keys():
                if row != "samples":
                    flatted_fields[row] = field["properties"][row]
                else:
                    flatted_fields[row] = str(field["properties"][row])
            # flatted_fields = {**flatted_fields, **field["properties"]}
            nfields.append(flatted_fields)
        nfields_df = pd.DataFrame(nfields)
        st.write(nfields_df)
    else:
        st.write(str(summary))

    # Step 4 - Generate goals
    if summary:
        st.sidebar.write("### Goal Selection")

        num_goals = st.sidebar.slider(
            "Number of goals to generate",
            min_value=1,
            max_value=10,
            value=4)
        own_goal = st.sidebar.checkbox("Add Your Own Goal")

        # **** lida.goals *****
        goals = lida.goals(summary, n=num_goals, textgen_config=textgen_config,
                           persona="A clinical data scientist interested in deeply understanding cancer data to improve patient outcomes.")
        st.write(f"## Goals ({len(goals)})")

        default_goal = goals[0].question
        goal_questions = [goal.question for goal in goals]

        if own_goal:
            user_goal = st.sidebar.text_input("Describe Your Goal")

            if user_goal:
                new_goal = Goal(question=user_goal, visualization=user_goal, rationale="")
                goals.append(new_goal)
                goal_questions.append(new_goal.question)

        selected_goal = st.selectbox('Choose a generated goal', options=goal_questions, index=0)

        # st.markdown("### Selected Goal")
        selected_goal_index = goal_questions.index(selected_goal)
        st.write(goals[selected_goal_index])

        selected_goal_object = goals[selected_goal_index]

        # Step 5 - Generate visualizations
        if selected_goal_object:
            st.sidebar.write("## Visualization Library")
            visualization_libraries = ["seaborn", "matplotlib", "plotly"]

            selected_library = st.sidebar.selectbox(
                'Choose a visualization library',
                options=visualization_libraries,
                index=0
            )

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

            st.write("### Visualization Code")
            st.code(selected_viz.code)

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
                    {"type": "text", "text": f"Chart description: {selected_goal_object.visualization + selected_goal_object.rationale}"}
                ]}
            ]

            response = client.chat.completions.create(model="gpt-4o",messages=messages,temperature=0.3,max_tokens=300)

            st.write(response.choices[0].message.content)
