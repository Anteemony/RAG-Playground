# RAG Playground 🛝

[Demo](https://github.com/Anteemony/RAG-Playground/assets/103512255/0d944420-e3e8-43cb-aad3-0a459d8d0318)

<video width="640" height="480" autoplay>
  <source src="../../../../_static/<RAG_playground>.mp4" type="video/mp4>
Your browser does not support the video tag.
</video>

RAG Playground is an application that allows you to interact with your PDF files using the Language Model of your choice.

## Introduction
Streamlit application that enables users to upload a pdf file and chat with an LLM for performing document analysis in a playground environment.
Compare the performance of LLMs across endpoint providers to find the best possible configuration for your speed, latency and cost requirements using the dynamic routing feature.
Play intuitively tuning the model hyperparameters as temperature, chunk size, chunk overlap or try the model with/without conversational capabilities.

You find more model/provider information in the [Unify benchmark interface](https://unify.ai/hub).

## Usage

1. Visit the application: [RAG Playground](https://unify-rag-playground.streamlit.app/)
2. Input your Unify APhttps://github.com/Anteemony/RAG-Playground/assets/103512255/0d944420-e3e8-43cb-aad3-0a459d8d0318I Key. If you don’t have one yet, log in to the [Unify Console](https://console.unify.ai/) to get yours.
3. Select the Model and endpoint provider of your choice from the drop-down menu. You can find both model and provider information in the benchmark interface.
4. Upload your document(s) and click the Submit button.
5. Enjoy the application!

## Repository and Local Deployment

The repository is located at [RAG Playground Repository](https://github.com/Anteemony/RAG-Playground).

To run the application locally, follow these steps:

1. Clone the repository to your local machine.
2. Set up your virtual environment and install the dependencies from `requirements.txt`:

```bash
python -m venv .venv
source .venv/bin/activate  # On Windows use `.venv\Scripts\activate`
pip install -r requirements.txt
```

3. Run rag_script.py from Streamlit module 

```bash
python -m streamlit run rag_script.py
```

## Contributors

| Name | GitHub Profile |
|------|----------------|
| Anthony Okonneh | [AO](https://github.com/Anteemony) |
| Oscar Arroyo Vega | [OscarAV](https://github.com/OscarArroyoVega) |
| Martin Oywa | [Martin Oywa](https://github.com/martinoywa) |
