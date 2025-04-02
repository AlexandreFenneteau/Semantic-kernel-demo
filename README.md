# Semantic kernel
## Activate env

Avec un python 3.12
```console
python -m venv .venv
.\.venv\Scripts\activate
pip install -r requirements.txt
```

## Intro
`streamlit run 00-intro_multiagent.py`

## Basic use
### chat
`streamlit run 01-basic_utilisation_llm.py`
### text to image
`streamlit run 01-basic_utilisation_dalle.py`

## Plugins
`streamlit run 02-plugins.py`

## Processes
`streamlit run 03-processes.py`

## Multi-agent
`streamlit run 00-intro_multiagent.py`

# Smolagent

Avec un python 3.13
```console
python -m venv .venv-smolagent
.\..venv-smolagent\Scripts\activate
pip install -r requirements-smolagents.txt
```

## Launch telemetry server
`python -m phoenix.server.main serve`

## In the browser
`http://localhost:6006/projects/`

## Launch smolagent
`python .\04-smolagents.py`
