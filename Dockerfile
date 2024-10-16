FROM python:3.10.15

WORKDIR /

COPY ./Fetch_App /Fetch_App

RUN pip install -r ./Fetch_App/requirements.txt

WORKDIR /Fetch_App

CMD ["streamlit", "run", "app.py", "--server.port", "6969"]
