FROM python:3.10.15

COPY ./Fetch_App .

RUN pip install -r ./Fetch_App/requirements.txt

RUN streamlit run ./Fetch_App/app.py --server.port 6969