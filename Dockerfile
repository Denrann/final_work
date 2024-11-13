FROM python:3.12
RUN pip install --upgrade pip
WORKDIR /home/denrann/anaconda3/envs/chat_apart_bot/Project/
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY apart.py .
EXPOSE 9999
CMD ["python3", "apart.py"]
