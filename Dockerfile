FROM python:3.6

RUN mkdir -p /app/gene/

COPY requirements.txt /app/gene/

WORKDIR /app/gene/

RUN pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple

COPY . /app/gene/

EXPOSE 8000

CMD ["python", "/app/gene/manage.py", "runserver", "0.0.0.0:8000" ]
