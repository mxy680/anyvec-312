from fastapi import FastAPI, HTTPException, Body, Depends, status, Request, Response
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from Vectorizer import Model

import os
import json
import requests
from io import BytesIO
import urllib.parse

app = FastAPI()

origins = [
  "http://localhost",
  "http://localhost:3000",
]

app.add_middleware(
  CORSMiddleware,
  allow_origins=origins,
  allow_credentials=True,
  allow_methods=["*"],
  allow_headers=["*"],
)

@app.on_event("startup")
async def startup():
  #instantiate embedding model
  app.state.pdf_vectorizer = Model()
  print("Embedding model initialised")
  
@app.get("/")
def read_root():
  return {}

@app.post("/vectorize")
async def vectorize(request: Request):
  #expects json object input, returns json output
  #input structure: ("", List<Bytes>)
  
  #process file url --> extract text + get List<Bytes>
  payload = requests.json()
  file_url: str = payload.get("file_url")
  
  #check if file is local or url
  if file_url.startswith("file"):
      prased_url = urllib.parse.urlparse(file_ur)
      file_path = prased_url.path
      with open(file_path, "rb") as file:
        file_data: bytes = file.read()
  else:
      response = requests.get(file_url)
      response.raise_for_status() #ensure we get a valid response back
      stream = BytesIO(response.content)
      file_data: bytes = stream.getvalue()
  #given file data is of type bytes either way, convert to text and image bytes
  
  data = (file_data, [])
  return app.state.pdf_vectorizer.process_elements(data)