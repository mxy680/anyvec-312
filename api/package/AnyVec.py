import requests

class AnyVec:
  def __init__(self, docker_endpoint=None):
    if docker_endpoint:
      self.docker_endpoint = docker_endpoint
    else:
      self.docker_endpoint = "http://localhost:8080/"
      #run bash script to initialise and run docker
    
    
  async def vectorize(self, file_path: Optional[str], url: Optional[str]):
    vectors = ()
    #vectorizes a pdf
    if url:
      #pass a tuple of (text:str, List<Bytes>) to Model.vectorize
      #make request to api and expect JSON response containing vectors
      
      #validate endpoint url, and endpoint tests
      payload = {
        "file_url": url
      }
      response = await requests.post((self.docker_endpoint + "/vectorize"), json=payload)
      if response.status_code==200:
        print("vectors generated successfully!")
      vectors = response.json()
    return vectors
  
  
  #my_package = Any