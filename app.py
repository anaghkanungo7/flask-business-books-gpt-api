from flask import Flask, request, json
from flask_cors import CORS, cross_origin


from answer import getAnswer
app = Flask(__name__)

cors = CORS(app)
app.config['CORS_HEADERS'] = 'Content-Type'

@app.route("/")
@cross_origin()
def hello_world():
    return "Hello, cross-origin-world!"

@app.route("/answer", methods=['POST'])
@cross_origin()
def answer():
    data = json.loads(request.data)
    # sample json request body - this is exactly how it needs to be sent
    # {
    #     "api_key" : "myAPIKey",
    #     "prompt" : "this is a sample prompt"
    #     "temperature" : 0.7
    # }
    api_key = data['api_key']
    temperature = data['temperature']
    prompt = data['prompt']
    
    # Do processing
    result = getAnswer(api_key, prompt, temperature)
    
    # Return result
    return f'{result}'
    

