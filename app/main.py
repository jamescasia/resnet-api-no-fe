from torchvision.models import resnet50, ResNet50_Weights
from flask import Flask, request  
import urllib.request,io 
from PIL import Image  

def create_app():

    app = Flask('image-classifier') 

    weights = ResNet50_Weights.DEFAULT
    model = resnet50(weights=weights)
    model.eval()
    preprocess = weights.transforms()

    @app.route('/classify', methods=['GET'])
    def classify(): 
        url = io.BytesIO(urllib.request.urlopen(request.get_json()['url']).read())
        img = Image.open(url)
        
        batch = preprocess(img).unsqueeze(0) 
        prediction = model(batch).squeeze(0).softmax(0)
        class_id = prediction.argmax().item()
        score = prediction[class_id].item()
        category_name = weights.meta["categories"][class_id] 
        return  { 'class': category_name, 'confidence': float("{:.2f}".format(score)) } 
 
    return app