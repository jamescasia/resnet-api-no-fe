from torchvision.models import resnet50, ResNet50_Weights
from flask import Flask, request ,render_template, url_for
from werkzeug.utils import secure_filename 
import urllib.request,io 
from PIL import Image 
import os 

def create_app():

    app = Flask('image-classifier') 

    weights = ResNet50_Weights.DEFAULT
    model = resnet50(weights=weights)
    model.eval()
    preprocess = weights.transforms()

    @app.route('/classify', methods=['GET'])
    def classify_link(): 
        url = io.BytesIO(urllib.request.urlopen(request.get_json()['url']).read())
        img = Image.open(url)
        
        batch = preprocess(img).unsqueeze(0) 
        prediction = model(batch).squeeze(0).softmax(0)
        class_id = prediction.argmax().item()
        score = prediction[class_id].item()
        category_name = weights.meta["categories"][class_id] 
        return  { 'class': category_name, 'confidence': float("{:.2f}".format(score)) } 
  
    @app.route('/')  
    @app.route('/gui', methods=['GET'])
    def gui():
        return render_template('gui.html')

    @app.route('/cli', methods=['GET'])
    def cli():
        return render_template('cli.html')
    
    @app.route('/about', methods=['GET'])
    def about():
        return render_template('about.html'  ) 
    
    
    @app.route('/gui', methods=['POST']) 
    @app.route('/', methods=['POST']) 
    def classify():
        clear_dir('./static/images/resnet/') 
        if 'file' not in request.files:
            return render_template('gui.html', msg='No file found')

        file = request.files['file']
        if file.filename == '':
            return render_template('gui.html', msg='No file selected')

        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            path = f'static/images/resnet/{filename}'
            file.save(path)

            img = Image.open(path)    
            batch = preprocess(img).unsqueeze(0) 
    
            prediction = model(batch).squeeze(0).softmax(0)
            class_id = prediction.argmax().item()
            score = prediction[class_id].item()
            category_name = weights.meta["categories"][class_id]  

            return render_template(
                'gui.html', result = f"{category_name}: {100 * score:.1f}%", 
                img_url = f"{url_for('static', filename=f'/images/resnet/{filename}')}"
            ) 

    


    def clear_dir(dir_path):
        for file in os.listdir(dir_path):
            os.remove(f'{dir_path}/{file}') 
    
    def allowed_file(filename):
        return '.' in filename and filename.rsplit('.', 1)[1].lower() in {'png', 'jpg', 'jpeg'}
    
    return app