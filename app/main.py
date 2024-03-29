from torchvision.models import resnet50, ResNet50_Weights
from flask import Flask, request ,render_template, url_for
from werkzeug.utils import secure_filename 
import urllib.request,io 
from PIL import Image 
import os 
import sys

def create_app():

    app = Flask('image-classifier') 

    weights = ResNet50_Weights.DEFAULT
    model = resnet50(weights=weights)
    model.eval()
    preprocess = weights.transforms()

    @app.route('/classify', methods=['POST'])
    def classify(): 
        
        try:
            external = request.args.get('external')
            if external == 'true':
                url = io.BytesIO(urllib.request.urlopen(request.args.get('url')).read())
                img = Image.open(url)
                batch = preprocess(img).unsqueeze(0) 
                prediction = model(batch).squeeze(0).softmax(0)
                class_id = prediction.argmax().item()
                score = prediction[class_id].item()
                category_name = weights.meta["categories"][class_id] 
            else:
                url = request.args.get('url').replace('/', '', 2)
                img = Image.open(url)    
                batch = preprocess(img).unsqueeze(0) 
        
                prediction = model(batch).squeeze(0).softmax(0)
                class_id = prediction.argmax().item()
                score = prediction[class_id].item()
                category_name = weights.meta["categories"][class_id]  
                
        except Exception as error:
            exc_type, _, exc_tb = sys.exc_info()
            fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
            return {'code': 'Error', 'body': {'msg': str(error), 'type': str(exc_type), 'fname': str(fname), 'line': str(exc_tb.tb_lineno)}}, 400

            # return {'error': 'UndefinedError'}, 400
        return  { 'class': category_name, 'confidence': float("{:.2f}".format(score)) }, 200
    
    @app.route('/')   
    def gui():
        return render_template('gui.html')

    @app.route('/upload', methods=['POST'])  
    def upload():
        clear_dir('./static/images/') 
        if 'image' not in request.files:
            return { "error": "NoImageFound"}, 400

        file = request.files['image']
        if file.filename == '':
            return { "error": "NoImageFound"}, 400

        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            path = f'static/images/{filename}'
            file.save(path)
            return { "code": "Success", "img_url" : f"{url_for('static', filename=f'/images/{filename}')}"}, 200
        else:
            return { "error": "FileNotAllowed"}, 400
    
    @app.route('/', methods=['POST'])  
    def classify_gui():
        clear_dir('./static/images/') 
        if 'file' not in request.files:
            return render_template('gui.html', msg='No file found')

        file = request.files['file']
        if file.filename == '':
            return render_template('gui.html', msg='No file selected')

        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            path = f'static/images/{filename}'
            file.save(path)

            img = Image.open(path)    
            batch = preprocess(img).unsqueeze(0) 
    
            prediction = model(batch).squeeze(0).softmax(0)
            class_id = prediction.argmax().item()
            score = prediction[class_id].item()
            category_name = weights.meta["categories"][class_id]  

            return render_template(
                'gui.html', result = f"{category_name}: {100 * score:.1f}%", 
                img_url = f"{url_for('static', filename=f'/images/{filename}')}"
            ) 

    return app


def clear_dir(dir_path):
    for file in os.listdir(dir_path):
        os.remove(f'{dir_path}/{file}') 
    
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in { 'jpg', 'jpeg'}
    
