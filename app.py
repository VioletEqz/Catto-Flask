from flask import Flask, render_template, request
from models import Catto,Face
import os
from utils import save_plot, save_plot_face

app = Flask(__name__)

app.config['UPLOAD_FOLDER'] = 'uploads'

Catto_Model = Catto()
Face_Model = Face()

Catto_only= False

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/infer', methods=['POST'])
def success():
    if request.method == 'POST':
        try:
            os.remove('./static/output.png')
        except:
            pass
        f = request.files['file']
        saveLocation = f.filename
        f.save(saveLocation)
        if Catto_only:
            preds = Catto_Model.infer(saveLocation)
            num = preds.shape[0]
            save_plot(preds, Catto_Model.original)
            # delete file after making an inference
            os.remove(saveLocation)
            # respond with the inference
            return render_template('inference.html', num=num)
        else:
            cat_preds = Catto_Model.infer(saveLocation)
            num = cat_preds.shape[0]
            face_preds = Face_Model.infer(saveLocation)
            save_plot_face(cat_preds,face_preds,Catto_Model.original)
            # delete file after making an inference
            os.remove(saveLocation)
            # respond with the inference
            return render_template('inference.html', num=num)
            


if __name__ == '__main__':
    app.debug = True
    port = int(os.environ.get("PORT", 80))
    app.run(host='0.0.0.0', port=port, debug=True)
