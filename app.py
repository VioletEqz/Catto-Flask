from flask import Flask, render_template, request
from models import Catto
import os
from utils import save_plot

app = Flask(__name__)

app.config['UPLOAD_FOLDER'] = 'uploads'

model = Catto()


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
        preds = model.infer(saveLocation)
        num = save_plot(preds,model.original)
        # delete file after making an inference
        os.remove(saveLocation)
        # respond with the inference
        return render_template('inference.html', num=num)


if __name__ == '__main__':
    app.debug = True
    port = int(os.environ.get("PORT", 80))
    app.run(host='0.0.0.0', port=port, debug=True)
