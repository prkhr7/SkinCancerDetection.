from flask import Flask, render_template, request
import nbformat
from nbconvert.preprocessors import ExecutePreprocessor

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return render_template('index.html', error='No file part')

    file = request.files['file']

    if file.filename == '':
        return render_template('index.html', error='No selected file')

    # Save the uploaded file
    file_path = 'uploads/' + file.filename
    file.save(file_path)

    # Load and execute the IPython Notebook
    with open('prediction_notebook.ipynb', 'r') as f:
        notebook_content = nbformat.read(f, as_version=4)

    ep = ExecutePreprocessor(timeout=600, kernel_name='python3')
    ep.preprocess(notebook_content, {'metadata': {'path': 'uploads/'}})

    return render_template('result.html', result='Prediction complete.')

if __name__ == '__main__':
    app.run(debug=True)
