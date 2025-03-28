from flask import Flask, request
import logging
import io
import requests
from PIL import Image

from models.plate_reader import PlateReader


app = Flask(__name__)
plate_reader_model = PlateReader.load_from_file('model_weights/plate_reader_model.pth')

# @app.route('/')
# def hello():
#     return '<h1><center>Hello!</center></h1>'

@app.route("/plate_reader", methods=['POST'])
def plate_reader():
    """
    Endpoint to recognize the text from a license plate image.

    This function receives a POST request with an image file in the request body.
    It uses the PlateReader model to process the image and extract the text from
    the license plate.

    Returns:
        dict: A dictionary containing the recognized text under the key "result".
    """

    data = request.get_data()
    im = io.BytesIO(data)
    text = plate_reader_model.read_text(im)
    return {"result": text}

@app.route("/predict_using_image_id", methods=['POST'])
def predict_using_image_id():
    """
    Endpoint to recognize the text from a license plate image by its ID.

    This function receives a POST request with the 'image_id' parameter in the
    request body. It uses the PlateReader model to process the image and extract
    the text from the license plate.

    Returns:
        dict: A dictionary containing the recognized text under the key "result".
    """
    image_id = request.args.get('image_id', type=int)
    response = requests.get(f" http://89.169.157.72:8080/images/{image_id}")
    try:
        response.raise_for_status() 
    except requests.exceptions.HTTPError as err:
        return {"error": str(err)}
    image = io.BytesIO(response.content)
    text = plate_reader_model.read_text(image)
    return {"result": text}


@app.route("/predict_using_image_ids", methods=['POST'])
def predict_using_image_ids():
    """
    Endpoint to recognize the text from a list of license plate images by their IDs.

    This function receives a POST request with the 'image_ids' parameter in the
    request body. It uses the PlateReader model to process the images and extract
    the text from each license plate.

    Returns:
        dict: A dictionary containing the recognized text under the key "result" for each image.
    """
    data = request.get_json()
    
    if not data or 'image_ids' not in data:
        return {"error": "No image_ids provided"}
    
    image_ids = data['image_ids']
    if not isinstance(image_ids, list):
        return {"error": "image_ids must be a list"}
    
    results = []
    for image_id in image_ids:
        response = requests.get(f" http://89.169.157.72:8080/images/{image_id}")
        try:
            response.raise_for_status() 
        except requests.exceptions.HTTPError as err:
            return {"error": str(err)}
        image = io.BytesIO(response.content)
        text = plate_reader_model.read_text(image)
        results.append(text)
    
    return {"results": results}


if __name__ == '__main__':
    logging.basicConfig(
        format='[%(levelname)s] [%(asctime)s] %(message)s',
        level=logging.INFO,
    )

    app.run(host='0.0.0.0', port=8080, debug=True)
