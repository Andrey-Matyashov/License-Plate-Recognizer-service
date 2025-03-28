import requests
import io
from PIL import Image


class PlateReaderClient:
    def __init__(self, base_url="http://localhost:8080"):
        self.base_url = base_url.rstrip('/')
        
    def read_plate_from_image(self, image_path):
        """
        Отправка изображения на распознавание (/plate_reader)
        
        :param image_path: путь к файлу изображения
        :return: распознанный текст
        """
        with open(image_path, 'rb') as f:
            image_data = f.read()
        
        response = requests.post(
            f"{self.base_url}/plate_reader",
            data=image_data,
            headers={'Content-Type': 'application/octet-stream'}
        )
        return self._handle_response(response)
    
    def read_plate_by_id(self, image_id):
        """
        Распознавание номера по ID изображения (/predict_using_image_id)
        
        :param image_id: числовой ID изображения
        :return: распознанный текст
        """
        response = requests.post(
            f"{self.base_url}/predict_using_image_id",
            params={'image_id': image_id}
        )
        return self._handle_response(response)
    
    def read_plates_by_ids(self, image_ids):
        """
        Распознавание номеров для списка ID (/predict_using_image_ids)
        
        :param image_ids: список числовых ID изображений
        :return: список результатов распознавания
        """
        response = requests.post(
            f"{self.base_url}/predict_using_image_ids",
            json={'image_ids': image_ids}
        )
        return self._handle_response(response)
    
    def _handle_response(self, response):
        """Обработка ответа сервера"""
        try:
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            return {"error": str(e)}
        

if __name__ == "__main__":
    client = PlateReaderClient()

    # 1. Тест /plate_reader - отправка файла изображения
    print("\nТест отправки изображения:")
    result = client.read_plate_from_image("/home/andmats/aaa/License-Plate-Recognizer-service/images/9965.jpg")
    print(result)
    
    # 2. Тест /predict_using_image_id - получение по ID
    print("\nТест запроса по одному ID:")
    result = client.read_plate_by_id(10022)
    print(result)
    
    # 3. Тест /predict_using_image_ids - получение по списку ID
    print("\nТест запроса по нескольким ID:")
    result = client.read_plates_by_ids([10022, 9965])
    print(result)