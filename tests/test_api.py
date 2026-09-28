import io,unittest
from unittest.mock import patch
from PIL import Image
from fastapi.testclient import TestClient
import app
from detector import read_image,infer
class ApiTests(unittest.TestCase):
    def setUp(self):self.client=TestClient(app.app)
    def test_invalid_input(self):
        self.assertEqual(self.client.post('/api/detect',content=b'not an image').status_code,400)
        self.assertEqual(self.client.post('/api/detect?confidence=2',content=b'x').status_code,422)
    def test_valid_image_and_missing_model(self):
        data=io.BytesIO();Image.new('RGB',(40,30)).save(data,format='PNG')
        self.assertEqual(read_image(data.getvalue()).size,(40,30))
        with patch('app.infer',side_effect=FileNotFoundError('Model missing')):
            self.assertEqual(self.client.post('/api/detect',content=data.getvalue()).status_code,503)
    def test_result_contract(self):
        data=io.BytesIO();Image.new('RGB',(40,30)).save(data,format='PNG')
        with patch('app.infer',return_value=([{'label':'fire','confidence':.8,'xyxy':[0,0,20,20]}],data.getvalue())):
            r=self.client.post('/api/detect',content=data.getvalue())
            self.assertEqual(r.status_code,200);self.assertEqual(r.json()['count'],1)
        self.assertEqual(self.client.get('/models/fire-smoke.pt').status_code,404)
    def test_threshold_validation(self):
        with self.assertRaises(ValueError):infer(Image.new('RGB',(10,10)),confidence=2)
if __name__=='__main__':unittest.main()
