import cv2
import easyocr

class NumberExtractor:
    def __init__(self, lang_list=['en']):
        self.reader = easyocr.Reader(lang_list, gpu=True)

    def extract(self, crop_img):
        gray = cv2.cvtColor(crop_img, cv2.COLOR_BGR2GRAY)
        results = self.reader.readtext(gray)
        # filter numeric results
        nums = [text for _, text, _ in results if text.isdigit()]
        return nums if nums else None