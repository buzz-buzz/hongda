from api.beautify import recipe_cartoonize, recipe_paster, paster_effect_nose
import cv2

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    raise IOError("Cannot open webcam")

def test_image(fpath):
    img = cv2.imread(fpath)
    output = paster_effect_nose(img)
    cv2.imshow('Test', output)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == '__main__':
    # test_image(r'C:\Users\Jeff\Downloads\xinwen5.jpg')
    # test_image(r'C:\Users\Jeff\Downloads\xinwen4.jpg')
    # test_image(r'C:\Users\Jeff\Downloads\xinwen3.jpg')
    # test_image(r'C:\Users\Jeff\Downloads\xinwen2.jpg')
    # test_image(r'C:\Users\Jeff\Downloads\xinwen.jpg')

    while True:
        ret, frame = cap.read()
        frame = cv2.resize(frame, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA)
        output = paster_effect_nose(frame)
        cv2.imshow('Input', output)

        c = cv2.waitKey(1)
        if c == 27:
            break

    cap.release()
    cv2.destroyAllWindows()