VIDEO_PATH = "C:/Users/alex1/Desktop/Ahmad_Stuff/Drone_Disaster/Input_Video/nyc_people_walking1.mp4"

from multiprocessing import Process, Queue, Value, Manager
from tracking import tracking
from face_processing import face_processing
from emotion_detection import emotion_detection

if __name__ == "__main__":
    queue = Queue()
    permanent_id_counter = Value('i', 1)
    temp_id_counter = Value('i', 1)

    with Manager() as manager:
        temporary_ids = manager.dict()
        emotion_results = manager.dict()

        tracking_process = Process(target=tracking, args=(VIDEO_PATH, queue, temp_id_counter, permanent_id_counter, temporary_ids, emotion_results))
        face_processing_process = Process(target=face_processing, args=(queue, permanent_id_counter, temporary_ids))
        emotion_detection_process = Process(target=emotion_detection, args=(queue, emotion_results))

        tracking_process.start()
        face_processing_process.start()
        emotion_detection_process.start()

        tracking_process.join()
        face_processing_process.join()
        emotion_detection_process.join()
