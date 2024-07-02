VIDEO_PATH = "C:/Users/alex1/Desktop/Ahmad_Stuff/Drone_Disaster/Input_Video/Jorge_and_Ahmad.MOV"

from multiprocessing import Process, Queue, Value, Manager
from tracking import tracking
from face_processing import face_processing

if __name__ == "__main__":
    queue = Queue()
    permanent_id_counter = Value('i', 1)
    temp_id_counter = Value('i', 1)

    with Manager() as manager:
        temporary_ids = manager.dict()

        tracking_process = Process(target=tracking, args=(VIDEO_PATH, queue, temp_id_counter, permanent_id_counter, temporary_ids))
        face_processing_process = Process(target=face_processing, args=(queue, permanent_id_counter, temporary_ids))

        tracking_process.start()
        face_processing_process.start()

        tracking_process.join()
        face_processing_process.join()