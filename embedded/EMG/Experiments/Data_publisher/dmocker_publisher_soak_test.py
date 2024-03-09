import argparse
import time
import zmq
import logging

LOG_LEVEL=logging.DEBUG  #change to logging.DEBUG to enable print logs


ZEROMQ_SOCKET="tcp://127.0.0.1:53421" #zeromq socket adress
DEFAULT_TIME=30   #time in mili seconds between data packets
#DATA_CHUNCK_SIZE=(16 * 1024)  #size of 30ms data chunk
DATA_CHUNCK_SIZE=(64*65*4)  #size of 30ms data chunk


def sleep_ms(ms):
    time.sleep(ms/1000)

def main(mock_file, enable_streaming):
    pass
    context = zmq.Context()
    publisher = context.socket(zmq.PUB)
    publisher.bind(ZEROMQ_SOCKET)

    with open(mock_file, "rb") as file:
        while True:
            data = file.read(DATA_CHUNCK_SIZE)
            logging.debug(f"read {len(data)} from file")
            if len(data) != DATA_CHUNCK_SIZE:
                if enable_streaming:
                    logging.debug("reset file pointer for streaming")
                    file.seek(0)
                else:
                    logging.debug("finish sending")
                    break;    
            else:
                logging.debug("publish data")
                publisher.send(data)
                sleep_ms(DEFAULT_TIME)

    publisher.close()
    context.term()

if __name__ == "__main__":
    logging.basicConfig(level=LOG_LEVEL)    
    parser = argparse.ArgumentParser(description="ARM NPU plataform data publisher for hardware emulation.")
    parser.add_argument("file", help="Path to read mock data.")
    parser.add_argument("-s","--streaming", 
                        action='store_true', 
                        help="enable streaming so data from file is sended over and over.")
    
    args = parser.parse_args()
    main(args.file, args.streaming)

   