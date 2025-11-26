import os 
import logging
from datetime import datetime
import structlog

class CustomLogger:
    def __init__(self,log_dir="logs"):
        #ensure logs directory exits
        self.logs_dir=os.path.join(os.getced(),log_dir)
        os.makedirs(self.logs_dir,exist_ok=True)
        
        # Timestamped  log file(for persistance)
        log_file = f"{datetime.now().strftime('%m_%d_%Y_%H_%M_%S')}.log"
        self.log_file_path = os.path.join(self.logs_dir,log_file)
        
    def get_logger(self,name=__file__):
        logger_name = os.path.basename(name)
        
        ## configure  logging for console + file(both json)
        file_handler = logging.FileHandler(self.log_file_path)
        file_handler.setLevel(logging.INFO)
        file_handler.setFormatter(logging.Formatter("%(message)s"))  ## RAW json lines
        
        console_handler= logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        console_handler.setFormatter(logging.Formatter("%(message)s"))
        
        logging.basicConfig(
            level=logging.INFO,
            format="%(message)s",## structlog will handle json rendering
            handlers=[console_handler,file_handler]
        )
        
        ## configure structlog for  json structured  logging
        structlog.configure(
            processors=[
                structlog.processors.Timestamper(fmt="iso",utc=True,key="timestamp"),
                structlog.processors.add_log_level,
                structlog.processors.EventRenamer(to="event"),
                structlog.processors.JSONRenderer()   
            ],
            logger_factory=structlog.stdlib.Loggerfactory(),
            cache_logger_on_first_use=True,
        )
        return structlog.get_logger(logger_name)
        
        
        
          


