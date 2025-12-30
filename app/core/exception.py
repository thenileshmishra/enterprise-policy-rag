import sys
from .logger import logger


def error_message_detail(error, error_detail:sys):
    _,_, exc_tb = error_detail.exc_info()

    if exc_tb is None:
        file_name = "<unknown>"
        line_no = 0
    else:
        tb = exc_tb
        while tb.tb_next:
            tb = tb.tb_next
        file_name = tb.tb_frame.f_code.co_filename
        line_no = tb.tb_lineno
    
    error_message = "Error occurred in python script [{0}] at line [{1}] with message: [{2}]".format(
        file_name,
        line_no,
        str(error)
    )
    return error_message

class CustomException(Exception):
    def __init__(self, error_message, error_detail:sys):
        
        self.error_message = error_message_detail(error_message, error_detail=error_detail)
        super().__init__(self.error_message)
        # LOG THE ERROR
        logger.error(self.error_message)
    def __str__(self):
        return self.error_message