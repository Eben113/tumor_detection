"""general settings for all project logs"""
import logging, logging.handlers

root_logger = logging.getLogger()
root_logger.setLevel(logging.INFO) #default level for all handlers is INFO

#remove any handlers that have been added by other packages
for handler in list(root_logger.handlers):
    root_logger.removeHandler(handler)

#custom format
fmt = "%(asctime)s %(levelname)-8s pid:%(process)d %(name)s:%(lineno)03d:%(funcName)s %(message)s"
formatter = root_logger.setFormatter(fmt)

#custom handler to create a stream out of logs
streamHandler = logging.streamHandler()
streamHandler.setFormatter(formatter)
streamHandler.setLevell(logging.DEBUG)#custom level for this handler is debug
root_logger.addHandler(streamHandler)