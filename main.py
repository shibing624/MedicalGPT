import sys
import os
from settings import LogPath
from logbook import Logger, TimedRotatingFileHandler, StreamHandler
from settings import PlatformPackage
from settings import PlatformClass


if __name__ == "__main__":
    os.environ['TZ'] = 'Asia/Shanghai'
    StreamHandler(sys.stdout, level="DEBUG").push_application
    if not os.path.exists(LogPath):
        os.mkdir(LogPath)
    platform_position = os.getenv("RUN_PACKAGE") if os.getenv("RUN_PACKAGE") else PlatformPackage
    platform_class = os.getenv("RUN_CLASS") if os.getenv("RUN_PACKAGE") else PlatformClass
    log_filename = os.path.join(LogPath, "%s%s" % (platform_position, ".log"))
    TimedRotatingFileHandler(filename=log_filename, level='INFO', date_format='%Y-%m-%d', \
                             bubble=False).push_application()
    log = Logger(platform_class)
    platform_position = __import__(platform_position, fromlist=True)
    if hasattr(platform_position, platform_class):
        platform = getattr(platform_position, platform_class)  # http://blog.csdn.net/d_ker/article/details/53671952
        platform_obj = platform(log=log)
        log.info("开始启动平台:%s" % platform_obj.name)
        try:
            platform_obj.before_run()
            platform_obj.run()

        except Exception as e:
            print("运行出错了")
            log.exception(e)
        finally:
            log.info("结束平台运行:%s" % platform_obj.name)
            platform_obj.after_run()
    else:
        raise Exception("请在settings.py里面设置正确的启动平台")
