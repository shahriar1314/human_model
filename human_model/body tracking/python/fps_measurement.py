import pyzed.sl as sl

init = sl.InitParameters()
init.set_from_svo_file("recorded.svo2")

zed = sl.Camera()
status = zed.open(init)

if status == sl.ERROR_CODE.SUCCESS:
    fps = zed.get_camera_information().camera_configuration.fps
    print("FPS:", fps)
    zed.close()
else:
    print("Error opening SVO file")