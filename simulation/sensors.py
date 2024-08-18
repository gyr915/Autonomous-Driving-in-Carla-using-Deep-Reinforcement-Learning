import math
import numpy as np
import weakref
import pygame
import cv2
from simulation.connection import carla
from simulation.settings import RGB_CAMERA, SSC_CAMERA
from ultralytics import YOLO

# ---------------------------------------------------------------------|
# ------------------------------- CAMERA |
# ---------------------------------------------------------------------|

class CameraSensor():

    def __init__(self, vehicle):
        self.sensor_name = SSC_CAMERA
        self.parent = vehicle
        self.front_camera = list()
        world = self.parent.get_world()
        self.sensor = self._set_camera_sensor(world)
        weak_self = weakref.ref(self)
        self.sensor.listen(
            lambda image: CameraSensor._get_front_camera_data(weak_self, image))

    # Main front camera is setup and provide the visual observations for our network.
    def _set_camera_sensor(self, world):
        front_camera_bp = world.get_blueprint_library().find(self.sensor_name)
        front_camera_bp.set_attribute('image_size_x', f'160')
        front_camera_bp.set_attribute('image_size_y', f'80')
        front_camera_bp.set_attribute('fov', f'125')
        front_camera = world.spawn_actor(front_camera_bp, carla.Transform(
            carla.Location(x=2.4, z=1.5), carla.Rotation(pitch= -10)), attach_to=self.parent)
        # print(front_camera)
        return front_camera

    @staticmethod
    def _get_front_camera_data(weak_self, image):
        self = weak_self()
        if not self:
            return
        image.convert(carla.ColorConverter.CityScapesPalette)
        placeholder = np.frombuffer(image.raw_data, dtype=np.dtype("uint8"))
        placeholder1 = placeholder.reshape((image.width, image.height, 4))
        target = placeholder1[:, :, :3]
        self.front_camera.append(target)#/255.0)

class RGBCameraSensor():
    
    def __init__(self, vehicle):
        pygame.init()
        self.display = pygame.display.set_mode((640, 320),pygame.HWSURFACE | pygame.DOUBLEBUF)
        self.sensor_name = RGB_CAMERA
        self.parent = vehicle
        self.surface2 = None
        self.model = YOLO("yolov8n.pt")
        self.rgb_camera = list()
        world = self.parent.get_world()
        self.sensor = self._set_rgb_camera(world)
        weak_self = weakref.ref(self)
        self.sensor.listen(
            lambda image: RGBCameraSensor._get_rgb_camera_data(weak_self, image))

    def _set_rgb_camera(self, world):
        rgb_camera_bp = world.get_blueprint_library().find(self.sensor_name)
        rgb_camera_bp.set_attribute('image_size_x', '640')
        rgb_camera_bp.set_attribute('image_size_y', '320')
        rgb_camera_bp.set_attribute('fov', '125')
        rgb_camera = world.spawn_actor(rgb_camera_bp, carla.Transform(
            carla.Location(x=2.4, z=1.5), carla.Rotation(pitch=0)), attach_to=self.parent)
        # print(rgb_camera)
        return rgb_camera

    @staticmethod
    def _get_rgb_camera_data(weak_self, image):
        self = weak_self()
        if not self:
            print('No self')
            return
        image.convert(carla.ColorConverter.Raw)
        rgb_image = np.frombuffer(image.raw_data, dtype=np.dtype("uint8"))
        rgb_image = rgb_image.reshape((image.height, image.width, 4))
        rgb_image = rgb_image[:, :, :3]  # Drop the alpha channel
        self.rgb_camera.append(rgb_image)
        rgb_image = rgb_image[:, :, ::-1]
        YOLO_detection = self._YOLO_detection(rgb_image)
        self.surface2 = pygame.surfarray.make_surface(YOLO_detection.swapaxes(0, 1))
        self.display.blit(self.surface2, (720, 0))
        pygame.display.flip()
        # cv2.imshow("",rgb_image)
        # cv2.waitKey(10)
        
        # Here you would pass rgb_image to your YOLO model for detection

    def _YOLO_detection(self, rgb_image):
        results = self.model(rgb_image)
        annotated_frame = results[0].plot()
        # cv2.imshow("", annotated_frame)
        # cv2.waitKey(10)
        return annotated_frame

# ---------------------------------------------------------------------|
# ------------------------------- ENV CAMERA |
# ---------------------------------------------------------------------|

class CameraSensorEnv:

    def __init__(self, vehicle):

        pygame.init()
        self.display = pygame.display.set_mode((1360, 720),pygame.HWSURFACE | pygame.DOUBLEBUF)
        self.sensor_name = RGB_CAMERA
        self.parent = vehicle
        self.surface1 = None
        world = self.parent.get_world()
        self.sensor = self._set_camera_sensor(world)
        weak_self = weakref.ref(self)
        self.sensor.listen(lambda image: CameraSensorEnv._get_third_person_camera(weak_self, image))

    # Third camera is setup and provide the visual observations for our environment.

    def _set_camera_sensor(self, world):

        thrid_person_camera_bp = world.get_blueprint_library().find(self.sensor_name)
        thrid_person_camera_bp.set_attribute('image_size_x', f'720')
        thrid_person_camera_bp.set_attribute('image_size_y', f'720')
        third_camera = world.spawn_actor(thrid_person_camera_bp, carla.Transform(
            carla.Location(x=-4.0, z=2.0), carla.Rotation(pitch=-12.0)), attach_to=self.parent)
        return third_camera

    @staticmethod
    def _get_third_person_camera(weak_self, image):
        self = weak_self()
        if not self:
            return
        array = np.frombuffer(image.raw_data, dtype=np.dtype("uint8"))
        placeholder1 = array.reshape((image.width, image.height, 4))
        placeholder2 = placeholder1[:, :, :3]
        placeholder2 = placeholder2[:, :, ::-1]
        self.surface1 = pygame.surfarray.make_surface(placeholder2.swapaxes(0, 1))
        self.display.blit(self.surface1, (0, 0))
        pygame.display.flip()



# ---------------------------------------------------------------------|
# ------------------------------- COLLISION SENSOR|
# ---------------------------------------------------------------------|

# It's an important as it helps us to tract collisions
# It also helps with resetting the vehicle after detecting any collisions
class CollisionSensor:

    def __init__(self, vehicle) -> None:
        self.sensor_name = 'sensor.other.collision'
        self.parent = vehicle
        self.collision_data = list()
        world = self.parent.get_world()
        self.sensor = self._set_collision_sensor(world)
        weak_self = weakref.ref(self)
        self.sensor.listen(
            lambda event: CollisionSensor._on_collision(weak_self, event))

    # Collision sensor to detect collisions occured in the driving process.
    def _set_collision_sensor(self, world) -> object:
        collision_sensor_bp = world.get_blueprint_library().find(self.sensor_name)
        sensor_relative_transform = carla.Transform(
            carla.Location(x=1.3, z=0.5))
        collision_sensor = world.spawn_actor(
            collision_sensor_bp, sensor_relative_transform, attach_to=self.parent)
        return collision_sensor

    @staticmethod
    def _on_collision(weak_self, event):
        self = weak_self()
        if not self:
            return
        impulse = event.normal_impulse
        intensity = math.sqrt(impulse.x ** 2 + impulse.y ** 2 + impulse.z ** 2)
        self.collision_data.append(intensity)

