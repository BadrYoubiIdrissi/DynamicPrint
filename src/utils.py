from fontTools import ttLib
import os
from PIL import Image, ImageFilter
from manim import *
import numpy as np
from colour import Color
import ffmpeg
from hsluv import hsluv_to_rgb


def get_font_details(font_path):
    _, type = os.path.splitext(font_path)
    font = ttLib.TTFont(font_path)
    names = {}
    for name in font.names:
        names[name.nameID] = name.toStr()
    family = names[16] if 16 in names else names[1]
    style = names[17] if 17 in names else names[2]
    full_name = names[4] if 4 in names else f"{family} {style}"
    return {"family": family, "style": style, "full_name": full_name}


def points_to_pixel_coords(
    camera,
    points,
):
    shifted_points = points - camera.frame_center

    result = np.zeros((len(points), 2))
    pixel_height = camera.pixel_height
    pixel_width = camera.pixel_width
    frame_height = camera.frame_height
    frame_width = camera.frame_width
    width_mult = pixel_width / frame_width
    width_add = pixel_width / 2
    height_mult = pixel_height / frame_height
    height_add = pixel_height / 2
    # Flip on y-axis as you go
    height_mult *= -1

    result[:, 0] = shifted_points[:, 0] * width_mult + width_add
    result[:, 1] = shifted_points[:, 1] * height_mult + height_add
    return result.astype("int")


def get_surrounding_bbox(mobject, buffer=0):
    bbox = [
        mobject.get_critical_point(UL) + buffer * UL,
        mobject.get_critical_point(DR) + buffer * DR,
    ]
    return np.array(bbox)


class CustomSceneFileWriter(SceneFileWriter):
    
    def open_movie_pipe(self, file_path=None):
        fps = config["frame_rate"]
        if fps == int(fps):  # fps is integer
            fps = int(fps)
       
        height = config["pixel_height"]
        width = config["pixel_width"]
        self.partial_movie_file_path = self.movie_file_path

        self.writing_process = (
            ffmpeg
            .input('pipe:', format='rawvideo', pix_fmt='rgba', s='{}x{}'.format(width, height))
            .output(self.movie_file_path, loglevel=config["ffmpeg_loglevel"].lower(), vcodec='libx264', pix_fmt='yuv420p')
            # .output(self.movie_file_path, vcodec='ffv1', pix_fmt='yuv420p')
            .overwrite_output()
            .run_async(pipe_stdin=True)
        )
    def finish(self):
        """
        Finishes writing to the FFMPEG buffer or writing images
        to output directory.
        Combines the partial movie files into the
        whole scene.
        If save_last_frame is True, saves the last
        frame in the default image directory.
        """
        if hasattr(self, "writing_process"):
            self.writing_process.terminate()


class DropShadow(ImageMobject):
    def __init__(self, mobject, radius=0.1, shadow_color=BLACK, shadow_opacity=0.5, **kwargs):
        self.mobject = mobject
        self.radius = radius
        self.shadow_color = Color(shadow_color)
        self.shadow_opacity = shadow_opacity
        self.cam = Camera()
        self.transparent_image_ = Image.new(
            "RGBA", (config["pixel_width"], config["pixel_height"]), (0, 0, 0, 0)
        )
        
        super().__init__(
            self.get_raster_image(), resampling_algorithm=Image.NEAREST, **kwargs
        )
        self.width = config.frame_width
        self.height = config.frame_height
        self.add_updater(self.update_pixel_array)

    def update_pixel_array(self, m):
        m.pixel_array = np.array(self.get_raster_image())

    def get_raster_image(self):
        self.cam.background = np.array(self.transparent_image_)
        self.cam.reset()
        self.cam.capture_mobject(self.mobject)
        mask = self.cam.pixel_array
        mask[:,:,:3] = (255*np.array(self.shadow_color.get_rgb()).reshape((1,1,3))).astype("uint8")
        mask[:,:,3] = mask[:,:,3] * self.shadow_opacity
        mask = Image.fromarray(mask)

        radius = self.radius * self.cam.pixel_width / self.cam.frame_width
        # We crop the image to make the gaussian filter faster to compute
        buffer = 4 * self.radius + 0.01*self.mobject.get_width() 
        bbox = get_surrounding_bbox(self.mobject, buffer)
        bbox = points_to_pixel_coords(self.cam, bbox)
        bbox[:,0] = np.clip(bbox[:,0], 0, self.cam.pixel_width-1)
        bbox[:,1] = np.clip(bbox[:,1], 0, self.cam.pixel_height-1)
        bbox = tuple(bbox.flatten())
        crop = mask.crop(bbox)
        drop_shadow = crop.filter(ImageFilter.GaussianBlur(radius=radius))
        result = self.transparent_image_.copy()
        result.paste(drop_shadow, bbox)
        return result


def get_slope_from_path(path, alpha, dx=0.001):
    sign = 1 if alpha < 1-dx else -1
    return angle_of_vector(sign * path.point_from_proportion(alpha + sign * dx) - sign * path.point_from_proportion(alpha))

def get_smooth_path(width=6, height=6, radius=1.5, k=20, **kwargs):
    """
    Returns a random smooth path.
    Additional keyword arguments are passed to the VMObject class constructor.
    """
    random_points = poisson_disc_samples(width=width, height=height, radius=radius, k=k)
    np.random.shuffle(random_points)
    # centering poisson disc sampled points
    random_points[:, 0] = random_points[:, 0] - width / 2
    random_points[:, 1] = random_points[:, 1] - height / 2
    # adding z coordinate and setting it to 0
    random_points = np.concatenate(
        [random_points, np.array(np.zeros((len(random_points), 1)))], axis=1
    )
    # creating smooth path
    smooth_path = VMobject(**kwargs)
    smooth_path.set_points_smoothly(random_points)
    return smooth_path


def get_smooth_random_backgrounds(n=2, shape=(5, 5, 3), width=8, height=8):
    bg_images = []
    for i in range(n):
        img = np.random.rand(*shape)
        img[:,:,0] = img[:,:,0] * 360
        img[:,:,1] = img[:,:,1] * 80
        img[:,:,2] = 93
        rgb = []
        for hsl in img.reshape(-1, 3):
            rgb.append(hsluv_to_rgb(hsl))
        rgb = (255*np.array(rgb).reshape(shape)).astype(np.uint8)
        bg_image = ImageMobject(rgb, z_index=-2)
        bg_image.width = width
        bg_image.height = height
        bg_images.append(bg_image)
    return bg_images

def background_animation(bg_images):
    bg_animations = []
    for i in range(len(bg_images) - 1):
        bg_animations.append(Transform(bg_images[0], bg_images[i + 1], run_time=10/(len(bg_images)-1)))
    return Succession(*bg_animations, run_time=10)