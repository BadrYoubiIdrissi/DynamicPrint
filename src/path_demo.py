from manim import *
from manim.utils import space_ops
from poisson_disc_sampling import poisson_disc_samples
import numpy as np
from hsluv import hsluv_to_rgb, hsluv_to_hex

from utils import DropShadow, get_smooth_random_backgrounds, background_animation, get_smooth_path, get_slope_from_path

def rotate_func(alpha):
    return PI * np.cos(2 * PI * alpha) /4

class DynamicPrint(Scene):
    def __init__(self, i, **kwargs):
        super().__init__(**kwargs)
        self.i = i
    def construct(self):
        bg_images = get_smooth_random_backgrounds(n=4)
        bg_animation = background_animation(bg_images)
        
        text = Text(f"{self.i}", font="Roboto Bold", fill_opacity=1, font_size=200, color=hsluv_to_hex((self.i*36, 80, 60)))
        drop_shadow = DropShadow(text, radius=0.05)
        text.save_state()
        smooth_path = get_smooth_path(width=8-1.8*text.width, height=8-1.8*text.height, color=BLACK, stroke_width=2, z_index=0)
        
        def update_text(m, alpha):
            m.restore()
            # m.set_stroke(color=BLACK, opacity=1, width=1)
            m.set_sheen_direction(
                    space_ops.rotate_vector(RIGHT, alpha * PI)
                )
            m.shift(smooth_path.point_from_proportion(alpha))
            m.rotate(get_slope_from_path(smooth_path, alpha, dx=0.001)-PI/2)
            # m.rotate(rotate_func(alpha), axis=RIGHT+UP)

        text.save_state()
       
        self.add(bg_images[0])
        self.add(smooth_path)
        self.add(drop_shadow, text)
        self.play(
            FadeIn(smooth_path, run_time=0.1),
            UpdateFromAlphaFunc(text, update_text, run_time=10),
            bg_animation,
            rate_func=linear
        )

def generate_anim(i):
    base_dir = f'output/folder_{i}'
    with tempconfig(
        dict(
            verbosity="WARNING",
            write_to_movie=True,
            pixel_width=256,
            pixel_height=256,
            frame_height=8,
            frame_width=8,
            frame_rate=24,
            media_dir=f'{base_dir}',
            video_dir='{media_dir}',
            images_dir='{media_dir}/.tmp',
            text_dir='{media_dir}/.tmp',
            log_dir='{media_dir}/.tmp',
            partial_movie_dir='{media_dir}/.tmp',
            output_file=f'video_{i}',
            format='gif',
            renderer="cairo",
            save_pngs=False,
            save_sections=False,
            background_color=WHITE,
            disable_cache=True,
            flush_cache=True,
        )
    ):
        scene = DynamicPrint(i)
        scene.render()


from multiprocessing import Pool

if __name__ == "__main__":
    with Pool(5) as p:
        p.map(generate_anim, range(10))
