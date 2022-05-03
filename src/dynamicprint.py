import functools
from itertools import repeat
from manim import *
from manim.utils import space_ops
from poisson_disc_sampling import poisson_disc_samples
import numpy as np
import datetime
from multiprocessing import Manager, Pool
from utils import get_font_details
import sampling as sample
from copy import deepcopy
import uuid
from omegaconf import OmegaConf
import pandas as pd
from tqdm import tqdm
from hsluv import hsluv_to_hex

class TextGenerator:
    """
    NOT IMPLEMENTED YET. 
    """
    def __init__(self, alphabets, fixed_font=None, **kwargs):
        self.build_possibles_chars(alphabets)
        if fixed_font is not None:
            self.font = fixed_font
        else:
            self.build_possible_fonts()
        self.text_kwargs = kwargs

    def build_possible_fonts():
        raise NotImplementedError()

    def build_possibles_chars(self, alphabets):
        self.possible_chars = []
        self.alphabets = []
        for a in alphabets:
            with open(
                os.path.join("alphabets", a), "r", encoding="utf8", errors="ignore"
            ) as f:
                char_list = [l for l in f.read().splitlines() if len(l) > 0]
                self.possible_chars.extend(char_list)
                self.alphabets.extend([a] * len(char_list))

    def __call__(self):
        char_id = np.random.randint(0, len(self.alphabets))
        char = self.possible_chars[char_id]
        alphabet = self.alphabets[char_id]
        if hasattr(self, "font"):
            font = self.font
        else:
            possible_fonts = self.possible_fonts[alphabet]
            font_id = np.random.randint(0, len(possible_fonts))
            font = possible_fonts[font_id]
        font_details = get_font_details(font)
        with register_font(font):
            return Text(char, font=font_details["full_name"])


class RandomModularTextAnimation(Animation):
    """
    Generates a random timeline for the animations.
    The generated timeline consists of a list of keypoints for each animation type.
    For each keypoint there is a random value for the animated attribute.
    This random value is sampled from a range given as a parameter in `value_ranges`.
    The animation is then a linear interpolation between the keypoints. 

    Params:
    -------
    mobject: Mobject to animate
    possible_animations: list of attributes to potentially animate
    possible_rate_functions: list of rate functions to potentially use (See shorturl.at/kvNRV)
    nb_anims: number of animations in this timeline. This attribute should be a range [min_nb, max_nb] to sample from.
    anims_with_replacement: whether to sample animations with replacement or not
    anim_duration: duration of each animation. This attribute should be a range [min_duration, max_duration] to sample from.
    animation_overlap: overlap between animations. This attribute should be a range [min_overlap, max_overlap] to sample from.
    transition_duration: This duration is used to split an animation into more keypoints in order to have more transitions. This attribute should be a range [min_transition, max_transition] to sample from.
    value_ranges: dictionary of ranges for each animation type.
    """
    def __init__(
        self,
        mobject,
        possible_animations,
        possible_rate_functions,
        nb_anims,
        anims_with_replacement,
        anim_duration,
        animation_overlap,
        transition_duration,
        value_ranges,
        **kwargs,
    ):
        super().__init__(mobject, **kwargs)

        self.possible_animations = possible_animations
        self.possible_rate_functions = possible_rate_functions
        self.anim_duration = anim_duration
        self.transition_duration = transition_duration
        self.animation_overlap = animation_overlap
        self.nb_anims = nb_anims
        self.anims_with_replacement = anims_with_replacement
        self.value_ranges = value_ranges
        self.current_angle = 0
        self.current_scale = 1
        self.generate_keypoints()
        self.generate_values()
        self.generate_animations()

    def get_timeline(self):
        current_time = 0
        previous_anim_duration = 0
        previous_anim_type = None
        timeline = {}
        animation_ids = np.random.choice(
            range(len(self.possible_animations)),
            sample.int(self.nb_anims),
            replace=self.anims_with_replacement,
        )
        for animation_id in animation_ids:
            animation_type = self.possible_animations[animation_id]
            anim_duration = sample.scalar(self.anim_duration)
            animation_overlap = sample.scalar(self.animation_overlap)
            overlap_duration = previous_anim_duration * animation_overlap
            # Can't overlap with previous animation if it's the same type
            overlap_duration *= 1 if (previous_anim_type != animation_type) else 0
            start, end = (
                max(0, current_time - overlap_duration),
                current_time + anim_duration - overlap_duration,
            )
            if animation_type in timeline:
                timeline[animation_type].append([start, end])
            else:
                timeline[animation_type] = [[start, end]]
            current_time = end
            previous_anim_duration = anim_duration
            previous_anim_type = animation_type
        return timeline

    def generate_keypoints(self):
        self.timeline = self.get_timeline()
        for anim_type in self.timeline:
            for i, (start, end) in enumerate(self.timeline[anim_type]):
                duration_per_anim = sample.scalar(self.transition_duration)
                self.timeline[anim_type][i] = np.linspace(
                    start, end, num=max(2, round((end - start) / duration_per_anim))
                )
        self.run_time = max(max(anim[-1]) for anim in self.timeline.values())
        self.timeline = {
            anim_type: [(a / self.run_time).tolist() for a in anim_timeline]
            for anim_type, anim_timeline in self.timeline.items()
        }

    def generate_values(self):
        self.value_samplers = self.get_samplers()
        self.setters = self.get_setters()
        self.init_values = {
            key: value_sampler() for key, value_sampler in self.value_samplers.items()
        }
        for key, value in self.init_values.items():
            self.setters[key](value)
        self.mobject.save_state()

        self.values = {k: [] for k in self.timeline}
        previous_values = deepcopy(self.init_values)
        for anim_type in self.timeline:
            for i in range(len(self.timeline[anim_type])):
                v = [previous_values[anim_type]]
                for _ in range(1, len(self.timeline[anim_type][i])):
                    v.append(self.value_samplers[anim_type]())
                previous_values[anim_type] = v[-1]
                self.values[anim_type].append(v)

    def generate_animations(self):
        self.animations = {
            anim_type: get_alpha_function(
                self.timeline[anim_type],
                self.values[anim_type],
                self.setters[anim_type],
            )
            for anim_type in self.timeline
        }

    def get_samplers(self):
        return {
            "scale": lambda: sample.scalar(self.value_ranges["scale"]),
            "angle": lambda: sample.scalar(self.value_ranges["angle"]),
            "position": lambda: sample.position(self.value_ranges["position"]),
            "fill_color": lambda: sample.color(self.value_ranges["fill_color"]),
            "fill_opacity": lambda: sample.scalar(self.value_ranges["fill_opacity"]),
            "stroke_color": lambda: sample.color(self.value_ranges["stroke_color"]),
            "stroke_opacity": lambda: sample.scalar(
                self.value_ranges["stroke_opacity"]
            ),
            "stroke_width": lambda: sample.scalar(self.value_ranges["stroke_width"]),
            "gradient_angle": lambda: sample.scalar(
                self.value_ranges["gradient_angle"]
            ),
        }

    def set_angle(self, angle):
        self.mobject.rotate(
            angle - self.current_angle, about_point=self.mobject.get_center_of_mass()
        )
        self.current_angle = angle

    def set_scale(self, scale):
        self.mobject.scale(scale / self.current_scale)
        self.current_scale = scale

    def get_setters(self):
        return {
            "scale": lambda val: self.set_scale(val),
            "angle": lambda val: self.set_angle(val),
            "position": lambda val: self.mobject.shift(
                val - self.mobject.get_center_of_mass()
            ),
            "fill_color": lambda val: self.mobject.set_fill(color=hsluv_to_hex(val)),
            "fill_opacity": lambda val: self.mobject.set_fill(opacity=val),
            "stroke_color": lambda val: self.mobject.set_stroke(
                color=hsluv_to_hex(val)
            ),
            "stroke_opacity": lambda val: self.mobject.set_stroke(opacity=val),
            "stroke_width": lambda val: self.mobject.set_stroke(width=val),
            "gradient_angle": lambda val: self.mobject.set_sheen_direction(
                space_ops.rotate_vector(RIGHT, val)
            ),
        }

    def interpolate(self, alpha):
        for anim_type in self.animations:
            self.animations[anim_type](alpha)


def get_interpolation_between_keyframes(keyframes, values, rate_function="linear"):
    rate_function = getattr(rate_functions, rate_function)

    def func(alpha):
        i = np.searchsorted(keyframes, alpha)
        if i == 0:
            a, b = 0, 1
        else:
            a, b = i - 1, i
        return values[a] + (values[b] - values[a]) * rate_function(
            (alpha - keyframes[a]) / (keyframes[b] - keyframes[a])
        )

    return func


def get_alpha_function(keyframes, values, set_attribute_func, rate_function="linear"):
    interpolation_funcs = [
        get_interpolation_between_keyframes(keyframes[i], values[i], rate_function)
        for i in range(len(keyframes))
    ]

    def func(alpha):
        for i, keyframe in enumerate(keyframes):
            if keyframe[0] <= alpha and alpha <= keyframe[-1]:
                set_attribute_func(interpolation_funcs[i](alpha))

    return func


class DynamicPrint(Scene):
    """
    Basic scene to generate an animated character with a single fixed font.
    """
    def __init__(self, cfg=None, **kwargs):
        super().__init__(**kwargs)
        self.cfg = cfg

    def construct(self):
        self.char = chr(np.random.randint(48, 50))
        text = Text(self.char)
        size = [text.width, text.height]
        i = np.argmax(size)
        if i == 0:
            text.scale_to_fit_width(3)
        else:
            text.scale_to_fit_height(3)

        anim = RandomModularTextAnimation(text, **self.cfg)
        self.add(text)
        self.play(anim)

    def get_metadata(self):
        return {"character": self.char}


class AnimationLauncher:
    """
    """
    def __init__(self, base_config):
        self.base_config = base_config
        self.base_dir = (
            base_config["dataset"]["base_dir"]
            if "base_dir" in base_config["dataset"]
            else datetime.datetime.now().strftime("data/%Y-%m-%d/%H-%M-%S")
        )

    def launch(self):
        sample_configs = self.generate_sample_configs()
        total = sum(self.base_config["dataset"]["sizes"])
        if config["runtime"]["multiprocessing"]:
            with Manager() as manager:
                metadata = manager.list()
                with Pool(config["runtime"]["nb_workers"]) as p:
                    list(
                        tqdm(
                            p.imap(
                                functools.partial(self.generate_anim, metadata),
                                args=sample_configs,
                            ),
                            total=total,
                        )
                    )
        else:
            metadata = []
            for sample_config in tqdm(sample_configs, total=total):
                self.generate_anim(sample_config, metadata)
        self.save_metadata(metadata)

    def save_metadata(self, metadata):
        df = pd.DataFrame(metadata)
        df.to_csv(os.path.join(self.base_dir, "metadata.csv"))
        df.index.name = "id"

    def generate_sample_configs(self):
        for i, (s, c) in enumerate(
            zip(self.base_config["dataset"]["sizes"], self.base_config["scenes"])
        ):
            for _ in range(s):
                manim = self.base_config["manim"].copy()
                scene = self.base_config["base_scene"].copy()
                scene = OmegaConf.merge(scene, c)
                idx = str(uuid.uuid4())
                manim["media_dir"] = os.path.join(self.base_dir, idx)
                yield {
                    "idx": idx,
                    "group": i,
                    "manim": OmegaConf.to_container(manim),
                    "scene": OmegaConf.to_container(scene),
                }

    def generate_anim(self, config, metadata_list):
        with tempconfig(config["manim"]):
            scene = DynamicPrint(cfg=config["scene"])
            scene.render()
            metadata = scene.get_metadata()
            metadata["format"] = config["manim"]["format"]
            metadata["group"] = config["group"]
            metadata["idx"] = config["idx"]
            metadata_list.append(metadata)

def get_combinatorial_generalization_scenes_and_durations():
    scenes = []
    durations = []
    origin = {
        "value_ranges": {
            "angle": [0, 0],
            "position": {"x": [-2, -2],"y": [-2, -2]},
            "scale": [1, 1],
        }
    }
    ranges = {
        "rotation": {"angle": [0, 2 * PI]},
        "position_x": {"position": {"x": [-2, 2]}},
        "position_y": {"position": {"y": [-2, 2]}},
        "scale": {"scale": [0.8, 1.2]},
    }
    animation = {
        "rotation": "angle",
        "position_x": "position",
        "position_y": "position",
        "scale": "scale",
    }
    for modif in ["rotation", "position_x", "position_y", "scale"]:
        possible_animation = OmegaConf.merge(origin, 
            {"possible_animations": [animation[modif]]}
        )
        scene = OmegaConf.merge(possible_animation, {"value_ranges": ranges[modif]})
        scenes.append(scene)
        durations.append(300)

    all_anims = {
        'possible_animations': ["angle", "position", "scale"],
        'value_ranges': {
            "angle": [0, 2 * PI],
            "position": {"x": [-2, 2], "y": [-2, 2]},
            "scale": [0.8, 1.2],
        }
    }
    combination_scene = OmegaConf.merge(origin, all_anims)
    combination_scene = OmegaConf.merge(combination_scene, {})
    scenes.append(combination_scene)
    durations.append(300)
    return scenes, durations

if __name__ == "__main__":
    with open("config.yaml", "r") as f:
        config = OmegaConf.load(f)

    scenes, durations = get_combinatorial_generalization_scenes_and_durations()
    config.scenes = scenes
    config.dataset.sizes = durations
    launcher = AnimationLauncher(config)
    launcher.launch()
