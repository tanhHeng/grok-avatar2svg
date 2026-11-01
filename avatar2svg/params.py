from __future__ import annotations

from dataclasses import dataclass, replace

BASE_SIZE = 1254


@dataclass(frozen=True)
class Params:
    """Algorithm parameters, see the parameter table in design.md.

    Pixel-valued fields are expressed at the 1254 px base size. Read them through
    `px()` / `area()` so they scale with the actual image; `for_image()` sets `scale`.
    """

    scale: float = 1.0

    # 2. segmentation
    flat_window: int = 3
    flat_std: float = 1.0
    seed_link_de: float = 1.2
    min_seed_area: float = 12.0
    tophat_kernel: float = 15.0
    tophat_thresh: float = 8.0
    tophat_min_area: float = 40.0
    merge_de: float = 2.5
    small_area: float = 50.0
    offcolor_de: float = 10.0
    offcolor_match_de: float = 15.0

    # 3. strokes
    stroke_max_width: float = 12.0
    stroke_min_elongation: float = 4.0
    stroke_max_plateau: float = 0.3
    spur_factor: float = 1.5
    ink_min_contrast: float = 4.0  # delta E between line and side color below which ink coverage is undefined
    tip_min_ink: float = 0.3  # traced tips end where the ridge holds less than this fraction of the line color
    tip_max_trace: float = 64.0  # px; a ridge still going beyond this is another feature, not a fading tip
    limb_radius: float = 2.0  # px; parts of a region narrower than about 2r + 1 are tested as strokes on their own

    # 4. background
    bg_de: float = 3.0

    # 5. boundary graph and junctions
    node_cluster_radius: float = 2.0
    tangent_r0: float = 3.0
    tangent_len: float = 8.0
    pair_angle_deg: float = 25.0
    corner_angle_deg: float = 60.0
    corner_window: float = 4.0

    # 5.5 occlusion continuity
    occlusion_max_gap: float = 64.0
    occlusion_angle_deg: float = 30.0
    occlusion_side_de: float = 6.0
    occlusion_inside: float = 0.7
    occlusion_fit_len: float = 16.0
    occlusion_max_slide: float = 1.5
    occlusion_probe: float = 6.0
    occlusion_corner_deg: float = 35.0
    occlusion_fit_rms: float = 0.5

    # 6. curve fitting
    subpixel_range: float = 3.0
    subpixel_min_de: float = 5.0
    eps_region: float = 0.6
    eps_stroke: float = 0.4
    smooth_region: float = 3.0  # Gaussian sigma (px) applied to region edges between corners before fitting
    smooth_stroke: float = 4.0  # Gaussian sigma (px) applied to stroke centerlines
    stroke_width_sigma: float = 2.0  # width-profile smoothing, in units of the stroke's mean width (min 1 px)

    # 7. coloring
    erode_radius: float = 2.0
    gradients: bool = False
    gradient_min_de: float = 4.0
    gradient_min_var_drop: float = 0.5
    unify_de: float = 1.0

    # 8. output
    underlay_width: float = 2.0  # output pixels, not scaled: seams come from anti-aliasing at render resolution
    precision: int = 1

    def px(self, value: float) -> float:
        return value * self.scale

    def area(self, value: float) -> float:
        return value * self.scale * self.scale

    def for_image(self, width: int, height: int) -> Params:
        return replace(self, scale=max(width, height) / BASE_SIZE)
