from dataclasses import dataclass


@dataclass(frozen=True)
class Config:
    """Fixed settings of the reconstruction pipeline."""

    # Object gating (SAM3)
    detection_threshold: float = 0.5
    min_box_area_fraction: float = 0.10
    sam3_batch_size: int = 4

    # Retrieval and dense matching (MASt3R retrieval, RoMa v2)
    retrieval_neighbours: int = 20
    roma_setting: str = "precise"
    samples_per_pair: int = 8000
    certainty_threshold: float = 0.20
    cycle_fraction: float = 0.004
    epipolar_fraction: float = 0.001
    magsac_confidence: float = 0.999
    magsac_max_iters: int = 10000
    keypoint_merge_fraction: float = 0.002

    # Structure from motion (COLMAP)
    default_focal_factor: float = 1.2
    verify_max_trials: int = 20000
    verify_min_inlier_ratio: float = 0.1
    min_shared_images: int = 3
    max_image_size: int = 1200

    # View selection
    min_confirmed_points: int = 100
    min_confirmed_fraction: float = 0.30
    min_valid_depth: float = 1e-3
    max_conductance: float = 0.15
    min_cluster_size: int = 5
    bridge_points: int = 30
    bridge_images: int = 5

    # Multi-view stereo (CasDiffMVS)
    source_views: int = 6
    depth_range_expand: float = 0.15
    confidence_thresholds: tuple = (0.3, 0.5, 0.5)

    # Monocular verification (MoGe-2, spherical filter)
    min_confident_pixels: int = 1000
    spherical_truncation: float = 1.0
    log_ratio_sigma: float = 1.5
    log_ratio_floor: float = 0.05
    min_component_pixels: int = 32
    min_overlap_pixels: int = 1000
    max_mad_log_ratio: float = 0.20
    min_global_inlier_fraction: float = 0.25

    # Mask selection and final check
    min_mask_mvs_pixels: int = 200
    recall_tie: float = 0.10
    nested_containment: float = 0.90
    min_verified_fraction: float = 0.05
    min_mask_pixels: int = 64
    main_subject_ratio: float = 5.0
    min_focal_ratio: float = 0.3
    max_focal_ratio: float = 10.0
    min_frames: int = 2
