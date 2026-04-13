use crate::{
    body::Body,
    utils_hierarchical::{circular_speed, push_binary_pair, tangent_xy},
};
use std::sync::atomic::Ordering;
use ultraviolet::Vec3;

#[derive(Clone, Copy)]
pub enum GalaxyType {
    MilkyWay,
    Triangulum,
    SmallMagellanicCloud,
    Whirlpool,
    Sombrero,
    ESO383_76,
    IC1101,
}

#[derive(Clone, Copy)]
pub enum GlobularTemplate {
    SparseSmall,
    Medium,
    DenseLarge,
}

pub fn create_galaxy_template(galaxy: GalaxyType, center: Vec3, system_id: u16, g_scale: f32) -> Vec<Body> {
    let spec = galaxy_spec(galaxy);
    let mut rng = fastrand::Rng::with_seed(seed_for(galaxy, system_id));

    let mut bodies = Vec::with_capacity(spec.capacity());
    add_core(&mut bodies, center, system_id, &spec, &mut rng);
    add_bulge(&mut bodies, center, system_id, g_scale, &spec, &mut rng);
    add_disc(&mut bodies, center, system_id, g_scale, &spec, &mut rng);
    add_outer_binaries(&mut bodies, center, system_id, g_scale, &spec, &mut rng);

    if let Some(companion) = spec.companion {
        add_companion(&mut bodies, center, system_id, g_scale, &spec, companion, &mut rng);
    }

    normalize_template_system(&mut bodies, center, &spec);
    let smbh_exclusion = (world(spec.bulge_radius_ly) * spec.disc_population.spatial_scale * 0.04).max(4.0);
    apply_flat_rotation_with_epicycles(
        &mut bodies,
        center,
        spec.spin,
        smbh_exclusion,
        &spec.disc_population,
    );
    apply_minimal_physical_radii(&mut bodies);
    bodies
}

pub fn create_globular_template(
    template: GlobularTemplate,
    center: Vec3,
    system_id: u16,
    g_scale: f32,
    scale_pow: u8,
) -> Vec<Body> {
    let spec = globular_spec(template);
    let size_mult = 3.0_f32.powi(scale_pow as i32);
    let count_mult = 3usize.pow(scale_pow as u32);
    let star_count = spec.star_count.saturating_mul(count_mult).min(600_000);
    let radius = spec.radius * size_mult;
    let core_radius = spec.core_radius * size_mult;

    let mut rng = fastrand::Rng::with_seed(globular_seed(template, system_id, scale_pow));
    let mut stars: Vec<(f32, Body)> = Vec::with_capacity(star_count);

    for _ in 0..star_count {
        // Truncated Plummer profile for realistic globular concentration.
        let u = rng.f32().clamp(1e-6, 1.0 - 1e-6);
        let r = (core_radius * (u.powf(-2.0 / 3.0) - 1.0).sqrt()).min(radius);
        let dir = random_unit_vec3(&mut rng);
        let pos = center + dir * r;
        let mass = spec.mass_min + rng.f32() * (spec.mass_max - spec.mass_min);
        stars.push((r, Body::new_with_system(pos, Vec3::zero(), mass, 0.00264, system_id)));
    }

    stars.sort_by(|(ra, _), (rb, _)| ra.partial_cmp(rb).unwrap_or(std::cmp::Ordering::Equal));

    let mut enclosed_mass = 0.0_f32;
    for (r, body) in &mut stars {
        enclosed_mass += body.mass;
        let rr = (*r).max(1.0);
        let v_circ = circular_speed(enclosed_mass, rr, g_scale, 1.0);
        let v_esc = (2.0 * enclosed_mass / rr).sqrt() * g_scale.sqrt();
        let v_disp = (v_circ * spec.dispersion * spec.vel_scale).min(v_esc * 0.72);
        let dir = body.pos - center;
        let radial = if dir.mag_sq() > 1e-6 { dir.normalized() } else { Vec3::zero() };
        let outer_frac = (rr / radius.max(1.0)).clamp(0.0, 1.0);
        let inward = radial * (v_circ * spec.infall_outer * outer_frac);
        body.vel = random_unit_vec3(&mut rng) * v_disp - inward;
        body.arm_strength = 0.0;
    }

    let mut bodies: Vec<Body> = stars.into_iter().map(|(_, b)| b).collect();
    let (dt_mass_scale, dt_speed_scale) = template_step_stability_scales();
    for body in &mut bodies {
        body.mass *= dt_mass_scale;
        body.vel *= dt_speed_scale;
    }

    // Remove center-of-mass drift so the cluster stays coherent after spawn.
    let mut total_mass = 0.0_f32;
    let mut com_vel = Vec3::zero();
    for body in &bodies {
        total_mass += body.mass;
        com_vel += body.vel * body.mass;
    }
    if total_mass > 0.0 {
        com_vel /= total_mass;
        for body in &mut bodies {
            body.vel -= com_vel;
        }
    }

    bodies
}

#[derive(Clone, Copy)]
struct GlobularSpec {
    star_count: usize,
    radius: f32,
    core_radius: f32,
    mass_min: f32,
    mass_max: f32,
    dispersion: f32,
    vel_scale: f32,
    infall_outer: f32,
}

fn globular_spec(template: GlobularTemplate) -> GlobularSpec {
    match template {
        // ',' -> small and sparse
        GlobularTemplate::SparseSmall => GlobularSpec {
            star_count: 1_200,
            radius: 420.0,
            core_radius: 110.0,
            mass_min: 0.45,
            mass_max: 1.10,
            dispersion: 0.48,
            vel_scale: 0.78,
            infall_outer: 0.03,
        },
        // '.' -> medium
        GlobularTemplate::Medium => GlobularSpec {
            star_count: 4_200,
            radius: 600.0,
            core_radius: 120.0,
            mass_min: 0.45,
            mass_max: 1.25,
            dispersion: 0.54,
            vel_scale: 0.82,
            infall_outer: 0.04,
        },
        // '-' -> many and comparatively dense
        GlobularTemplate::DenseLarge => GlobularSpec {
            star_count: 12_000,
            radius: 760.0,
            core_radius: 110.0,
            mass_min: 0.50,
            mass_max: 1.40,
            dispersion: 0.60,
            vel_scale: 0.86,
            infall_outer: 0.05,
        },
    }
}

fn globular_seed(template: GlobularTemplate, system_id: u16, scale_pow: u8) -> u64 {
    let tag = match template {
        GlobularTemplate::SparseSmall => 0x7a11_b31du64,
        GlobularTemplate::Medium => 0x29b7_9a51u64,
        GlobularTemplate::DenseLarge => 0xc38f_4d19u64,
    };
    tag ^ ((system_id as u64) << 20) ^ ((scale_pow as u64) << 8)
}

fn random_unit_vec3(rng: &mut fastrand::Rng) -> Vec3 {
    loop {
        let v = Vec3::new(
            rng.f32() * 2.0 - 1.0,
            rng.f32() * 2.0 - 1.0,
            rng.f32() * 2.0 - 1.0,
        );
        let m2 = v.mag_sq();
        if m2 > 1e-6 && m2 <= 1.0 {
            return v / m2.sqrt();
        }
    }
}

#[derive(Clone, Copy)]
struct GalaxySpec {
    morphology: GalaxyMorphology,
    visible_radius_ly: f32,
    bulge_radius_ly: f32,
    bulge_height_ly: f32,
    disc_height_ly: f32,
    halo_radius_ly: f32,
    core_bodies: usize,
    bulge_bodies: usize,
    disc_bodies: usize,
    binary_pairs: usize,
    arm_count: usize,
    arm_twist: f32,
    arm_width: f32,
    bar_fraction: f32,
    bar_length_ly: f32,
    irregularity: f32,
    warp: f32,
    flattening: f32,
    disc_bias: f32,
    spin: f32,
    core_mass: f32,
    bulge_mass: f32,
    disc_mass: f32,
    halo_mass: f32,
    rotation_support: f32,
    arm_contrast: f32,
    arm_coherence: f32,
    dispersion_scale: f32,
    disc_population: DiscPopulationProfile,
    companion: Option<CompanionSpec>,
}

#[derive(Clone, Copy)]
struct DiscPopulationProfile {
    spatial_scale: f32,
    outer_count_bias: f32,
    inner_mass_scale: f32,
    outer_mass_scale: f32,
    inner_radius_scale: f32,
    outer_radius_scale: f32,
    mass_falloff_exp: f32,
    radius_falloff_exp: f32,
    disc_mass_multiplier: f32,
    arm_contrast_gain: f32,
    arm_lock_gain: f32,
    arm_noise_scale: f32,
    inner_orbit_boost: f32,
    outer_orbit_boost: f32,
    outer_osc_start_frac: f32,
    outer_osc_boost_max: f32,
    outer_osc_amp_cap: f32,
}

impl GalaxySpec {
    fn capacity(self) -> usize {
        self.core_bodies + self.bulge_bodies + self.disc_bodies + self.binary_pairs * 2
            + self.companion.map(|companion| companion.bodies).unwrap_or(0)
    }
}

#[derive(Clone, Copy)]
struct CompanionSpec {
    offset_ly: f32,
    radius_ly: f32,
    bodies: usize,
    orbit_tilt: f32,
    orbit_phase: f32,
    orbit_speed_scale: f32,
    spin: f32,
}

#[derive(Clone, Copy)]
enum GalaxyMorphology {
    Spiral,
    Irregular,
    Lenticular,
    Elliptical,
    CD,
}

const LY_TO_WORLD: f32 = 0.03;
const TEMPLATE_POPULATION_SCALE: usize = 6;
const MIN_PHYSICAL_RADIUS: f32 = 0.01;
const TARGET_PHYSICAL_RADIUS: f32 = MIN_PHYSICAL_RADIUS * 4.0;

const STAR_RADIUS: f32 = TARGET_PHYSICAL_RADIUS;
const CORE_RADIUS: f32 = TARGET_PHYSICAL_RADIUS;
const CORE_ORBITER_RADIUS: f32 = TARGET_PHYSICAL_RADIUS;
const CORE_RADIUS_CAP: f32 = TARGET_PHYSICAL_RADIUS;
const FLAT_ROTATION_SPEED_UNITS: f32 = 240.0;
const FLAT_ROTATION_SPEED_SCALE: f32 = 0.01;
const EPICYCLE_AMPLITUDE_MIN: f32 = 0.05;
const EPICYCLE_AMPLITUDE_MAX: f32 = 0.10;
const EPICYCLE_FREQUENCY_FACTOR: f32 = 2.0;

const fn scaled_count(count: usize) -> usize {
    count * TEMPLATE_POPULATION_SCALE
}

fn template_step_stability_scales() -> (f32, f32) {
    let dt_factor = f32::from_bits(crate::renderer::DT_FACTOR_BITS.load(Ordering::Relaxed))
        .abs()
        .max(1.0);
    if dt_factor <= 1.0 {
        return (1.0, 1.0);
    }

    let relief = dt_factor.powf(0.25);
    let speed_scale = 1.0 / relief;
    let mass_scale = 1.0 / (relief * relief);
    (mass_scale, speed_scale)
}

struct NormalizationTargets {
    target_v_rms: f32,
    target_mass_per_radius: f32,
    mass_scale_min: f32,
    mass_scale_max: f32,
    vel_corr_min: f32,
    vel_corr_max: f32,
}

fn normalization_targets() -> NormalizationTargets {
    match crate::simulation::TEMPLATE_PROFILE_IDX.load(Ordering::Relaxed) {
        1 => NormalizationTargets {
            target_v_rms: 3.1,
            target_mass_per_radius: 1.3,
            mass_scale_min: 0.45,
            mass_scale_max: 2.9,
            vel_corr_min: 0.80,
            vel_corr_max: 1.20,
        },
        _ => NormalizationTargets {
            target_v_rms: 2.4,
            target_mass_per_radius: 1.1,
            mass_scale_min: 0.55,
            mass_scale_max: 2.4,
            vel_corr_min: 0.88,
            vel_corr_max: 1.12,
        },
    }
}

fn galaxy_spec(galaxy: GalaxyType) -> GalaxySpec {
    match galaxy {
        GalaxyType::MilkyWay => GalaxySpec {
            morphology: GalaxyMorphology::Spiral,
            visible_radius_ly: 52_850.0,
            bulge_radius_ly: 8_000.0,
            bulge_height_ly: 3_500.0,
            disc_height_ly: 1_400.0,
            halo_radius_ly: 72_000.0,
            core_bodies: scaled_count(4),
            bulge_bodies: scaled_count(360),
            disc_bodies: scaled_count(1_080),
            binary_pairs: scaled_count(90),
            arm_count: 4,
            arm_twist: 7.2,
            arm_width: 0.095,
            bar_fraction: 0.28,
            bar_length_ly: 14_000.0,
            irregularity: 0.012,
            warp: 0.045,
            flattening: 0.92,
            disc_bias: 1.85,
            spin: 1.0,
            core_mass: 4.0,
            bulge_mass: 416.0,
            disc_mass: 1_120.0,
            halo_mass: 240.0,
            rotation_support: 1.05,
            arm_contrast: 0.98,
            arm_coherence: 0.96,
            dispersion_scale: 0.42,
            disc_population: DiscPopulationProfile {
                spatial_scale: 0.80,
                outer_count_bias: 1.22,
                inner_mass_scale: 4.8,
                outer_mass_scale: 0.24,
                inner_radius_scale: 1.34,
                outer_radius_scale: 0.58,
                mass_falloff_exp: 1.62,
                radius_falloff_exp: 1.34,
                disc_mass_multiplier: 1.24,
                arm_contrast_gain: 1.30,
                arm_lock_gain: 1.38,
                arm_noise_scale: 0.45,
                inner_orbit_boost: 1.46,
                outer_orbit_boost: 1.02,
                outer_osc_start_frac: 0.34,
                outer_osc_boost_max: 1.90,
                outer_osc_amp_cap: 0.23,
            },
            companion: None,
        },
        GalaxyType::Triangulum => GalaxySpec {
            morphology: GalaxyMorphology::Spiral,
            visible_radius_ly: 30_000.0,
            bulge_radius_ly: 3_200.0,
            bulge_height_ly: 1_200.0,
            disc_height_ly: 900.0,
            halo_radius_ly: 40_000.0,
            core_bodies: scaled_count(2),
            bulge_bodies: scaled_count(110),
            disc_bodies: scaled_count(820),
            binary_pairs: scaled_count(54),
            arm_count: 3,
            arm_twist: 5.0,
            arm_width: 0.15,
            bar_fraction: 0.0,
            bar_length_ly: 0.0,
            irregularity: 0.035,
            warp: 0.02,
            flattening: 0.95,
            disc_bias: 1.65,
            spin: 1.0,
            core_mass: 3.0,
            bulge_mass: 105.0,
            disc_mass: 760.0,
            halo_mass: 130.0,
            rotation_support: 0.98,
            arm_contrast: 0.78,
            arm_coherence: 0.84,
            dispersion_scale: 0.56,
            disc_population: DiscPopulationProfile {
                spatial_scale: 0.82,
                outer_count_bias: 1.35,
                inner_mass_scale: 3.5,
                outer_mass_scale: 0.22,
                inner_radius_scale: 1.22,
                outer_radius_scale: 0.56,
                mass_falloff_exp: 1.34,
                radius_falloff_exp: 1.18,
                disc_mass_multiplier: 1.14,
                arm_contrast_gain: 1.22,
                arm_lock_gain: 1.30,
                arm_noise_scale: 0.52,
                inner_orbit_boost: 1.38,
                outer_orbit_boost: 1.12,
                outer_osc_start_frac: 0.30,
                outer_osc_boost_max: 2.20,
                outer_osc_amp_cap: 0.28,
            },
            companion: None,
        },
        GalaxyType::SmallMagellanicCloud => GalaxySpec {
            morphology: GalaxyMorphology::Irregular,
            visible_radius_ly: 9_000.0,
            bulge_radius_ly: 1_800.0,
            bulge_height_ly: 1_700.0,
            disc_height_ly: 1_500.0,
            halo_radius_ly: 13_000.0,
            core_bodies: scaled_count(1),
            bulge_bodies: scaled_count(70),
            disc_bodies: scaled_count(430),
            binary_pairs: scaled_count(36),
            arm_count: 0,
            arm_twist: 0.0,
            arm_width: 0.5,
            bar_fraction: 0.55,
            bar_length_ly: 4_500.0,
            irregularity: 0.42,
            warp: 0.08,
            flattening: 0.8,
            disc_bias: 1.2,
            spin: 0.8,
            core_mass: 1.2,
            bulge_mass: 80.8,
            disc_mass: 320.0,
            halo_mass: 80.0,
            rotation_support: 0.66,
            arm_contrast: 0.18,
            arm_coherence: 0.28,
            dispersion_scale: 1.18,
            disc_population: DiscPopulationProfile {
                spatial_scale: 0.86,
                outer_count_bias: 1.18,
                inner_mass_scale: 2.2,
                outer_mass_scale: 0.42,
                inner_radius_scale: 1.08,
                outer_radius_scale: 0.72,
                mass_falloff_exp: 1.08,
                radius_falloff_exp: 0.94,
                disc_mass_multiplier: 1.10,
                arm_contrast_gain: 0.72,
                arm_lock_gain: 0.65,
                arm_noise_scale: 0.95,
                inner_orbit_boost: 1.20,
                outer_orbit_boost: 1.18,
                outer_osc_start_frac: 0.28,
                outer_osc_boost_max: 2.35,
                outer_osc_amp_cap: 0.30,
            },
            companion: None,
        },
        GalaxyType::Whirlpool => GalaxySpec {
            morphology: GalaxyMorphology::Spiral,
            visible_radius_ly: 38_000.0,
            bulge_radius_ly: 6_000.0,
            bulge_height_ly: 2_500.0,
            disc_height_ly: 1_000.0,
            halo_radius_ly: 52_000.0,
            core_bodies: scaled_count(3),
            bulge_bodies: scaled_count(220),
            disc_bodies: scaled_count(960),
            binary_pairs: scaled_count(70),
            arm_count: 2,
            arm_twist: 8.8,
            arm_width: 0.085,
            bar_fraction: 0.0,
            bar_length_ly: 0.0,
            irregularity: 0.016,
            warp: 0.03,
            flattening: 0.96,
            disc_bias: 1.7,
            spin: 1.0,
            core_mass: 3.0,
            bulge_mass: 203.0,
            disc_mass: 840.0,
            halo_mass: 160.0,
            rotation_support: 1.08,
            arm_contrast: 0.99,
            arm_coherence: 0.97,
            dispersion_scale: 0.40,
            disc_population: DiscPopulationProfile {
                spatial_scale: 0.79,
                outer_count_bias: 1.26,
                inner_mass_scale: 5.1,
                outer_mass_scale: 0.22,
                inner_radius_scale: 1.36,
                outer_radius_scale: 0.56,
                mass_falloff_exp: 1.70,
                radius_falloff_exp: 1.46,
                disc_mass_multiplier: 1.28,
                arm_contrast_gain: 1.35,
                arm_lock_gain: 1.46,
                arm_noise_scale: 0.40,
                inner_orbit_boost: 1.52,
                outer_orbit_boost: 1.11,
                outer_osc_start_frac: 0.33,
                outer_osc_boost_max: 2.10,
                outer_osc_amp_cap: 0.27,
            },
            companion: Some(CompanionSpec {
                offset_ly: 31_000.0,
                radius_ly: 8_500.0,
                bodies: scaled_count(220),
                orbit_tilt: 0.35,
                orbit_phase: 1.25,
                orbit_speed_scale: 0.86,
                spin: -0.9,
            }),
        },
        GalaxyType::Sombrero => GalaxySpec {
            morphology: GalaxyMorphology::Lenticular,
            visible_radius_ly: 24_000.0,
            bulge_radius_ly: 9_500.0,
            bulge_height_ly: 7_000.0,
            disc_height_ly: 320.0,
            halo_radius_ly: 36_000.0,
            core_bodies: scaled_count(3),
            bulge_bodies: scaled_count(620),
            disc_bodies: scaled_count(620),
            binary_pairs: scaled_count(48),
            arm_count: 0,
            arm_twist: 0.0,
            arm_width: 0.06,
            bar_fraction: 0.0,
            bar_length_ly: 0.0,
            irregularity: 0.01,
            warp: 0.0,
            flattening: 0.46,
            disc_bias: 2.0,
            spin: 1.0,
            core_mass: 3.0,
            bulge_mass: 585.0,
            disc_mass: 500.0,
            halo_mass: 100.0,
            rotation_support: 0.90,
            arm_contrast: 0.03,
            arm_coherence: 0.16,
            dispersion_scale: 0.62,
            disc_population: DiscPopulationProfile {
                spatial_scale: 0.76,
                outer_count_bias: 1.10,
                inner_mass_scale: 6.2,
                outer_mass_scale: 0.28,
                inner_radius_scale: 1.58,
                outer_radius_scale: 0.60,
                mass_falloff_exp: 1.92,
                radius_falloff_exp: 1.62,
                disc_mass_multiplier: 1.38,
                arm_contrast_gain: 0.55,
                arm_lock_gain: 0.58,
                arm_noise_scale: 1.00,
                inner_orbit_boost: 1.58,
                outer_orbit_boost: 1.06,
                outer_osc_start_frac: 0.36,
                outer_osc_boost_max: 1.85,
                outer_osc_amp_cap: 0.24,
            },
            companion: None,
        },
        GalaxyType::ESO383_76 => GalaxySpec {
            morphology: GalaxyMorphology::Elliptical,
            visible_radius_ly: 60_000.0,
            bulge_radius_ly: 18_000.0,
            bulge_height_ly: 14_000.0,
            disc_height_ly: 6_500.0,
            halo_radius_ly: 92_000.0,
            core_bodies: scaled_count(3),
            bulge_bodies: scaled_count(760),
            disc_bodies: scaled_count(260),
            binary_pairs: scaled_count(84),
            arm_count: 0,
            arm_twist: 0.0,
            arm_width: 0.34,
            bar_fraction: 0.0,
            bar_length_ly: 0.0,
            irregularity: 0.08,
            warp: 0.01,
            flattening: 0.72,
            disc_bias: 1.4,
            spin: 0.35,
            core_mass: 3.0,
            bulge_mass: 791.0,
            disc_mass: 180.0,
            halo_mass: 190.0,
            rotation_support: 0.40,
            arm_contrast: 0.0,
            arm_coherence: 0.0,
            dispersion_scale: 1.34,
            disc_population: DiscPopulationProfile {
                spatial_scale: 0.84,
                outer_count_bias: 1.14,
                inner_mass_scale: 3.9,
                outer_mass_scale: 0.34,
                inner_radius_scale: 1.30,
                outer_radius_scale: 0.70,
                mass_falloff_exp: 1.38,
                radius_falloff_exp: 1.20,
                disc_mass_multiplier: 1.20,
                arm_contrast_gain: 0.58,
                arm_lock_gain: 0.62,
                arm_noise_scale: 0.96,
                inner_orbit_boost: 1.28,
                outer_orbit_boost: 1.08,
                outer_osc_start_frac: 0.35,
                outer_osc_boost_max: 1.78,
                outer_osc_amp_cap: 0.22,
            },
            companion: None,
        },
        GalaxyType::IC1101 => GalaxySpec {
            morphology: GalaxyMorphology::CD,
            visible_radius_ly: 275_000.0,
            bulge_radius_ly: 85_000.0,
            bulge_height_ly: 65_000.0,
            disc_height_ly: 18_000.0,
            halo_radius_ly: 420_000.0,
            core_bodies: scaled_count(5),
            bulge_bodies: scaled_count(1_120),
            disc_bodies: scaled_count(900),
            binary_pairs: scaled_count(140),
            arm_count: 0,
            arm_twist: 0.0,
            arm_width: 0.28,
            bar_fraction: 0.0,
            bar_length_ly: 0.0,
            irregularity: 0.05,
            warp: 0.0,
            flattening: 0.82,
            disc_bias: 1.25,
            spin: 0.22,
            core_mass: 6.0,
            bulge_mass: 1_366.0,
            disc_mass: 680.0,
            halo_mass: 320.0,
            rotation_support: 0.32,
            arm_contrast: 0.0,
            arm_coherence: 0.0,
            dispersion_scale: 1.48,
            disc_population: DiscPopulationProfile {
                spatial_scale: 0.74,
                outer_count_bias: 1.08,
                inner_mass_scale: 7.4,
                outer_mass_scale: 0.32,
                inner_radius_scale: 1.74,
                outer_radius_scale: 0.72,
                mass_falloff_exp: 2.05,
                radius_falloff_exp: 1.78,
                disc_mass_multiplier: 1.46,
                arm_contrast_gain: 0.46,
                arm_lock_gain: 0.48,
                arm_noise_scale: 1.04,
                inner_orbit_boost: 1.72,
                outer_orbit_boost: 1.04,
                outer_osc_start_frac: 0.38,
                outer_osc_boost_max: 1.70,
                outer_osc_amp_cap: 0.20,
            },
            companion: None,
        },
    }
}

fn seed_for(galaxy: GalaxyType, system_id: u16) -> u64 {
    let tag = match galaxy {
        GalaxyType::MilkyWay => 0x9e37_79b1u64,
        GalaxyType::Triangulum => 0x243f_6a88u64,
        GalaxyType::SmallMagellanicCloud => 0xb7e1_5163u64,
        GalaxyType::Whirlpool => 0x94d0_49bbu64,
        GalaxyType::Sombrero => 0x3c6e_f372u64,
        GalaxyType::ESO383_76 => 0xa54f_f53au64,
        GalaxyType::IC1101 => 0x510e_527fu64,
    };
    tag ^ ((system_id as u64) << 20)
}

fn world(ly: f32) -> f32 {
    ly * LY_TO_WORLD
}

fn signed(rng: &mut fastrand::Rng) -> f32 {
    rng.f32() * 2.0 - 1.0
}

fn noisy(rng: &mut fastrand::Rng) -> f32 {
    (signed(rng) + signed(rng) + signed(rng)) / 3.0
}

fn mass_inside(spec: &GalaxySpec, radius: f32) -> f32 {
    let bulge_frac = (radius / world(spec.bulge_radius_ly).max(1.0)).clamp(0.0, 1.0).powf(1.25);
    let disc_frac = (radius / world(spec.visible_radius_ly).max(1.0)).clamp(0.0, 1.0).powf(1.1);
    let halo_frac = (radius / world(spec.halo_radius_ly).max(1.0)).clamp(0.0, 1.0).powf(0.9);
    spec.core_mass + spec.bulge_mass * bulge_frac + spec.disc_mass * disc_frac + spec.halo_mass * halo_frac
}

fn shortest_angle_delta(from: f32, to: f32) -> f32 {
    let tau = std::f32::consts::TAU;
    let mut delta = (to - from).rem_euclid(tau);
    if delta > std::f32::consts::PI {
        delta -= tau;
    }
    delta
}

fn orbital_speed_for_spec(spec: &GalaxySpec, radius: f32, g_scale: f32) -> f32 {
    let visible_radius = world(spec.visible_radius_ly).max(1.0);
    let x = (radius / visible_radius).max(0.0);
    let base = circular_speed(mass_inside(spec, radius), radius.max(1.0), g_scale, 1.0);
    let curve = match spec.morphology {
        GalaxyMorphology::Spiral => {
            let rise = (x / 0.16).clamp(0.0, 1.0);
            let flat = 0.88 + 0.18 * (1.0 - (-3.2 * x).exp());
            let fall = 1.0 / (1.0 + 0.18 * x * x).sqrt();
            rise * flat * fall
        }
        GalaxyMorphology::Irregular => {
            let rise = (x / 0.24).clamp(0.0, 1.0);
            let fall = 1.0 / (1.0 + 0.32 * x * x).sqrt();
            rise * fall
        }
        GalaxyMorphology::Lenticular => {
            let rise = (x / 0.20).clamp(0.0, 1.0);
            let fall = 1.0 / (1.0 + 0.24 * x * x).sqrt();
            rise * (0.96 * fall)
        }
        GalaxyMorphology::Elliptical => {
            let rise = (x / 0.28).clamp(0.0, 1.0);
            let fall = 1.0 / (1.0 + 0.46 * x * x).sqrt();
            rise * (0.72 * fall)
        }
        GalaxyMorphology::CD => {
            let rise = (x / 0.34).clamp(0.0, 1.0);
            let fall = 1.0 / (1.0 + 0.58 * x * x).sqrt();
            rise * (0.64 * fall)
        }
    };
    base * curve * spec.rotation_support
}

fn disc_dispersion(spec: &GalaxySpec, arm_membership: f32) -> f32 {
    let arm_cooling = 1.0 - arm_membership * (0.42 * spec.arm_coherence);
    spec.dispersion_scale * arm_cooling.max(0.45)
}

fn add_core(
    bodies: &mut Vec<Body>,
    center: Vec3,
    system_id: u16,
    spec: &GalaxySpec,
    rng: &mut fastrand::Rng,
) {
    let mut core = Body::new_with_system(center, Vec3::zero(), spec.core_mass, CORE_RADIUS, system_id);
    core.arm_strength = 0.0;
    bodies.push(core);
    let core_orbit_radius = world(spec.bulge_radius_ly) * spec.disc_population.spatial_scale * 0.08;
    let orbiters = spec.core_bodies.saturating_sub(1).max(1);
    for index in 0..orbiters {
        let phase = index as f32 * std::f32::consts::TAU / orbiters as f32;
        let (sin_p, cos_p) = phase.sin_cos();
        let pos = center
            + Vec3::new(
                cos_p * core_orbit_radius,
                sin_p * core_orbit_radius,
                noisy(rng) * core_orbit_radius * 0.12,
            );
        let tangent = tangent_xy(pos - center, spec.spin);
        let speed = circular_speed(spec.core_mass * 1.2, core_orbit_radius, 1.0, 0.35);
        let mut body = Body::new_with_system(
            pos,
            tangent * speed,
            spec.core_mass / orbiters as f32 * 0.15,
            CORE_ORBITER_RADIUS,
            system_id,
        );
        body.arm_strength = 0.0;
        bodies.push(body);
    }
}

fn add_bulge(
    bodies: &mut Vec<Body>,
    center: Vec3,
    system_id: u16,
    g_scale: f32,
    spec: &GalaxySpec,
    rng: &mut fastrand::Rng,
) {
    let bulge_radius = world(spec.bulge_radius_ly) * spec.disc_population.spatial_scale;
    let bulge_height = world(spec.bulge_height_ly) * spec.disc_population.spatial_scale;
    let mass_per_body = spec.bulge_mass / spec.bulge_bodies.max(1) as f32;

    for _ in 0..spec.bulge_bodies {
        let angle = rng.f32() * std::f32::consts::TAU;
        let (sin_a, cos_a) = angle.sin_cos();
        let radius = bulge_radius * rng.f32().powf(1.9);
        let flatten = 0.65 + 0.35 * spec.flattening;
        let offset = Vec3::new(
            cos_a * radius,
            sin_a * radius * flatten,
            noisy(rng) * bulge_height * (1.0 - radius / bulge_radius).max(0.15),
        );
        let tangent = tangent_xy(offset, spec.spin);
        let speed = orbital_speed_for_spec(spec, radius, g_scale) * 0.82;
        let dispersion = Vec3::new(noisy(rng), noisy(rng), noisy(rng) * 1.3)
            * speed
            * (0.05 * spec.dispersion_scale.max(0.8));
        let mut body = Body::new_with_system(
            center + offset,
            tangent * speed + dispersion,
            mass_per_body,
            STAR_RADIUS,
            system_id,
        );
        body.arm_strength = 0.0;
        bodies.push(body);
    }
}

fn add_disc(
    bodies: &mut Vec<Body>,
    center: Vec3,
    system_id: u16,
    g_scale: f32,
    spec: &GalaxySpec,
    rng: &mut fastrand::Rng,
) {
    let visible_radius = world(spec.visible_radius_ly) * spec.disc_population.spatial_scale;
    let disc_height = world(spec.disc_height_ly) * spec.disc_population.spatial_scale;
    let bar_length = world(spec.bar_length_ly) * spec.disc_population.spatial_scale;
    let mass_per_body = spec.disc_mass / spec.disc_bodies.max(1) as f32;
    let target_disc_mass = spec.disc_mass * spec.disc_population.disc_mass_multiplier;
    let disc_start = bodies.len();
    let mut built_disc_mass = 0.0f32;
    let arm_wave_amp = if spec.arm_count > 0 {
        (spec.arm_contrast * spec.arm_coherence * spec.disc_population.arm_contrast_gain * 0.14)
            .clamp(0.0, 0.24)
    } else {
        0.0
    };

    for _ in 0..spec.disc_bodies {
        let base_angle = rng.f32() * std::f32::consts::TAU;
        let radial_u = rng.f32();
        let outer_bias = spec.disc_population.outer_count_bias.max(0.2);
        let legacy_bias = (1.0 / spec.disc_bias.max(0.25)).clamp(0.35, 2.4);
        let radial_frac = (1.0 - (1.0 - radial_u).powf(outer_bias)).powf(legacy_bias);
        let radius = (visible_radius * radial_frac).max(1.0);
        let radial_norm = (radius / visible_radius.max(1.0)).clamp(0.0, 1.0);
        let in_bar = spec.bar_fraction > 0.0 && radius < bar_length && rng.f32() < spec.bar_fraction;

        let offset = if in_bar {
            let x = signed(rng) * bar_length;
            let y = noisy(rng) * visible_radius * spec.arm_width * 0.06;
            let z = noisy(rng) * disc_height * 0.6;
            Vec3::new(x, y, z)
        } else {
            let irregular = visible_radius * spec.irregularity;
            let x = base_angle.cos() * radius + noisy(rng) * irregular;
            let y = base_angle.sin() * radius + noisy(rng) * irregular;
            let warp = spec.warp * radius * (base_angle * 1.5).sin();
            let z = noisy(rng) * disc_height * (0.25 + 0.75 * radius / visible_radius) + warp;
            Vec3::new(x, y, z)
        };

        let planar_radius = Vec3::new(offset.x, offset.y, 0.0).mag().max(1.0);
        let tangent = tangent_xy(offset, spec.spin);
        let speed = orbital_speed_for_spec(spec, planar_radius, g_scale);
        let planar_rad = Vec3::new(offset.x / planar_radius, offset.y / planar_radius, 0.0);
        let (wave_dv_r, wave_dv_t, arm_membership) = if arm_wave_amp > 1e-4 && !in_bar {
            let phase_r = spec.arm_twist * radial_norm;
            let mut psi = std::f32::consts::PI;
            for arm_idx in 0..spec.arm_count {
                let ridge = (arm_idx as f32 * std::f32::consts::TAU / spec.arm_count as f32 + phase_r)
                    .rem_euclid(std::f32::consts::TAU);
                let delta = shortest_angle_delta(base_angle, ridge);
                if delta.abs() < psi.abs() {
                    psi = delta;
                }
            }

            let t_in = ((radial_norm - 0.15) / 0.12).clamp(0.0, 1.0);
            let t_out = ((0.88 - radial_norm) / 0.10).clamp(0.0, 1.0);
            let taper_mix = t_in * t_out;
            let taper = taper_mix * taper_mix * (3.0 - 2.0 * taper_mix);
            let amp = arm_wave_amp * taper * speed;
            let crest_proximity = ((psi.cos() + 1.0) * 0.5).powi(2);
            let membership = (
                crest_proximity * spec.arm_contrast * spec.disc_population.arm_contrast_gain
            )
            .clamp(0.0, 1.0);
            (
                amp * psi.sin(),
                -amp * std::f32::consts::SQRT_2 * psi.cos(),
                membership,
            )
        } else {
            (0.0, 0.0, 0.0)
        };
        let velocity_noise = Vec3::new(noisy(rng), noisy(rng), noisy(rng) * 0.7)
            * speed
            * (0.028 + spec.irregularity * 0.08)
            * disc_dispersion(spec, arm_membership);
        let mass_t = radial_norm.powf(spec.disc_population.mass_falloff_exp.max(0.1));
        let radius_t = radial_norm.powf(spec.disc_population.radius_falloff_exp.max(0.1));
        let body_mass_factor = spec.disc_population.inner_mass_scale
            + (spec.disc_population.outer_mass_scale - spec.disc_population.inner_mass_scale) * mass_t;
        let body_radius_factor = spec.disc_population.inner_radius_scale
            + (spec.disc_population.outer_radius_scale - spec.disc_population.inner_radius_scale) * radius_t;
        let body_mass = mass_per_body * body_mass_factor.max(0.05);
        let mut body = Body::new_with_system(
            center + offset,
            tangent * (speed + wave_dv_t) + planar_rad * wave_dv_r + velocity_noise,
            body_mass,
            STAR_RADIUS * body_radius_factor.max(0.25),
            system_id,
        );
        let arm_boost = spec.disc_population.arm_contrast_gain.max(0.2).sqrt();
        body.arm_strength = (arm_membership * arm_boost).clamp(0.0, 1.0);
        built_disc_mass += body.mass;
        bodies.push(body);
    }

    if built_disc_mass > 0.0 {
        let mass_fix = target_disc_mass / built_disc_mass;
        for body in bodies[disc_start..].iter_mut() {
            body.mass *= mass_fix;
        }
    }
}

fn add_outer_binaries(
    bodies: &mut Vec<Body>,
    center: Vec3,
    system_id: u16,
    g_scale: f32,
    spec: &GalaxySpec,
    rng: &mut fastrand::Rng,
) {
    let visible_radius = world(spec.visible_radius_ly) * spec.disc_population.spatial_scale;
    let halo_radius = world(spec.halo_radius_ly) * spec.disc_population.spatial_scale;
    let pair_mass = (spec.halo_mass / spec.binary_pairs.max(1) as f32).max(0.8);

    for _ in 0..spec.binary_pairs {
        let angle = rng.f32() * std::f32::consts::TAU;
        let distance = visible_radius + (halo_radius - visible_radius) * rng.f32().powf(0.8);
        let halo_height = world(spec.disc_height_ly) * 3.0 + distance * 0.06;
        let pair_center = center
            + Vec3::new(
                angle.cos() * distance,
                angle.sin() * distance * spec.flattening.max(0.55),
                noisy(rng) * halo_height,
            );
        let com_offset = pair_center - center;
        let com_speed = orbital_speed_for_spec(spec, distance, g_scale) * 0.86;
        let pair_center_vel = tangent_xy(com_offset, spec.spin) * com_speed;
        let separation = (world(spec.visible_radius_ly) * 0.014).max(6.0) * (0.7 + rng.f32() * 0.9);
        push_binary_pair(
            bodies,
            pair_center,
            pair_center_vel,
            separation,
            rng.f32() * std::f32::consts::TAU,
            separation * 0.18,
            pair_mass * 2.0,
            pair_mass,
            STAR_RADIUS,
            system_id,
            spec.spin,
        );
    }
}

fn add_companion(
    bodies: &mut Vec<Body>,
    center: Vec3,
    system_id: u16,
    g_scale: f32,
    host: &GalaxySpec,
    companion: CompanionSpec,
    rng: &mut fastrand::Rng,
) {
    let offset = world(companion.offset_ly) * host.disc_population.spatial_scale;
    let radius = world(companion.radius_ly) * host.disc_population.spatial_scale;
    let angle = companion.orbit_phase;
    let companion_center = center + Vec3::new(angle.cos() * offset, angle.sin() * offset, offset * companion.orbit_tilt);
    let host_offset = companion_center - center;
    let host_speed = orbital_speed_for_spec(host, offset, g_scale) * companion.orbit_speed_scale;
    let companion_velocity = tangent_xy(host_offset, host.spin) * host_speed;
    let mass_per_body = (host.disc_mass * 0.18) / companion.bodies.max(1) as f32;

    for _ in 0..companion.bodies {
        let local_angle = rng.f32() * std::f32::consts::TAU;
        let (sin_a, cos_a) = local_angle.sin_cos();
        let local_radius = radius * rng.f32().powf(1.45);
        let local_offset = Vec3::new(
            cos_a * local_radius,
            sin_a * local_radius * 0.9,
            noisy(rng) * radius * 0.14,
        );
        let local_tangent = tangent_xy(local_offset, companion.spin);
        let local_speed = orbital_speed_for_spec(host, local_radius.max(3.0), g_scale) * 0.74;
        let mut body_a = Body::new_with_system(
            companion_center + local_offset,
            companion_velocity + local_tangent * local_speed,
            mass_per_body,
            STAR_RADIUS,
            system_id,
        );
        body_a.arm_strength = 0.0;
        bodies.push(body_a);
    }
}

fn normalize_template_system(bodies: &mut [Body], center: Vec3, spec: &GalaxySpec) {
    if bodies.is_empty() {
        return;
    }

    let targets = normalization_targets();

    let mut total_mass = 0.0;
    let mut pos_sum = Vec3::zero();
    let mut vel_sum = Vec3::zero();
    for body in bodies.iter() {
        total_mass += body.mass;
        pos_sum += body.pos * body.mass;
        vel_sum += body.vel * body.mass;
    }
    if total_mass <= 0.0 {
        return;
    }

    let com_pos = pos_sum / total_mass;
    let com_vel = vel_sum / total_mass;
    let center_shift = center - com_pos;

    for body in bodies.iter_mut() {
        body.pos += center_shift;
        body.vel -= com_vel;
    }

    let mut r2_mass = 0.0;
    let mut v2_mass = 0.0;
    for body in bodies.iter() {
        r2_mass += (body.pos - center).mag_sq() * body.mass;
        v2_mass += body.vel.mag_sq() * body.mass;
    }

    let r_rms = (r2_mass / total_mass).sqrt().max(world(spec.bulge_radius_ly) * 0.4).max(1.0);
    let v_rms = (v2_mass / total_mass).sqrt().max(1e-4);

    let target_total_mass = targets.target_mass_per_radius * r_rms;
    let mass_scale = (target_total_mass / total_mass).clamp(targets.mass_scale_min, targets.mass_scale_max);
    let mut velocity_scale = mass_scale.sqrt();

    let predicted_v_rms = (v_rms * velocity_scale).max(1e-4);
    let velocity_correction = (targets.target_v_rms / predicted_v_rms)
        .clamp(targets.vel_corr_min, targets.vel_corr_max);
    velocity_scale *= velocity_correction;

    for body in bodies.iter_mut() {
        body.mass *= mass_scale;
        body.vel *= velocity_scale;
    }

    let (dt_mass_scale, dt_speed_scale) = template_step_stability_scales();
    for body in bodies.iter_mut() {
        body.mass *= dt_mass_scale;
        body.vel *= dt_speed_scale;
    }

    stabilize_center_region(bodies, center, spec);
}

fn stabilize_center_region(bodies: &mut [Body], center: Vec3, spec: &GalaxySpec) {
    if bodies.is_empty() {
        return;
    }

    // Track one dominant central particle, but keep it physically free (no hard pinning).
    let mut anchor_idx = 0usize;
    let mut anchor_score = f32::MIN;
    for (idx, body) in bodies.iter().enumerate() {
        let d = (body.pos - center).mag();
        let score = body.mass / (1.0 + d);
        if score > anchor_score {
            anchor_score = score;
            anchor_idx = idx;
        }
    }

    let inner_radius = world(spec.bulge_radius_ly).max(6.0);
    let core_lock_radius = (inner_radius * 0.22).max(4.0);

    for (idx, body) in bodies.iter_mut().enumerate() {
        let is_anchor = idx == anchor_idx;
        if idx == anchor_idx {
            body.radius = body.radius.min(CORE_RADIUS_CAP).max(CORE_RADIUS * 0.75);
        }

        let rel = body.pos - center;
        let r = rel.mag();
        if r < 1e-4 {
            body.vel = Vec3::zero();
            continue;
        }

        let radial_dir = rel / r;
        let radial_v = body.vel.dot(radial_dir);
        let tangential = body.vel - radial_dir * radial_v;

        let enclosed_mass = mass_inside(spec, r).max(1.0);
        let v_circ = (enclosed_mass / r.max(1.0)).sqrt();
        let v_escape = (2.0 * enclosed_mass / r.max(1.0)).sqrt();

        if r <= inner_radius {
            // Suppress outward burst in the inner galaxy while preserving rotation.
            let radial_keep = if r <= core_lock_radius {
                if is_anchor { 0.16 } else { 0.06 }
            } else {
                0.20
            };
            let new_radial = radial_v * radial_keep;

            let tan_mag = tangential.mag();
            let tan_dir = if tan_mag > 1e-6 {
                tangential / tan_mag
            } else {
                tangent_xy(rel, spec.spin)
            };

            let blend = (1.0 - (r / inner_radius)).clamp(0.0, 1.0);
            let target_tan = v_circ * (0.92 + 0.05 * (1.0 - blend));
            let new_tan = tan_mag + (target_tan - tan_mag) * (0.65 * blend + 0.20);

            body.vel = tan_dir * new_tan.max(0.0) + radial_dir * new_radial;
        }

        // Enforce bounded starts for all particles.
        let speed = body.vel.mag();
        let vmax = if r <= inner_radius {
            if is_anchor { 0.97 * v_escape } else { 0.94 * v_escape }
        } else {
            0.98 * v_escape
        };
        if speed > vmax {
            body.vel *= vmax / speed;
        }
    }
}

fn apply_flat_rotation_with_epicycles(
    bodies: &mut [Body],
    center: Vec3,
    spin: f32,
    exclusion_radius: f32,
    profile: &DiscPopulationProfile,
) {
    if bodies.is_empty() {
        return;
    }

    let (_, dt_speed_scale) = template_step_stability_scales();

    // 220 in galactic units mapped into stable world-space speeds.
    let v_flat = (FLAT_ROTATION_SPEED_UNITS * FLAT_ROTATION_SPEED_SCALE * dt_speed_scale).max(1e-4);
    let spin_sign = if spin >= 0.0 { 1.0 } else { -1.0 };
    let exclusion_sq = exclusion_radius * exclusion_radius;
    let mut max_disc_r = exclusion_radius.max(1e-4);
    for body in bodies.iter() {
        let rel = body.pos - center;
        let r = (rel.x * rel.x + rel.y * rel.y).sqrt();
        if r > max_disc_r {
            max_disc_r = r;
        }
    }
    let outer_start = max_disc_r * profile.outer_osc_start_frac.clamp(0.15, 0.75);
    let outer_span = (max_disc_r - outer_start).max(1e-4);

    for body in bodies.iter_mut() {
        let rel = body.pos - center;
        let r_sq = rel.x * rel.x + rel.y * rel.y;
        if r_sq <= exclusion_sq {
            continue;
        }

        let r = r_sq.sqrt().max(1e-4);
        let radial_dir = Vec3::new(rel.x / r, rel.y / r, 0.0);
        let tangent_dir = Vec3::new(-radial_dir.y, radial_dir.x, 0.0) * spin_sign;
        let radial_norm = (r / max_disc_r.max(1e-4)).clamp(0.0, 1.0);
        let inner_boost = 1.0
            + (profile.inner_orbit_boost.max(1.0) - 1.0) * (1.0 - radial_norm).powf(1.5);
        let outer_boost = 1.0
            + (profile.outer_orbit_boost.max(1.0) - 1.0) * radial_norm.powf(1.1);
        let tangential_speed = v_flat * inner_boost * outer_boost;

        // Differential rotation with flat velocity curve: omega = v / r.
        let omega = tangential_speed / r;
        let phase = (rel.x * 0.0037
            + rel.y * 0.0059
            + body.mass * 0.13
            + body.system_id as f32 * 0.071)
            .rem_euclid(std::f32::consts::TAU);
        let amp_mix = (phase * 1.73).sin() * 0.5 + 0.5;
        let base_amp_frac = EPICYCLE_AMPLITUDE_MIN
            + (EPICYCLE_AMPLITUDE_MAX - EPICYCLE_AMPLITUDE_MIN) * amp_mix;
        // Smoothly strengthen epicycles in the outer two-thirds of the disc.
        let outer_t = ((r - outer_start) / outer_span).clamp(0.0, 1.0);
        let smooth_outer_t = outer_t * outer_t * (3.0 - 2.0 * outer_t);
        let outer_boost = 1.0
            + (profile.outer_osc_boost_max.max(1.0) - 1.0) * smooth_outer_t;
        let amp_frac = (base_amp_frac * outer_boost)
            .min(profile.outer_osc_amp_cap.max(EPICYCLE_AMPLITUDE_MAX));
        let omega_epi = EPICYCLE_FREQUENCY_FACTOR * omega;
        let radial_speed = amp_frac * r * omega_epi * phase.cos();

        body.vel = tangent_dir * tangential_speed
            + radial_dir * radial_speed
            + Vec3::new(0.0, 0.0, body.vel.z * 0.2);
    }

    // Remove residual center-of-mass drift after remapping velocities.
    let mut total_mass = 0.0f32;
    let mut com_vel = Vec3::zero();
    for body in bodies.iter() {
        total_mass += body.mass;
        com_vel += body.vel * body.mass;
    }
    if total_mass > 0.0 {
        com_vel /= total_mass;
        for body in bodies.iter_mut() {
            body.vel -= com_vel;
        }
    }
}

fn apply_minimal_physical_radii(bodies: &mut [Body]) {
    for body in bodies.iter_mut() {
        body.radius = body.radius.max(MIN_PHYSICAL_RADIUS * 0.5);
    }
}
