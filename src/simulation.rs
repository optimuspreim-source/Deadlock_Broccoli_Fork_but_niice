use std::collections::HashMap;
use std::sync::atomic::AtomicU8;

use crate::{body::Body, quadtree::Octree};
use crate::observer::BehaviorObserver;
use crate::utils_hierarchical::circular_speed;

use ultraviolet::Vec3;
use rayon::prelude::*;

use broccoli::aabb::ManySwappable;
use broccoli::Tree;
use broccoli_rayon::prelude::*;
use crossbeam::queue::SegQueue;

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

pub const BASE_DT: f32 = 0.065;
pub const BASE_GRAVITY_SCALE: f32 = 6.0;
pub const BASE_SOFTENING: f32 = 16.0;

pub static TEMPLATE_PROFILE_IDX: AtomicU8 = AtomicU8::new(0);

pub fn template_profile_label(idx: u8) -> &'static str {
    match idx {
        1 => "Dynamisch",
        _ => "Konservativ",
    }
}

const MAX_BODIES: usize = 220_000;
const BULGE_SAMPLE_DIVISOR: usize = 7;
const BULGE_SAMPLE_MIN: usize = 96;
const BULGE_RADIUS_FLOOR: f32 = 8.0;
const ANCHOR_CENTER_SPRING: f32 = 0.0016;
const ANCHOR_VEL_DAMPING: f32 = 0.14;

// ---------------------------------------------------------------------------
// Disk accretion: gas cooling onto the galactic plane → new star formation
// ---------------------------------------------------------------------------
/// Fraction of the central BH mass recycled per step into cooled disk inflow.
/// 0.0001% = 1e-6
const BH_FEEDBACK_FRACTION: f32 = 0.000001;
const BH_MIN_MASS: f32 = 1.0;
const BH_FEEDBACK_MIN_TOTAL_MASS: f32 = 0.00002;
const BH_FEEDBACK_MIN_PARTICLE_MASS: f32 = 0.00001;
const DISK_INNER_KEPLER_FRAC: f32 = 1.0 / 3.0;
const DISK_SPAWN_BASE_MASS: f32 = 1.2;
const DISK_SPAWN_BASE_RADIUS: f32 = 0.00264;
const DISK_SPAWN_MAX_PER_SYSTEM: usize = 12;
const DISK_SPAWN_MAX_GLOBAL: usize = 60;

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

// Lookup table: all 256 possible merge_count values pre-evaluated.
static GHOST_FACTOR_LUT: [f32; 256] = {
    let mut lut = [0.0f32; 256];
    let mut i = 0usize;
    while i < 256 {
        lut[i] = match i as u8 {
            0..=10  => 0.0,
            11..=20 => 2.0,
            21..=39 => 7.0,
            40..=69 => 29.0,
            70..=89 => 149.0,
            90..=99 => 59.0,
            _       => 499.0,
        };
        i += 1;
    }
    lut
};

const GHOST_GROWTH_RATE: f32 = 0.005;

fn vec3_dot(a: Vec3, b: Vec3) -> f32 {
    a.x * b.x + a.y * b.y + a.z * b.z
}

fn vec3_cross(a: Vec3, b: Vec3) -> Vec3 {
    Vec3::new(
        a.y * b.z - a.z * b.y,
        a.z * b.x - a.x * b.z,
        a.x * b.y - a.y * b.x,
    )
}

fn normalized_or(v: Vec3, fallback: Vec3) -> Vec3 {
    let m2 = v.mag_sq();
    if m2 > 1e-10 {
        v / m2.sqrt()
    } else {
        fallback
    }
}

fn disk_spawn_mass_for_radius(r_norm: f32) -> f32 {
    let x = r_norm.clamp(0.0, 1.0);
    // Calibrated three-zone profile:
    // - inner disk:   8.0 -> 4.0   (compact, massive population)
    // - mid disk:     4.0 -> 1.2   (mixed population)
    // - outer disk:   1.2 -> 0.35  (light outskirts)
    if x < 0.20 {
        let t = x / 0.20;
        8.0 + (4.0 - 8.0) * t
    } else if x < 0.65 {
        let t = (x - 0.20) / 0.45;
        4.0 + (1.2 - 4.0) * t
    } else {
        let t = (x - 0.65) / 0.35;
        1.2 + (0.35 - 1.2) * t
    }
}

fn disk_spawn_radius_for_mass(spawn_mass: f32, r_norm: f32) -> f32 {
    // Slightly softened scaling to avoid overly aggressive collision cascades.
    let mass_scale = (spawn_mass / DISK_SPAWN_BASE_MASS).max(0.2).powf(0.58);
    let central_boost = 1.0 + (1.0 - r_norm.clamp(0.0, 1.0)) * 0.22;
    DISK_SPAWN_BASE_RADIUS * mass_scale * central_boost
}


#[derive(Clone, Copy)]
struct SystemCollisionProfile {
    center: Vec3,
    bulge_radius: f32,
}

fn build_system_collision_profiles(bodies: &[Body]) -> HashMap<u16, SystemCollisionProfile> {
    let mut indices_by_system: HashMap<u16, Vec<usize>> = HashMap::new();
    for (index, body) in bodies.iter().enumerate() {
        if body.mass <= 0.0 || body.system_id == 0 {
            continue;
        }
        indices_by_system.entry(body.system_id).or_default().push(index);
    }

    let mut profiles = HashMap::with_capacity(indices_by_system.len());
    for (system_id, indices) in indices_by_system {
        if indices.is_empty() {
            continue;
        }

        let anchor_index = indices
            .iter()
            .copied()
            .max_by(|&lhs, &rhs| bodies[lhs]
                .mass
                .partial_cmp(&bodies[rhs].mass)
                .unwrap_or(std::cmp::Ordering::Equal))
            .unwrap_or(indices[0]);
        let center = bodies[anchor_index].pos;

        let mut distances: Vec<f32> = indices
            .iter()
            .map(|&index| (bodies[index].pos - center).mag())
            .filter(|distance| distance.is_finite())
            .collect();
        if distances.is_empty() {
            continue;
        }
        distances.sort_by(|lhs, rhs| lhs.partial_cmp(rhs).unwrap_or(std::cmp::Ordering::Equal));

        let sample_len = (distances.len() / BULGE_SAMPLE_DIVISOR)
            .max(BULGE_SAMPLE_MIN)
            .min(distances.len());
        let bulge_radius = distances[sample_len - 1].max(BULGE_RADIUS_FLOOR);

        profiles.insert(system_id, SystemCollisionProfile { center, bulge_radius });
    }

    profiles
}

fn bulges_touch(
    profiles: &HashMap<u16, SystemCollisionProfile>,
    lhs_system: u16,
    rhs_system: u16,
) -> bool {
    let Some(lhs) = profiles.get(&lhs_system) else {
        return false;
    };
    let Some(rhs) = profiles.get(&rhs_system) else {
        return false;
    };

    let max_touch_distance = lhs.bulge_radius + rhs.bulge_radius;
    (lhs.center - rhs.center).mag_sq() <= max_touch_distance * max_touch_distance
}

// ---------------------------------------------------------------------------
// Simulation
// ---------------------------------------------------------------------------

pub struct Simulation {
    pub dt: f32,
    pub gravity_scale: f32,
    pub frame: usize,
    pub bodies: Vec<Body>,
    pub octree: Octree,
    rng: fastrand::Rng,
    observer: BehaviorObserver,
}

#[derive(Clone, Copy, Default)]
struct CollisionStats {
    merges: usize,
    elastics: usize,
}

/// Per-system state gathered in a single pass over `bodies` for disk accretion.
#[derive(Default)]
struct DiskAccretionProfile {
    /// Indices into `bodies` that belong to this system (alive, system_id != 0).
    indices: Vec<usize>,
    anchor_idx: usize,
    anchor_mass: f32,
    mass_sum: f32,
    /// Mass-weighted position sum — divide by mass_sum to get COM.
    pos_mass_sum: Vec3,
    /// Mass-weighted velocity sum — divide by mass_sum to get bulk velocity.
    vel_mass_sum: Vec3,
}

#[derive(Clone, Copy)]
struct SystemAnchorStats {
    mass_sum: f32,
    pos_sum: Vec3,
    vel_sum: Vec3,
    anchor_idx: usize,
    anchor_mass: f32,
}

impl Simulation {
    pub fn new() -> Self {
        let dt = BASE_DT;
        let theta = 0.9;
        let epsilon = BASE_SOFTENING;
        let leaf_capacity = 24;
        let thread_capacity = 2048;

        let octree = Octree::new(theta, epsilon, leaf_capacity, thread_capacity);

        Self {
            dt,
            gravity_scale: BASE_GRAVITY_SCALE,
            frame: 0,
            bodies: Vec::new(),
            octree,
            rng: fastrand::Rng::with_seed(0xDEADC0DE),
            observer: BehaviorObserver::from_env(),
        }
    }

    pub fn step(&mut self) {
        self.attract();
        self.iterate();
        self.stabilize_system_anchors();
        let mut collision_stats = CollisionStats::default();

        // Collision detection every 2 frames for performance
        if self.frame % 2 == 0 {
            let (new_bodies, stats) = self.collide();
            collision_stats = stats;

            let mut i = 0;
            while i < self.bodies.len() {
                if self.bodies[i].mass <= 0.0 {
                    self.bodies.swap_remove(i);
                } else {
                    i += 1;
                }
            }

            for body in new_bodies {
                if self.bodies.len() < MAX_BODIES {
                    self.bodies.push(body);
                } else {
                    break;
                }
            }
        }

        // Disk accretion: continuous gas-cooling infall → new stars on the disk plane.
        let accreted = self.disk_accretion_spawn();
        for body in accreted {
            if self.bodies.len() < MAX_BODIES {
                self.bodies.push(body);
            } else {
                break;
            }
        }

        self.observer.maybe_record(
            self.frame,
            self.dt,
            &self.bodies,
            collision_stats.merges,
            collision_stats.elastics,
        );

        self.frame += 1;
    }

    pub fn attract(&mut self) {
        self.octree.build(&mut self.bodies);
        self.octree.acc(&mut self.bodies, self.gravity_scale);
    }

    pub fn iterate(&mut self) {
        let dt = self.dt;
        self.bodies.par_iter_mut().for_each(|body| {
            let target = body.mass * GHOST_FACTOR_LUT[body.merge_count as usize];
            body.ghost_mass += (target - body.ghost_mass) * GHOST_GROWTH_RATE;
            body.update(dt);
        });
    }

    fn stabilize_system_anchors(&mut self) {
        let mut systems: HashMap<u16, SystemAnchorStats> = HashMap::new();

        for (idx, body) in self.bodies.iter().enumerate() {
            if body.mass <= 0.0 || body.system_id == 0 {
                continue;
            }
            let entry = systems.entry(body.system_id).or_insert(SystemAnchorStats {
                mass_sum: 0.0,
                pos_sum: Vec3::zero(),
                vel_sum: Vec3::zero(),
                anchor_idx: idx,
                anchor_mass: body.mass,
            });
            entry.mass_sum += body.mass;
            entry.pos_sum += body.pos * body.mass;
            entry.vel_sum += body.vel * body.mass;
            if body.mass > entry.anchor_mass {
                entry.anchor_mass = body.mass;
                entry.anchor_idx = idx;
            }
        }

        let dt = self.dt;
        for (_, stats) in systems {
            if stats.mass_sum <= 0.0 {
                continue;
            }
            let com_pos = stats.pos_sum / stats.mass_sum;
            let com_vel = stats.vel_sum / stats.mass_sum;

            let anchor = &mut self.bodies[stats.anchor_idx];
            let pos_offset = anchor.pos - com_pos;
            anchor.vel -= pos_offset * (ANCHOR_CENTER_SPRING * dt);
            anchor.vel -= (anchor.vel - com_vel) * ANCHOR_VEL_DAMPING;
        }
    }

    // -----------------------------------------------------------------------
    // Broad-phase: broccoli tree (parallel KD-tree + sweep-and-prune hybrid)
    // -----------------------------------------------------------------------

    fn collide(&mut self) -> (Vec<Body>, CollisionStats) {
        let n = self.bodies.len();
        if n == 0 {
            return (Vec::new(), CollisionStats::default());
        }

        let system_profiles = build_system_collision_profiles(&self.bodies);

        // Central BH proxy: the most massive body in each system.
        let mut system_anchor_idx: HashMap<u16, usize> = HashMap::new();
        let mut system_anchor_mass: HashMap<u16, f32> = HashMap::new();
        for (idx, body) in self.bodies.iter().enumerate() {
            if body.mass <= 0.0 || body.system_id == 0 {
                continue;
            }
            let current = system_anchor_mass.get(&body.system_id).copied().unwrap_or(-1.0);
            if body.mass > current {
                system_anchor_mass.insert(body.system_id, body.mass);
                system_anchor_idx.insert(body.system_id, idx);
            }
        }

        // Build 2D AABBs (XY projection) for each body, storing the body index.
        let mut bots: Vec<ManySwappable<broccoli::aabb::BBox<f32, usize>>> = self.bodies
            .iter()
            .enumerate()
            .filter(|(_, b)| b.mass > 0.0)
            .map(|(i, b)| {
                let r = b.radius;
                ManySwappable(broccoli::bbox(
                    broccoli::rect(b.pos.x - r, b.pos.x + r, b.pos.y - r, b.pos.y + r),
                    i,
                ))
            })
            .collect();

        // Collect pairs using a lock-free queue to reduce callback contention.
        let pairs_queue: SegQueue<(usize, usize)> = SegQueue::new();

        {
            let mut tree = Tree::new(&mut bots);
            let bodies_ref = &self.bodies;
            tree.par_find_colliding_pairs(|a, b| {
                let ia = *a.unpack_inner();
                let ib = *b.unpack_inner();
                // Full 3D distance check
                let ba = &bodies_ref[ia];
                let bb = &bodies_ref[ib];
                let d = bb.pos - ba.pos;
                let r_sum = ba.radius + bb.radius;
                if d.mag_sq() < r_sum * r_sum {
                    pairs_queue.push((ia, ib));
                }
            });
        }

        let mut pairs = Vec::new();
        while let Some(pair) = pairs_queue.pop() {
            pairs.push(pair);
        }

        // ---------------------------------------------------------------
        // Two-Phase Resolve: classify pairs in parallel, apply sequentially
        // ---------------------------------------------------------------

        enum ResolveAction {
            Elastic { i: usize, j: usize, impulse_i: Vec3, impulse_j: Vec3 },
            Accrete { bh: usize, victim: usize },
            Merge { winner: usize, loser: usize, new_pos: Vec3, new_vel: Vec3,
                    total_mass: f32, new_radius: f32, new_mc: u8 },
            None,
        }

        let bodies_ref = &self.bodies;
        let profiles_ref = &system_profiles;
        let anchors_ref = &system_anchor_idx;
        let actions: Vec<ResolveAction> = pairs.par_iter().map(|&(i, j)| {
            let b1 = bodies_ref[i];
            let b2 = bodies_ref[j];
            if b1.mass <= 0.0 || b2.mass <= 0.0 { return ResolveAction::None; }

            let d = b2.pos - b1.pos;
            let dist = d.mag();
            let r_sum = b1.radius + b2.radius;
            if dist >= r_sum { return ResolveAction::None; }

            let normal = if dist < 1e-6 { Vec3::new(1.0, 0.0, 0.0) } else { d / dist };
            let total_mass = b1.mass + b2.mass;
            let v_rel_vec = b2.vel - b1.vel;
            let v_rel = v_rel_vec.mag();
            let v_escape = (2.0 * total_mass / r_sum.max(1e-4)).sqrt();
            let same_system = b1.system_id == b2.system_id;
            let cross_system_bulge_touch = !same_system && bulges_touch(profiles_ref, b1.system_id, b2.system_id);
            let merge_allowed = same_system || cross_system_bulge_touch;

            let compute_elastic = || -> ResolveAction {
                let v_approach = -(v_rel_vec.dot(normal));
                if v_approach <= 0.0 { return ResolveAction::None; }
                let impulse = 2.0 * b1.mass * b2.mass / total_mass * v_approach;
                ResolveAction::Elastic {
                    i, j,
                    impulse_i: normal * (impulse / b1.mass),
                    impulse_j: normal * (impulse / b2.mass),
                }
            };

            if v_rel >= v_escape { return compute_elastic(); }
            if !merge_allowed { return compute_elastic(); }

            // BH accretion sink: if exactly one participant is the central anchor,
            // absorb the other body without increasing BH mass.
            let is_anchor_b1 = b1.system_id != 0 && anchors_ref.get(&b1.system_id).copied() == Some(i);
            let is_anchor_b2 = b2.system_id != 0 && anchors_ref.get(&b2.system_id).copied() == Some(j);
            if is_anchor_b1 ^ is_anchor_b2 {
                let (bh, victim) = if is_anchor_b1 { (i, j) } else { (j, i) };
                return ResolveAction::Accrete { bh, victim };
            }

            // Merge
            let (winner, loser) = if b1.mass >= b2.mass { (i, j) } else { (j, i) };
            let new_pos = (b1.pos * b1.mass + b2.pos * b2.mass) / total_mass;
            ResolveAction::Merge {
                winner, loser,
                new_pos,
                new_vel: (b1.vel * b1.mass + b2.vel * b2.mass) / total_mass,
                total_mass,
                new_radius: (b1.radius.powi(3) + b2.radius.powi(3)).cbrt(),
                new_mc: b1.merge_count.max(b2.merge_count).saturating_add(1),
            }
        }).collect();

        // Apply actions sequentially
        let new_bodies: Vec<Body> = Vec::new();
        let mut merges_this_step: usize = 0;
        let mut elastics_this_step: usize = 0;

        for action in actions {
            match action {
                ResolveAction::Elastic { i, j, impulse_i, impulse_j } => {
                    self.bodies[i].vel -= impulse_i;
                    self.bodies[j].vel += impulse_j;
                    elastics_this_step += 1;
                }
                ResolveAction::Accrete { bh, victim } => {
                    if self.bodies[bh].mass <= 0.0 || self.bodies[victim].mass <= 0.0 { continue; }
                    self.bodies[victim].mass = 0.0;
                    merges_this_step += 1;
                }
                ResolveAction::Merge {
                    winner,
                    loser,
                    new_pos,
                    new_vel,
                    total_mass,
                    new_radius,
                    new_mc,
                } => {
                    if self.bodies[winner].mass <= 0.0 || self.bodies[loser].mass <= 0.0 { continue; }
                    let winner_arm = self.bodies[winner].arm_strength;
                    let loser_arm  = self.bodies[loser].arm_strength;
                    self.bodies[winner].pos          = new_pos;
                    self.bodies[winner].vel          = new_vel;
                    self.bodies[winner].mass         = total_mass;
                    self.bodies[winner].radius       = new_radius;
                    self.bodies[winner].arm_strength =
                        (winner_arm * self.bodies[winner].mass + loser_arm * self.bodies[loser].mass)
                        / total_mass;
                    self.bodies[winner].merge_count  = new_mc;
                    self.bodies[loser].mass          = 0.0;
                    merges_this_step += 1;
                }
                ResolveAction::None => {}
            }
        }

        (
            new_bodies,
            CollisionStats {
                merges: merges_this_step,
                elastics: elastics_this_step,
            },
        )
    }

    // -----------------------------------------------------------------------
    // Disk accretion: density-weighted star-formation on the galactic plane
    // -----------------------------------------------------------------------

    /// Spawn new bodies on the galactic disk plane (z ≈ 0) to simulate the
    /// continuous infall of cooling gas from the halo.
    ///
    /// Mechanism:
    ///  1. For each system, gather a radial sample by **reservoir-sampling** an
    ///     existing body's XY distance from the COM.  Because denser regions
    ///     contain more bodies they are sampled proportionally more often →
    ///     the spawn distribution automatically mirrors the current surface
    ///     density and reinforces spiral-arm overdensities.
    ///  2. A Keplerian tangential velocity derived from the enclosed mass at
    ///     that radius and aligned with the system's current spin direction
    ///     ensures newly-born bodies immediately settle into stable orbits.
    ///  3. Spawns are placed on the disk plane (z = COM.z).
    fn disk_accretion_spawn(&mut self) -> Vec<Body> {
        let available = MAX_BODIES.saturating_sub(self.bodies.len());
        if available == 0 {
            return Vec::new();
        }

        // --- single pass: build per-system profiles -----------------------
        let mut by_system: HashMap<u16, DiskAccretionProfile> = HashMap::new();
        for (idx, body) in self.bodies.iter().enumerate() {
            if body.mass <= 0.0 || body.system_id == 0 {
                continue;
            }
            let entry = by_system.entry(body.system_id).or_default();
            entry.indices.push(idx);
            if body.mass > entry.anchor_mass {
                entry.anchor_mass = body.mass;
                entry.anchor_idx = idx;
            }
            entry.mass_sum     += body.mass;
            entry.pos_mass_sum += body.pos * body.mass;
            entry.vel_mass_sum += body.vel * body.mass;
        }

        // --- spawn loop ----------------------------------------------------
        let mut result = Vec::new();
        let mut global_spawned = 0usize;
        let g_scale = self.gravity_scale;

        for (sys_id, profile) in &by_system {
            let remaining_global = DISK_SPAWN_MAX_GLOBAL
                .min(available)
                .saturating_sub(global_spawned);
            if remaining_global == 0 {
                break;
            }
            // Need at least a few reference bodies for a meaningful radial sample.
            if profile.indices.len() < 4 {
                continue;
            }

            let com_pos = profile.pos_mass_sum / profile.mass_sum;
            let com_vel = profile.vel_mass_sum / profile.mass_sum;

            // Build cumulative enclosed-mass profile M(<r) in the actual disk plane
            // and infer disk normal from total angular momentum.
            let mut angular_momentum = Vec3::zero();
            for &idx in &profile.indices {
                let body = self.bodies[idx];
                let rel_pos = body.pos - com_pos;
                let rel_vel = body.vel - com_vel;
                angular_momentum += vec3_cross(rel_pos, rel_vel) * body.mass;
            }

            let disk_normal = {
                let n = normalized_or(angular_momentum, Vec3::new(0.0, 0.0, 1.0));
                if n.mag_sq() > 1e-10 {
                    n
                } else {
                    Vec3::new(0.0, 0.0, 1.0)
                }
            };

            // Stable orthonormal basis (u, v) spanning the inclined disk plane.
            let basis_seed = if disk_normal.z.abs() < 0.95 {
                Vec3::new(0.0, 0.0, 1.0)
            } else {
                Vec3::new(1.0, 0.0, 0.0)
            };
            let basis_u = normalized_or(vec3_cross(disk_normal, basis_seed), Vec3::new(1.0, 0.0, 0.0));
            let basis_v = normalized_or(vec3_cross(disk_normal, basis_u), Vec3::new(0.0, 1.0, 0.0));

            let mut radial_mass: Vec<(f32, f32)> = Vec::with_capacity(profile.indices.len());
            for &idx in &profile.indices {
                let body = self.bodies[idx];
                let rel_pos = body.pos - com_pos;
                let in_plane = rel_pos - disk_normal * vec3_dot(rel_pos, disk_normal);
                let r_plane = in_plane.mag();
                if r_plane.is_finite() {
                    radial_mass.push((r_plane.max(1.0), body.mass.max(0.0)));
                }
            }
            if radial_mass.is_empty() {
                continue;
            }
            radial_mass.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));
            let mut enclosed_mass = 0.0f32;
            for entry in &mut radial_mass {
                enclosed_mass += entry.1;
                entry.1 = enclosed_mass;
            }

            let max_radius = radial_mass.last().map(|(r, _)| *r).unwrap_or(1.0).max(1.0);
            let inner_kepler_limit = max_radius * DISK_INNER_KEPLER_FRAC;
            let outer_span = (max_radius - inner_kepler_limit).max(1e-4);

            // Feedback budget: tiny fraction of BH mass re-rains onto the disk.
            let bh_idx = profile.anchor_idx;
            let bh_mass = self.bodies[bh_idx].mass;
            if bh_mass <= BH_MIN_MASS {
                continue;
            }
            let max_extractable = (bh_mass - BH_MIN_MASS).max(0.0);
            let feedback_budget = (bh_mass * BH_FEEDBACK_FRACTION).min(max_extractable);
            if feedback_budget < BH_FEEDBACK_MIN_TOTAL_MASS {
                continue;
            }

            let target = DISK_SPAWN_MAX_PER_SYSTEM
                .min(DISK_SPAWN_MAX_PER_SYSTEM)
                .min(remaining_global)
                .min(available.saturating_sub(result.len()));
            if target == 0 {
                continue;
            }

            let mut remaining_budget = feedback_budget;
            let mut spawned_this_system = 0usize;

            for _ in 0..target {
                if remaining_budget < BH_FEEDBACK_MIN_PARTICLE_MASS {
                    break;
                }
                // Sample a random existing body → its XY radius becomes the
                // spawn radius.  This is O(1) and density-weighted by construction.
                let ref_idx = profile.indices[self.rng.usize(..profile.indices.len())];
                let ref_body = self.bodies[ref_idx];
                let ref_rel_pos = ref_body.pos - com_pos;
                let ref_rel_vel = ref_body.vel - com_vel;

                let ref_in_plane = ref_rel_pos - disk_normal * vec3_dot(ref_rel_pos, disk_normal);
                let r_plane = ref_in_plane.mag().max(5.0);
                let r_norm = (r_plane / max_radius).clamp(0.0, 1.0);
                let nominal_mass = disk_spawn_mass_for_radius(r_norm);
                let spawn_mass = nominal_mass.min(remaining_budget);
                if spawn_mass < BH_FEEDBACK_MIN_PARTICLE_MASS {
                    break;
                }
                let spawn_radius = disk_spawn_radius_for_mass(spawn_mass, r_norm);

                // Random azimuthal angle — new body lands anywhere on the ring.
                let angle      = self.rng.f32() * std::f32::consts::TAU;
                let (sin_a, cos_a) = angle.sin_cos();

                let radial_dir = normalized_or(basis_u * cos_a + basis_v * sin_a, basis_u);
                let tangent_dir = normalized_or(vec3_cross(disk_normal, radial_dir), basis_v);
                let spawn_pos = com_pos + radial_dir * r_plane;

                let idx = radial_mass.partition_point(|(radius, _)| *radius <= r_plane);
                let m_enclosed = if idx == 0 {
                    radial_mass[0].1.max(spawn_mass)
                } else {
                    radial_mass[idx - 1].1.max(spawn_mass)
                };

                let v_kepler = circular_speed(m_enclosed, r_plane, g_scale, 1.0);
                let outer_t = ((r_plane - inner_kepler_limit) / outer_span).clamp(0.0, 1.0);
                let smooth_outer_t = outer_t * outer_t * (3.0 - 2.0 * outer_t);

                let local_tangent_speed = vec3_dot(ref_rel_vel, tangent_dir).abs();
                let tangential_speed = if r_plane <= inner_kepler_limit {
                    v_kepler
                } else {
                    let oscillation_tangent = (local_tangent_speed * 0.92).max(v_kepler * 0.58);
                    v_kepler * (1.0 - smooth_outer_t) + oscillation_tangent * smooth_outer_t
                };

                let local_radial_speed = vec3_dot(ref_rel_vel, radial_dir);
                let radial_osc_speed = local_radial_speed.clamp(-v_kepler * 0.42, v_kepler * 0.42)
                    * smooth_outer_t;

                let spawn_vel = com_vel
                    + tangent_dir * tangential_speed
                    + radial_dir * radial_osc_speed;

                result.push(Body::new_with_system(
                    spawn_pos, spawn_vel,
                    spawn_mass, spawn_radius,
                    *sys_id,
                ));
                remaining_budget -= spawn_mass;
                spawned_this_system += 1;
            }

            let used_budget = feedback_budget - remaining_budget;
            if used_budget > 0.0 {
                self.bodies[bh_idx].mass = (self.bodies[bh_idx].mass - used_budget).max(BH_MIN_MASS);
                global_spawned += spawned_this_system;
            }
        }

        result
    }
}
