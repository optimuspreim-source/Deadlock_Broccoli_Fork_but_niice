use std::collections::HashMap;
use std::sync::atomic::AtomicU8;

use crate::{body::Body, quadtree::Octree};

use ultraviolet::Vec3;
use rayon::prelude::*;

use broccoli::aabb::ManySwappable;
use broccoli::Tree;
use broccoli_rayon::prelude::*;
use crossbeam::queue::SegQueue;

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

pub const BASE_DT: f32 = 0.08;
pub const BASE_GRAVITY_SCALE: f32 = 6.0;
pub const BASE_SOFTENING: f32 = 12.0;

pub static TEMPLATE_PROFILE_IDX: AtomicU8 = AtomicU8::new(0);

pub fn template_profile_label(idx: u8) -> &'static str {
    match idx {
        1 => "Dynamisch",
        _ => "Konservativ",
    }
}

const MAX_BODIES: usize = 220_000;
const ORBITAL_SPAWN_INTERVAL: usize = 3;
const ORBITAL_SPAWN_COUNT: usize = 4;
const ORBITAL_MASS_BONUS: f32 = 1.2;
const MERGE_MIN_AGE: u8 = 80;
const MAX_SPAWNS_PER_SYSTEM_PER_STEP: usize = 2;
const NET_GROWTH_FACTOR: f32 = 1.02;

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

/// Spawn `ORBITAL_SPAWN_COUNT` bodies in a circular orbit (in the XY plane)
/// around `center`, inheriting its bulk velocity plus Keplerian tangential speed.
fn spawn_orbiters(center: &Body) -> Vec<Body> {
    let orbit_radius = (center.radius * 5.0 + 5.0).max(10.0);
    (0..ORBITAL_SPAWN_COUNT)
        .map(|k| {
            let angle = k as f32 * std::f32::consts::TAU / ORBITAL_SPAWN_COUNT as f32;
            let (sin_a, cos_a) = angle.sin_cos();
            let pos = center.pos + Vec3::new(cos_a, sin_a, 0.0) * orbit_radius;
            let v = (center.mass / orbit_radius).sqrt();
            // Tangential direction counter-clockwise in XY plane
            let vel = center.vel + Vec3::new(-sin_a, cos_a, 0.0) * v;
            Body::new_with_system(pos, vel, ORBITAL_MASS_BONUS, 0.00264, center.system_id)
        })
        .collect()
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
    merge_counters: HashMap<u16, usize>,
}

impl Simulation {
    pub fn new() -> Self {
        let dt = BASE_DT;
        let theta = 1.0;
        let epsilon = BASE_SOFTENING;
        let leaf_capacity = 16;
        let thread_capacity = 1024;

        let octree = Octree::new(theta, epsilon, leaf_capacity, thread_capacity);

        Self {
            dt,
            gravity_scale: BASE_GRAVITY_SCALE,
            frame: 0,
            bodies: Vec::new(),
            octree,
            merge_counters: HashMap::new(),
        }
    }

    pub fn step(&mut self) {
        self.attract();
        self.iterate();

        // Collision detection every 2 frames for performance
        if self.frame % 2 == 0 {
            let new_bodies = self.collide();

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

    // -----------------------------------------------------------------------
    // Broad-phase: broccoli tree (parallel KD-tree + sweep-and-prune hybrid)
    // -----------------------------------------------------------------------

    fn collide(&mut self) -> Vec<Body> {
        let n = self.bodies.len();
        if n == 0 { return Vec::new(); }

        let available_slots = MAX_BODIES.saturating_sub(self.bodies.len());
        if available_slots == 0 {
            return Vec::new();
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
            Merge { winner: usize, loser: usize, new_pos: Vec3, new_vel: Vec3,
                    total_mass: f32, new_radius: f32, new_mc: u8, sys_id: u16 },
            None,
        }

        let bodies_ref = &self.bodies;
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
            if b1.age < MERGE_MIN_AGE || b2.age < MERGE_MIN_AGE { return compute_elastic(); }
            if !same_system { return compute_elastic(); }

            // Merge
            let (winner, loser) = if b1.mass >= b2.mass { (i, j) } else { (j, i) };
            ResolveAction::Merge {
                winner, loser,
                new_pos: (b1.pos * b1.mass + b2.pos * b2.mass) / total_mass,
                new_vel: (b1.vel * b1.mass + b2.vel * b2.mass) / total_mass,
                total_mass,
                new_radius: b1.radius.max(b2.radius) * 1.0003,
                new_mc: b1.merge_count.max(b2.merge_count).saturating_add(1),
                sys_id: b1.system_id,
            }
        }).collect();

        // Apply actions sequentially
        let mut new_bodies: Vec<Body> = Vec::new();
        let mut spawns_per_system: HashMap<u16, usize> = HashMap::new();
        let mut merges_this_step: usize = 0;
        let mut spawned_this_step: usize = 0;

        for action in actions {
            match action {
                ResolveAction::Elastic { i, j, impulse_i, impulse_j } => {
                    self.bodies[i].vel -= impulse_i;
                    self.bodies[j].vel += impulse_j;
                }
                ResolveAction::Merge { winner, loser, new_pos, new_vel, total_mass, new_radius, new_mc, sys_id } => {
                    if self.bodies[winner].mass <= 0.0 || self.bodies[loser].mass <= 0.0 { continue; }
                    let winner_arm = self.bodies[winner].arm_strength;
                    let loser_arm = self.bodies[loser].arm_strength;
                    self.bodies[winner].pos         = new_pos;
                    self.bodies[winner].vel         = new_vel;
                    self.bodies[winner].mass        = total_mass;
                    self.bodies[winner].radius      = new_radius;
                    self.bodies[winner].arm_strength = winner_arm.max(loser_arm) * 0.985;
                    self.bodies[winner].merge_count = new_mc;
                    self.bodies[loser].mass         = 0.0;
                    merges_this_step += 1;

                    let counter = self.merge_counters.entry(sys_id).or_insert(0);
                    *counter += 1;
                    if *counter % ORBITAL_SPAWN_INTERVAL == 0 {
                        if spawned_this_step >= available_slots {
                            continue;
                        }
                        let sys_spawns = spawns_per_system.entry(sys_id).or_insert(0);
                        let max_net = (merges_this_step as f32 * NET_GROWTH_FACTOR) as usize;
                        let remaining = available_slots.saturating_sub(spawned_this_step);
                        let sys_ok = MAX_SPAWNS_PER_SYSTEM_PER_STEP == 0
                            || *sys_spawns < MAX_SPAWNS_PER_SYSTEM_PER_STEP;
                        if sys_ok && spawned_this_step < max_net && remaining > 0 {
                            let merged = self.bodies[winner];
                            let mut orbiters = spawn_orbiters(&merged);
                            if orbiters.len() > remaining {
                                orbiters.truncate(remaining);
                            }
                            spawned_this_step += orbiters.len();
                            *sys_spawns += 1;
                            new_bodies.extend(orbiters);
                        }
                    }
                }
                ResolveAction::None => {}
            }
        }

        new_bodies
    }
}
