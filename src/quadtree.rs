use std::{
    ops::Range,
    sync::atomic::{AtomicBool, AtomicI32, AtomicUsize, Ordering},
};

use crate::{body::Body, partition::Partition};
use ultraviolet::Vec3;

use rayon::prelude::*;

pub static SHIELD_AGE_CTRL: AtomicI32 = AtomicI32::new(12);
pub static BLEND_AGE_CTRL: AtomicI32 = AtomicI32::new(28);
/// false = sicher (A), true = paralleler unsafe-Pfad (B)
pub static OCTREE_BUILD_PARALLEL: AtomicBool = AtomicBool::new(false);

// ---------------------------------------------------------------------------
// Cube – axis-aligned bounding cube for a node in the octree
// ---------------------------------------------------------------------------

#[derive(Clone, Copy)]
pub struct Cube {
    pub center: Vec3,
    pub size: f32,
}

impl Cube {
    pub fn new_containing(bodies: &[Body]) -> Self {
        if bodies.is_empty() {
            return Self { center: Vec3::zero(), size: 1.0 };
        }
        let mut min = Vec3::broadcast(f32::MAX);
        let mut max = Vec3::broadcast(f32::MIN);
        for body in bodies {
            min = min.min_by_component(body.pos);
            max = max.max_by_component(body.pos);
        }
        let center = (min + max) * 0.5;
        let size = (max.x - min.x).max(max.y - min.y).max(max.z - min.z);
        Self { center, size }
    }

    /// Returns the child cube for the given octant index.
    /// Bit layout: bit0 = x-positive, bit1 = y-positive, bit2 = z-positive.
    pub fn into_octant(mut self, octant: usize) -> Self {
        self.size *= 0.5;
        self.center.x += ((octant & 1) as f32 - 0.5) * self.size;
        self.center.y += (((octant >> 1) & 1) as f32 - 0.5) * self.size;
        self.center.z += (((octant >> 2) & 1) as f32 - 0.5) * self.size;
        self
    }

    pub fn subdivide(&self) -> [Cube; 8] {
        [0, 1, 2, 3, 4, 5, 6, 7].map(|i| self.into_octant(i))
    }
}

// ---------------------------------------------------------------------------
// Node
// ---------------------------------------------------------------------------

#[derive(Clone)]
pub struct Node {
    pub children: usize,
    pub next: usize,
    pub pos: Vec3,
    pub mass: f32,
    pub cube: Cube,
    pub bodies: Range<usize>,
}

#[derive(Clone, Copy)]
pub struct GravSample {
    pub pos: Vec3,
    pub radius: f32,
    pub grav_mass: f32,
}

impl Node {
    pub const ZEROED: Self = Self {
        children: 0,
        next: 0,
        pos: Vec3 { x: 0.0, y: 0.0, z: 0.0 },
        mass: 0.0,
        cube: Cube {
            center: Vec3 { x: 0.0, y: 0.0, z: 0.0 },
            size: 0.0,
        },
        bodies: 0..0,
    };

    pub fn new(next: usize, cube: Cube, bodies: Range<usize>) -> Self {
        Self {
            children: 0,
            next,
            pos: Vec3 { x: 0.0, y: 0.0, z: 0.0 },
            mass: 0.0,
            cube,
            bodies,
        }
    }

    pub fn is_leaf(&self) -> bool { self.children == 0 }
    pub fn is_branch(&self) -> bool { self.children != 0 }
    pub fn is_empty(&self) -> bool { self.mass == 0.0 }
}

// ---------------------------------------------------------------------------
// Octree
// ---------------------------------------------------------------------------

pub struct Octree {
    pub t_sq: f32,
    pub e_sq: f32,
    pub leaf_capacity: usize,
    pub thread_capacity: usize,
    pub atomic_len: AtomicUsize,
    pub nodes: Vec<Node>,
    pub parents: Vec<usize>,
}

impl Octree {
    pub const ROOT: usize = 0;

    pub fn new(theta: f32, epsilon: f32, leaf_capacity: usize, thread_capacity: usize) -> Self {
        Self {
            t_sq: theta * theta,
            e_sq: epsilon * epsilon,
            leaf_capacity,
            thread_capacity,
            atomic_len: 0.into(),
            nodes: Vec::new(),
            parents: Vec::new(),
        }
    }

    pub fn clear(&mut self) {
        self.atomic_len.store(0, Ordering::Relaxed);
    }

    /// Partitions `bodies[range]` into 8 octants and creates 8 child nodes.
    /// Returns the index of the first child (children occupy children..children+8).
    pub fn subdivide(&mut self, node: usize, bodies: &mut [Body], range: Range<usize>) -> usize {
        let center = self.nodes[node].cube.center;

        // Partition bodies into 8 octants using 3 axis splits.
        // Octant index bit layout: bit0=x≥cx, bit1=y≥cy, bit2=z≥cz
        let mut s = [0usize; 9];
        s[0] = range.start;
        s[8] = range.end;

        // Split all on z-axis
        let predz = |b: &Body| b.pos.z < center.z;
        s[4] = s[0] + bodies[s[0]..s[8]].partition(predz);

        // Split each z-half on y-axis
        let predy = |b: &Body| b.pos.y < center.y;
        s[2] = s[0] + bodies[s[0]..s[4]].partition(predy);
        s[6] = s[4] + bodies[s[4]..s[8]].partition(predy);

        // Split each y-sub-half on x-axis
        let predx = |b: &Body| b.pos.x < center.x;
        s[1] = s[0] + bodies[s[0]..s[2]].partition(predx);
        s[3] = s[2] + bodies[s[2]..s[4]].partition(predx);
        s[5] = s[4] + bodies[s[4]..s[6]].partition(predx);
        s[7] = s[6] + bodies[s[6]..s[8]].partition(predx);

        let len = self.atomic_len.fetch_add(1, Ordering::Relaxed);
        let children = len * 8 + 1;
        self.parents[len] = node;
        self.nodes[node].children = children;

        let nexts = [
            children + 1, children + 2, children + 3, children + 4,
            children + 5, children + 6, children + 7,
            self.nodes[node].next,
        ];
        let cubes = self.nodes[node].cube.subdivide();
        for i in 0..8 {
            self.nodes[children + i] = Node::new(nexts[i], cubes[i], s[i]..s[i + 1]);
        }

        children
    }

    pub fn propagate(&mut self) {
        let len = self.atomic_len.load(Ordering::Relaxed);
        for &node in self.parents[..len].iter().rev() {
            let i = self.nodes[node].children;
            let mut pos_sum = Vec3 { x: 0.0, y: 0.0, z: 0.0 };
            let mut mass_sum = 0.0f32;
            for k in 0..8 {
                pos_sum += self.nodes[i + k].pos;
                mass_sum += self.nodes[i + k].mass;
            }
            self.nodes[node].pos = pos_sum;
            self.nodes[node].mass = mass_sum;
        }
        let max_idx = (len * 8 + 1).min(self.nodes.len());
        self.nodes[0..max_idx].par_iter_mut().for_each(|node| {
            node.pos /= node.mass.max(f32::MIN_POSITIVE);
        });
    }

    pub fn build(&mut self, bodies: &mut [Body]) {
        if OCTREE_BUILD_PARALLEL.load(Ordering::Relaxed) {
            self.build_parallel_unsafe(bodies);
        } else {
            self.build_safe(bodies);
        }

        #[cfg(debug_assertions)]
        self.debug_ab_check_against_unsafe(bodies);
    }

    fn init_build_storage(&mut self, bodies: &[Body]) -> bool {
        self.clear();

        if bodies.is_empty() {
            return false;
        }

        let new_len = 8 * bodies.len() + 4096;
        if self.nodes.len() < new_len {
            self.nodes.resize(new_len, Node::ZEROED);
            self.parents.resize(new_len / 8 + 1, 0);
        }

        let cube = Cube::new_containing(bodies);
        self.nodes[Self::ROOT] = Node::new(0, cube, 0..bodies.len());
        true
    }

    fn build_safe(&mut self, bodies: &mut [Body]) {
        if !self.init_build_storage(bodies) {
            return;
        }

        // Safe depth-first traversal with identical leaf/subdivide criteria.
        let mut stack = vec![Self::ROOT];
        while let Some(node) = stack.pop() {
            let range = self.nodes[node].bodies.clone();
            if range.len() <= self.leaf_capacity || self.nodes[node].cube.size < 1e-4 {
                self.nodes[node].pos = bodies[range.clone()]
                    .iter()
                    .fold(Vec3 { x: 0.0, y: 0.0, z: 0.0 }, |acc, b| {
                        acc + b.pos * (b.mass + b.ghost_mass)
                    });
                self.nodes[node].mass = bodies[range.clone()]
                    .iter()
                    .map(|b| b.mass + b.ghost_mass)
                    .sum();
                continue;
            }

            let children = self.subdivide(node, bodies, range);
            for i in 0..8 {
                if !self.nodes[children + i].bodies.is_empty() {
                    stack.push(children + i);
                }
            }
        }

        self.propagate();
    }

    fn build_parallel_unsafe(&mut self, bodies: &mut [Body]) {
        if !self.init_build_storage(bodies) {
            return;
        }

        let (tx, rx) = crossbeam::channel::unbounded();
        tx.send(Self::ROOT).unwrap();

        let octree_ptr = self as *mut Octree as usize;
        let bodies_ptr = bodies.as_ptr() as usize;
        let bodies_len = bodies.len();

        let counter = AtomicUsize::new(0);
        rayon::broadcast(|_| {
            let mut stack = Vec::new();
            let octree = unsafe { &mut *(octree_ptr as *mut Octree) };
            let bodies =
                unsafe { std::slice::from_raw_parts_mut(bodies_ptr as *mut Body, bodies_len) };

            while counter.load(Ordering::Relaxed) != bodies.len() {
                while let Ok(node) = rx.try_recv() {
                    let range = octree.nodes[node].bodies.clone();
                    let len = octree.nodes[node].bodies.len();

                    if range.len() >= octree.thread_capacity && octree.nodes[node].cube.size >= 1e-4 {
                        let children = octree.subdivide(node, bodies, range);
                        for i in 0..8 {
                            if !octree.nodes[children + i].bodies.is_empty() {
                                tx.send(children + i).unwrap();
                            }
                        }
                        continue;
                    }

                    counter.fetch_add(len, Ordering::Relaxed);

                    stack.push(node);
                    while let Some(node) = stack.pop() {
                        let range = octree.nodes[node].bodies.clone();
                        if range.len() <= octree.leaf_capacity || octree.nodes[node].cube.size < 1e-4 {
                            octree.nodes[node].pos = bodies[range.clone()]
                                .iter()
                                .fold(Vec3 { x: 0.0, y: 0.0, z: 0.0 }, |acc, b| {
                                    acc + b.pos * (b.mass + b.ghost_mass)
                                });
                            octree.nodes[node].mass = bodies[range.clone()]
                                .iter()
                                .map(|b| b.mass + b.ghost_mass)
                                .sum();
                            continue;
                        }
                        let children = octree.subdivide(node, bodies, range);
                        for i in 0..8 {
                            if !octree.nodes[children + i].bodies.is_empty() {
                                stack.push(children + i);
                            }
                        }
                    }
                }
            }
        });

        self.propagate();
    }

    #[cfg(debug_assertions)]
    fn debug_ab_check_against_unsafe(&self, bodies: &[Body]) {
        if std::env::var_os("BARNES_HUT_OCTREE_AB_CHECK").is_none() || bodies.is_empty() {
            return;
        }

        let theta = self.t_sq.sqrt();
        let epsilon = self.e_sq.sqrt();
        let mut ref_tree = Octree::new(theta, epsilon, self.leaf_capacity, self.thread_capacity);
        let mut ref_bodies = bodies.to_vec();
        ref_tree.build_parallel_unsafe(&mut ref_bodies);

        let self_root = &self.nodes[Self::ROOT];
        let ref_root = &ref_tree.nodes[Self::ROOT];
        let root_mass_delta = (self_root.mass - ref_root.mass).abs();
        let root_mass_scale = self_root.mass.abs().max(ref_root.mass.abs()).max(1.0);
        assert!(
            root_mass_delta <= root_mass_scale * 1e-4,
            "AB-check failed: root mass mismatch safe={} unsafe={} delta={}",
            self_root.mass,
            ref_root.mass,
            root_mass_delta
        );

        let root_pos_delta = (self_root.pos - ref_root.pos).mag();
        let root_pos_scale = self_root.pos.mag().max(ref_root.pos.mag()).max(1.0);
        assert!(
            root_pos_delta <= root_pos_scale * 1e-3,
            "AB-check failed: root COM mismatch safe={:?} unsafe={:?} delta={}",
            self_root.pos,
            ref_root.pos,
            root_pos_delta
        );

        let safe_samples: Vec<GravSample> = bodies
            .iter()
            .map(|b| GravSample {
                pos: b.pos,
                radius: b.radius,
                grav_mass: b.mass + b.ghost_mass,
            })
            .collect();
        let ref_samples: Vec<GravSample> = ref_bodies
            .iter()
            .map(|b| GravSample {
                pos: b.pos,
                radius: b.radius,
                grav_mass: b.mass + b.ghost_mass,
            })
            .collect();

        let sample_count = bodies.len().min(64);
        for s in 0..sample_count {
            let idx = s * bodies.len() / sample_count;
            let pos = bodies[idx].pos;
            let a_safe = self.acc_pos(pos, &safe_samples);
            let a_ref = ref_tree.acc_pos(pos, &ref_samples);
            let delta = (a_safe - a_ref).mag();
            let scale = a_safe.mag().max(a_ref.mag()).max(1.0);
            assert!(
                delta <= scale * 5e-3,
                "AB-check failed: acc mismatch idx={} safe={:?} unsafe={:?} delta={}",
                idx,
                a_safe,
                a_ref,
                delta
            );
        }
    }

    pub fn acc_pos(&self, pos: Vec3, bodies: &[GravSample]) -> Vec3 {
        let mut acc = Vec3 { x: 0.0, y: 0.0, z: 0.0 };

        let mut node = Self::ROOT;
        loop {
            let n = &self.nodes[node];
            let n_size  = n.cube.size;
            let n_pos   = n.pos;
            let n_mass  = n.mass;
            let n_next  = n.next;
            let n_leaf  = n.is_leaf();
            let n_child = n.children;
            let n_bodies = n.bodies.start..n.bodies.end;

            let d = n_pos - pos;
            let d_sq = d.mag_sq();

            if n_size * n_size < d_sq * self.t_sq {
                let denom = (d_sq + self.e_sq) * d_sq.sqrt();
                acc += d * (n_mass / denom);

                if n_next == 0 { break; }
                node = n_next;
            } else if n_leaf {
                for i in n_bodies {
                    let body = &bodies[i];
                    let d = body.pos - pos;
                    let d_sq = d.mag_sq();
                    let soft_sq = self.e_sq.max(body.radius * body.radius);
                    let denom = (d_sq + soft_sq) * d_sq.sqrt();
                    acc += d * (body.grav_mass / denom).min(f32::MAX);
                }

                if n_next == 0 { break; }
                node = n_next;
            } else {
                node = n_child;
            }
        }

        acc
    }

    pub fn acc(&self, bodies: &mut Vec<Body>, gravity_scale: f32) {
        let shield_age_i = SHIELD_AGE_CTRL.load(Ordering::Relaxed).clamp(0, 254);
        let mut blend_age_i = BLEND_AGE_CTRL.load(Ordering::Relaxed).clamp(1, 255);
        if blend_age_i <= shield_age_i {
            blend_age_i = (shield_age_i + 1).min(255);
        }
        let shield_age = shield_age_i as u8;
        let blend_age = blend_age_i as u8;

        let grav_samples: Vec<GravSample> = bodies
            .iter()
            .map(|b| GravSample {
                pos: b.pos,
                radius: b.radius,
                grav_mass: b.mass + b.ghost_mass,
            })
            .collect();

        bodies.par_iter_mut().for_each(|body| {
            if body.age < shield_age && body.system_id > 0 {
                body.acc = Vec3 { x: 0.0, y: 0.0, z: 0.0 };
                return;
            }

            let full_acc = self.acc_pos(body.pos, &grav_samples) * gravity_scale;

            if body.age < blend_age && body.system_id > 0 {
                let blend = (body.age - shield_age) as f32 / (blend_age - shield_age) as f32;
                body.acc = full_acc * blend;
            } else {
                body.acc = full_acc;
            }
        });
    }
}
