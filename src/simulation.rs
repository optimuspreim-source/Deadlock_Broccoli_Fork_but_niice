use crate::{
    body::Body,
    galaxy_templates::GalaxyTemplate,
    quadtree::{Quad, Quadtree},
    utils,
};

use broccoli::aabb::Rect;
use ultraviolet::Vec2;

/// How many simulation steps between full galaxy-center re-detections.
const CENTER_DETECT_INTERVAL: usize = 100;

pub struct Simulation {
    pub dt: f32,
    pub frame: usize,
    pub bodies: Vec<Body>,
    pub quadtree: Quadtree,
    /// Cached galaxy center positions, refreshed every [`CENTER_DETECT_INTERVAL`] steps.
    galaxy_centers: Vec<Vec2>,
    /// Template used to derive accretion spawn parameters.
    accretion_template: GalaxyTemplate,
}

impl Simulation {
    pub fn new() -> Self {
        let dt = 0.05;
        let n = 100000;
        let theta = 1.0;
        let epsilon = 1.0;

        let bodies: Vec<Body> = utils::uniform_disc(n);
        let quadtree = Quadtree::new(theta, epsilon);
        let accretion_template = GalaxyTemplate::spiral(n);

        Self {
            dt,
            frame: 0,
            bodies,
            quadtree,
            galaxy_centers: Vec::new(),
            accretion_template,
        }
    }

    pub fn step(&mut self) {
        self.iterate();
        self.collide();
        self.attract();

        // Periodically refresh galaxy-center positions from the particle density field.
        if self.frame % CENTER_DETECT_INTERVAL == 0 {
            let cell_size = self.accretion_template.outer_radius * 0.05;
            self.galaxy_centers =
                utils::find_galaxy_centers(&self.bodies, cell_size, 50);
        }

        self.spawn_accretion();
        self.frame += 1;
    }

    pub fn attract(&mut self) {
        let quad = Quad::new_containing(&self.bodies);
        self.quadtree.clear(quad);

        for body in &self.bodies {
            self.quadtree.insert(body.pos, body.mass);
        }

        self.quadtree.propagate();

        for body in &mut self.bodies {
            body.acc = self.quadtree.acc(body.pos);
        }
    }

    pub fn iterate(&mut self) {
        for body in &mut self.bodies {
            body.update(self.dt);
        }
    }

    pub fn collide(&mut self) {
        let mut rects = self
            .bodies
            .iter()
            .enumerate()
            .map(|(index, body)| {
                let pos = body.pos;
                let radius = body.radius;
                let min = pos - Vec2::one() * radius;
                let max = pos + Vec2::one() * radius;
                (Rect::new(min.x, max.x, min.y, max.y), index)
            })
            .collect::<Vec<_>>();

        let mut broccoli = broccoli::Tree::new(&mut rects);

        broccoli.find_colliding_pairs(|i, j| {
            let i = *i.unpack_inner();
            let j = *j.unpack_inner();

            self.resolve(i, j);
        });
    }

    /// Spawns accretion particles near each detected galaxy center.
    ///
    /// On each step every detected center independently rolls against
    /// `accretion_spawn_rate`.  A successful roll places one new particle at a
    /// random position within the accretion zone
    /// (`accretion_inner_radius`..`accretion_outer_radius`) and assigns it a
    /// circular orbital velocity derived from the local gravitational acceleration
    /// reported by the quadtree.
    fn spawn_accretion(&mut self) {
        let inner_r = self.accretion_template.accretion_inner_radius;
        let outer_r = self.accretion_template.accretion_outer_radius;
        if inner_r >= outer_r {
            return;
        }

        let (mass_min, mass_max) = self.accretion_template.particle_mass_range;
        let spawn_rate = self.accretion_template.accretion_spawn_rate;

        // Clone centers to avoid simultaneous borrow of self.
        let centers: Vec<Vec2> = self.galaxy_centers.clone();

        for center in centers {
            if fastrand::f32() > spawn_rate {
                continue;
            }

            let a = fastrand::f32() * std::f32::consts::TAU;
            let (sin, cos) = a.sin_cos();
            let r = inner_r + fastrand::f32() * (outer_r - inner_r);
            let spawn_pos = center + Vec2::new(cos, sin) * r;
            // Tangent direction for a clockwise orbit
            let tangent = Vec2::new(sin, -cos);

            let mass = mass_min + fastrand::f32() * (mass_max - mass_min);
            let radius = mass.cbrt();

            // Estimate circular orbital speed from the gravitational acceleration
            // at the spawn point as reported by the already-built quadtree.
            let acc = self.quadtree.acc(spawn_pos);
            let orbital_speed = (acc.mag() * r).sqrt();
            let vel = tangent * orbital_speed;

            self.bodies.push(Body::new(spawn_pos, vel, mass, radius));
        }
    }

    fn resolve(&mut self, i: usize, j: usize) {
        let b1 = &self.bodies[i];
        let b2 = &self.bodies[j];

        let p1 = b1.pos;
        let p2 = b2.pos;

        let r1 = b1.radius;
        let r2 = b2.radius;

        let d = p2 - p1;
        let r = r1 + r2;

        if d.mag_sq() > r * r {
            return;
        }

        let v1 = b1.vel;
        let v2 = b2.vel;

        let v = v2 - v1;

        let d_dot_v = d.dot(v);

        let m1 = b1.mass;
        let m2 = b2.mass;

        let weight1 = m2 / (m1 + m2);
        let weight2 = m1 / (m1 + m2);

        if d_dot_v >= 0.0 && d != Vec2::zero() {
            let tmp = d * (r / d.mag() - 1.0);
            self.bodies[i].pos -= weight1 * tmp;
            self.bodies[j].pos += weight2 * tmp;
            return;
        }

        let v_sq = v.mag_sq();
        let d_sq = d.mag_sq();
        let r_sq = r * r;

        let t = (d_dot_v + (d_dot_v * d_dot_v - v_sq * (d_sq - r_sq)).max(0.0).sqrt()) / v_sq;

        self.bodies[i].pos -= v1 * t;
        self.bodies[j].pos -= v2 * t;

        let p1 = self.bodies[i].pos;
        let p2 = self.bodies[j].pos;
        let d = p2 - p1;
        let d_dot_v = d.dot(v);
        let d_sq = d.mag_sq();

        let tmp = d * (1.5 * d_dot_v / d_sq);
        let v1 = v1 + tmp * weight1;
        let v2 = v2 - tmp * weight2;

        self.bodies[i].vel = v1;
        self.bodies[j].vel = v2;
        self.bodies[i].pos += v1 * t;
        self.bodies[j].pos += v2 * t;
    }
}
