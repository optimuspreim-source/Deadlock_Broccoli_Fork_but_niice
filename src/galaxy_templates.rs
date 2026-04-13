use crate::body::Body;
use ultraviolet::Vec2;

/// Configuration for a galaxy template, controlling initial particle distribution,
/// mass ranges, and accretion spawn behaviour.
pub struct GalaxyTemplate {
    /// Number of disc particles to generate (excluding the central body).
    pub n: usize,
    /// Radius of the central mass (galactic nucleus / black hole).
    pub inner_radius: f32,
    /// Outer radius of the galactic disc.
    pub outer_radius: f32,
    /// Mass of the central body.
    pub central_mass: f32,
    /// Per-particle mass range `(min, max)` sampled uniformly at random.
    pub particle_mass_range: (f32, f32),
    /// Probability (0–1) that one new accretion particle is spawned near a
    /// detected galaxy center on any given simulation step.
    pub accretion_spawn_rate: f32,
    /// Minimum spawn distance from a detected galaxy center (accretion inner edge).
    pub accretion_inner_radius: f32,
    /// Maximum spawn distance from a detected galaxy center (accretion outer edge).
    pub accretion_outer_radius: f32,
}

impl GalaxyTemplate {
    /// Dense spiral galaxy: large central black hole, variable-mass disc particles.
    pub fn spiral(n: usize) -> Self {
        let outer_radius = (n as f32).sqrt() * 5.0;
        Self {
            n,
            inner_radius: 25.0,
            outer_radius,
            central_mass: 1e6,
            particle_mass_range: (0.5, 2.0),
            accretion_spawn_rate: 0.3,
            accretion_inner_radius: outer_radius * 0.02,
            accretion_outer_radius: outer_radius * 0.12,
        }
    }

    /// Diffuse elliptical galaxy: lighter central mass, heavier disc particles.
    #[allow(dead_code)]
    pub fn elliptical(n: usize) -> Self {
        let outer_radius = (n as f32).sqrt() * 7.0;
        Self {
            n,
            inner_radius: 15.0,
            outer_radius,
            central_mass: 5e5,
            particle_mass_range: (1.0, 3.0),
            accretion_spawn_rate: 0.1,
            accretion_inner_radius: outer_radius * 0.02,
            accretion_outer_radius: outer_radius * 0.08,
        }
    }

    /// Generates bodies from this template placed at `center` with bulk velocity `velocity`.
    ///
    /// Each particle receives a Keplerian orbital velocity based on the enclosed mass
    /// interior to its orbit, ensuring stable circular orbits around the central body.
    pub fn generate(&self, center: Vec2, velocity: Vec2) -> Vec<Body> {
        let mut bodies: Vec<Body> = Vec::with_capacity(self.n + 1);

        // Central massive body (black hole / galactic nucleus)
        bodies.push(Body::new(center, velocity, self.central_mass, self.inner_radius));

        let (mass_min, mass_max) = self.particle_mass_range;

        // Pass 1: place disc particles.  body.vel is temporarily set to the unit
        // tangent direction; the actual orbital velocity is assigned in pass 2.
        while bodies.len() <= self.n {
            let a = fastrand::f32() * std::f32::consts::TAU;
            let (sin, cos) = a.sin_cos();
            let t = self.inner_radius / self.outer_radius;
            let r = fastrand::f32() * (1.0 - t * t) + t * t;
            let offset = Vec2::new(cos, sin) * self.outer_radius * r.sqrt();
            // Unit tangent for a clockwise orbit
            let tangent = Vec2::new(sin, -cos);
            let mass = mass_min + fastrand::f32() * (mass_max - mass_min);
            let radius = mass.cbrt();
            bodies.push(Body::new(center + offset, tangent, mass, radius));
        }

        // Sort by distance from center so the enclosed-mass accumulation is correct.
        bodies.sort_by(|a, b| {
            (a.pos - center).mag_sq().total_cmp(&(b.pos - center).mag_sq())
        });

        // Pass 2: assign Keplerian orbital velocities.
        let mut enclosed_mass = 0.0_f32;
        for body in &mut bodies {
            enclosed_mass += body.mass;
            let offset = body.pos - center;
            if offset == Vec2::zero() {
                continue;
            }
            let orbital_speed = (enclosed_mass / offset.mag()).sqrt();
            // body.vel currently holds the unit tangent direction assigned above.
            body.vel = velocity + body.vel * orbital_speed;
        }

        bodies
    }
}
