use ultraviolet::Vec2;

/// Minimum number of sub-particles in a constellation.
const MIN_SUB_PARTICLES: usize = 2;
/// Maximum number of sub-particles in a constellation (inclusive).
const MAX_SUB_PARTICLES: usize = 6;

/// Sub-particles orbit at `orbit_radius * [MIN_RADIUS_FACTOR, MIN_RADIUS_FACTOR + RADIUS_RANGE]`.
const MIN_RADIUS_FACTOR: f32 = 0.40;
const RADIUS_RANGE: f32 = 0.55;

/// Orbit ellipse aspect ratio (radius_y / radius_x) in `[MIN_ASPECT, MIN_ASPECT + ASPECT_RANGE]`.
const MIN_ASPECT: f32 = 0.35;
const ASPECT_RANGE: f32 = 0.65;

/// Angular speed (rad / simulation-time-unit) in `[MIN_OMEGA, MIN_OMEGA + OMEGA_RANGE]`.
const MIN_OMEGA: f32 = 0.12;
const OMEGA_RANGE: f32 = 0.28;

/// A single sub-particle that traces a Lissajous figure around its mass-center.
#[derive(Clone, Copy)]
pub struct SubParticle {
    /// Current orbit phase (radians, advances each step).
    pub phase: f32,
    /// Angular speed (radians per simulation time-unit).
    pub omega: f32,
    /// Lissajous x-frequency multiplier (integer ≥ 1).
    pub freq_x: f32,
    /// Lissajous y-frequency multiplier (integer ≥ 1).
    pub freq_y: f32,
    /// Phase offset between x and y components (radians).
    pub delta: f32,
    /// Semi-axis of the Lissajous figure in the local x-direction.
    pub radius_x: f32,
    /// Semi-axis of the Lissajous figure in the local y-direction.
    pub radius_y: f32,
    /// Rotation angle of the entire orbit plane (radians).
    pub tilt: f32,
}

impl SubParticle {
    /// Position of this sub-particle relative to its mass-center.
    pub fn local_pos(&self) -> Vec2 {
        let x = self.radius_x * (self.freq_x * self.phase + self.delta).sin();
        let y = self.radius_y * (self.freq_y * self.phase).sin();
        let (sin_t, cos_t) = self.tilt.sin_cos();
        Vec2::new(x * cos_t - y * sin_t, x * sin_t + y * cos_t)
    }

    /// Advance the orbit phase by one time-step.
    pub fn advance(&mut self, dt: f32) {
        self.phase += self.omega * dt;
        if self.phase > std::f32::consts::TAU * 100.0 {
            self.phase -= std::f32::consts::TAU * 100.0;
        }
    }
}

/// A group of sub-particles orbiting a single mass-center in Lissajous figures.
#[derive(Clone)]
pub struct Constellation {
    pub sub_particles: Vec<SubParticle>,
}

/// Integer Lissajous frequency-ratio pairs (a:b) that produce closed, visually
/// interesting figures.
const LISSAJOUS_FREQS: [(u8, u8); 7] = [(1, 1), (1, 2), (1, 3), (2, 3), (3, 4), (3, 5), (2, 5)];

impl Constellation {
    /// Create a new constellation with 2–6 randomly parameterised sub-particles.
    ///
    /// `orbit_radius` sets the characteristic orbit size; it should be small
    /// relative to the spacing between neighbouring mass-centers so that
    /// constellations never physically intersect.
    pub fn new(orbit_radius: f32) -> Self {
        let n = MIN_SUB_PARTICLES
            + fastrand::usize(..(MAX_SUB_PARTICLES - MIN_SUB_PARTICLES + 1));
        let mut sub_particles = Vec::with_capacity(n);

        for _ in 0..n {
            let (fx, fy) = LISSAJOUS_FREQS[fastrand::usize(..LISSAJOUS_FREQS.len())];
            let r = orbit_radius * (MIN_RADIUS_FACTOR + fastrand::f32() * RADIUS_RANGE);
            let aspect = MIN_ASPECT + fastrand::f32() * ASPECT_RANGE;

            sub_particles.push(SubParticle {
                phase: fastrand::f32() * std::f32::consts::TAU,
                omega: MIN_OMEGA + fastrand::f32() * OMEGA_RANGE,
                freq_x: fx as f32,
                freq_y: fy as f32,
                delta: fastrand::f32() * std::f32::consts::TAU,
                radius_x: r,
                radius_y: r * aspect,
                tilt: fastrand::f32() * std::f32::consts::TAU,
            });
        }

        Constellation { sub_particles }
    }

    /// Advance all sub-particle phases by one time-step.
    pub fn advance(&mut self, dt: f32) {
        for sp in &mut self.sub_particles {
            sp.advance(dt);
        }
    }

    /// Compute the world-space positions of all sub-particles given their
    /// mass-center position.
    pub fn world_positions(&self, center: Vec2) -> impl Iterator<Item = Vec2> + '_ {
        self.sub_particles.iter().map(move |sp| center + sp.local_pos())
    }
}
