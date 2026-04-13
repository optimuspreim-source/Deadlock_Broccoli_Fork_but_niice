use crate::body::Body;
use crate::constellation::Constellation;
use ultraviolet::Vec2;

/// Characteristic orbit radius for sub-particles around regular (mass = 1) mass-centers.
/// Must be small relative to the average inter-body spacing so that neighbouring
/// constellations never physically intersect.
const ORBIT_RADIUS: f32 = 2.5;

/// Orbit radius for the galactic-center body (much more massive, larger cloud).
const CENTER_ORBIT_RADIUS: f32 = 14.0;

/// Collision/physics radius for regular mass-center bodies.
/// Kept small so that mass-centers rarely trigger hard collisions, allowing the
/// galaxy to evolve smoothly without chaotic numerical merges.
const MASS_CENTER_RADIUS: f32 = 0.25;

pub fn uniform_disc(n: usize) -> (Vec<Body>, Vec<Constellation>) {
    fastrand::seed(0);
    let inner_radius = 25.0;
    let outer_radius = (n as f32).sqrt() * 5.0;

    let mut bodies: Vec<Body> = Vec::with_capacity(n);

    let m = 1e6;
    // The galactic center: large mass, small physical collision radius.
    let center = Body::new(Vec2::zero(), Vec2::zero(), m as f32, inner_radius * MASS_CENTER_RADIUS);
    bodies.push(center);

    while bodies.len() < n {
        let a = fastrand::f32() * std::f32::consts::TAU;
        let (sin, cos) = a.sin_cos();
        let t = inner_radius / outer_radius;
        let r = fastrand::f32() * (1.0 - t * t) + t * t;
        let pos = Vec2::new(cos, sin) * outer_radius * r.sqrt();
        let vel = Vec2::new(sin, -cos);
        let mass = 1.0f32;

        bodies.push(Body::new(pos, vel, mass, MASS_CENTER_RADIUS));
    }

    bodies.sort_by(|a, b| a.pos.mag_sq().total_cmp(&b.pos.mag_sq()));
    let mut mass = 0.0;
    for i in 0..n {
        mass += bodies[i].mass;
        if bodies[i].pos == Vec2::zero() {
            continue;
        }

        let v = (mass / bodies[i].pos.mag()).sqrt();
        bodies[i].vel *= v;
    }

    // Build a constellation for every mass-center.
    // Index 0 is the galactic center; all others are regular bodies.
    let constellations: Vec<Constellation> = (0..n)
        .map(|i| {
            if i == 0 {
                Constellation::new(CENTER_ORBIT_RADIUS)
            } else {
                Constellation::new(ORBIT_RADIUS)
            }
        })
        .collect();

    (bodies, constellations)
}
