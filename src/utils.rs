#![allow(dead_code)]
use crate::body::Body;
use ultraviolet::Vec3;

/// Velocity damping for self-gravitating discs (no center body).
const VELOCITY_DAMPING: f32 = 1.0;

/// Returns (yaw, pitch) for a uniformly random disc orientation on the sphere.
pub fn random_inclination() -> (f32, f32) {
    // Map uniform random to a uniform normal direction on the sphere.
    // Normal after R_y(yaw)·R_x(pitch) of (0,0,1) = (cos(p)*sin(y), -sin(p), cos(p)*cos(y))
    let ny  = 1.0 - 2.0 * fastrand::f32();   // uniform in [-1, 1]
    let r   = (1.0_f32 - ny * ny).max(0.0).sqrt();
    let phi = fastrand::f32() * std::f32::consts::TAU;
    let (nx, nz) = (r * phi.sin(), r * phi.cos());
    let pitch = (-ny).asin();
    let yaw   = nx.atan2(nz);
    (yaw, pitch)
}
/// Rotation order: R_y(yaw) · R_x(pitch).
/// - Dragging left/right → yaw  (tips the disc edge-on horizontally)
/// - Dragging up/down   → pitch (tips the disc edge-on vertically)
pub fn apply_inclination(bodies: &mut Vec<Body>, center: Vec3, yaw: f32, pitch: f32) {
    let (sy, cy) = yaw.sin_cos();
    let (sp, cp) = pitch.sin_cos();
    for body in bodies.iter_mut() {
        let r = body.pos - center;
        let v = body.vel;
        // First R_x(pitch), then R_y(yaw)
        let rx = Vec3::new(r.x,  cp * r.y - sp * r.z,  sp * r.y + cp * r.z);
        let vx = Vec3::new(v.x,  cp * v.y - sp * v.z,  sp * v.y + cp * v.z);
        body.pos = center + Vec3::new(cy * rx.x + sy * rx.z,  rx.y,  -sy * rx.x + cy * rx.z);
        body.vel =          Vec3::new(cy * vx.x + sy * vx.z,  vx.y,  -sy * vx.x + cy * vx.z);
    }
}

/// Flat galactic disc in the XY plane with a small z-perturbation for disc thickness.
pub fn uniform_disc(n: usize) -> Vec<Body> {
    fastrand::seed(0);

    let inner_radius = 10.0_f32;
    let outer_radius = (n as f32).sqrt() * 12.5;
    let body_radius = 1.0;

    let mut bodies: Vec<Body> = Vec::with_capacity(n);

    while bodies.len() < n {
        let a = fastrand::f32() * std::f32::consts::TAU;
        let (sin, cos) = a.sin_cos();
        let t = inner_radius / outer_radius;
        let r = outer_radius * (fastrand::f32() * (1.0 - t * t) + t * t).sqrt();
        let pos = Vec3::new(cos * r, sin * r, (fastrand::f32() - 0.5) * r * 0.05);
        let vel = Vec3::new(sin, -cos, (fastrand::f32() - 0.5) * 0.05);
        bodies.push(Body::new(pos, vel, 1.0, body_radius));
    }

    bodies.sort_by(|a, b| {
        a.pos.mag_sq().partial_cmp(&b.pos.mag_sq()).unwrap_or(std::cmp::Ordering::Equal)
    });

    let local_damping = 1.0;
    let mut mass_acc = 0.0_f32;
    for body in &mut bodies {
        mass_acc += body.mass;
        if body.pos.mag_sq() < 1e-6 { continue; }
        let r = body.pos.mag();
        let v = (mass_acc / r).sqrt() * local_damping;
        body.vel *= v;
    }

    bodies
}

pub fn uniform_disc_at(n: usize, center_pos: Vec3) -> Vec<Body> {
    let mut bodies = uniform_disc(n);
    for body in &mut bodies {
        body.pos += center_pos;
    }
    bodies
}

pub fn uniform_disc_at_scaled(n: usize, center_pos: Vec3, scale: f32) -> Vec<Body> {
    let mut bodies = uniform_disc(n);
    let vel_scale = 1.0 / scale.sqrt();
    for body in &mut bodies {
        body.pos = body.pos * scale + center_pos;
        body.vel *= vel_scale;
        body.radius *= scale;
    }
    bodies
}

pub fn uniform_disc_at_random(n: usize, center_pos: Vec3, g_scale: f32) -> Vec<Body> {
    let seed = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_nanos() as u64)
        .unwrap_or(0);
    fastrand::seed(seed);

    let inner_radius = 10.0_f32;
    let outer_radius = (n as f32).sqrt() * 12.5;
    let body_radius = 1.0;

    let mut bodies: Vec<Body> = Vec::with_capacity(n);

    while bodies.len() < n {
        let a = fastrand::f32() * std::f32::consts::TAU;
        let (sin, cos) = a.sin_cos();
        let t = inner_radius / outer_radius;
        let r = outer_radius * (fastrand::f32() * (1.0 - t * t) + t * t).sqrt();
        let pos = center_pos + Vec3::new(cos * r, sin * r, (fastrand::f32() - 0.5) * r * 0.05);
        let vel = Vec3::new(sin, -cos, (fastrand::f32() - 0.5) * 0.05);
        bodies.push(Body::new(pos, vel, 1.0, body_radius));
    }

    bodies.sort_by(|a, b| {
        let da = (a.pos - center_pos).mag_sq();
        let db = (b.pos - center_pos).mag_sq();
        da.partial_cmp(&db).unwrap_or(std::cmp::Ordering::Equal)
    });

    let mut mass_acc = 0.0_f32;
    for body in &mut bodies {
        mass_acc += body.mass;
        let d = body.pos - center_pos;
        if d.mag_sq() < 1e-6 { continue; }
        let r = d.mag();
        let v = (mass_acc / r).sqrt() * VELOCITY_DAMPING * g_scale.sqrt();
        body.vel *= v;
    }

    bodies
}

pub fn uniform_disc_at_scaled_random(n: usize, center_pos: Vec3, scale: f32, g_scale: f32) -> Vec<Body> {
    let mut bodies = uniform_disc_at_random(n, center_pos, g_scale);
    let vel_scale = 1.0 / scale.sqrt();
    for body in &mut bodies {
        body.pos = (body.pos - center_pos) * scale + center_pos;
        body.vel *= vel_scale;
        body.radius *= scale;
    }
    bodies
}

pub fn uniform_disc_at_scaled_random_with_system(
    n: usize,
    center_pos: Vec3,
    scale: f32,
    system_id: u16,
    g_scale: f32,
) -> Vec<Body> {
    let mut bodies = uniform_disc_at_random(n, center_pos, g_scale);
    let vel_scale = 1.0 / scale.sqrt();
    for body in &mut bodies {
        body.pos = (body.pos - center_pos) * scale + center_pos;
        body.vel *= vel_scale;
          body.radius *= scale; // verdoppelte globale Partikelgröße beibehalten
        body.system_id = system_id;
        body.age = 0;
    }
    // Inclination is now set interactively by the user (middle mouse drag).
    // Return flat disc — the renderer applies apply_inclination() on top.
    bodies
}

pub fn uniform_disc_at_scaled_random_with_system_and_mass(
    n: usize,
    center_pos: Vec3,
    scale: f32,
    system_id: u16,
    _center_mass: f32,
    g_scale: f32,
) -> Vec<Body> {
    uniform_disc_at_scaled_random_with_system(n, center_pos, scale, system_id, g_scale)
}

/// 10,000 cluster centers on a large disc, 10 orbiters each = 100,000 bodies total.
/// Cluster centers follow Keplerian bulk velocities; orbiters circle their local cluster.
pub fn clustered_disc(
    center_pos: Vec3,
    disc_radius: f32,
    orbit_radius: f32,
    system_id: u16,
    g_scale: f32,
) -> Vec<Body> {
    const N_CLUSTERS: usize = 10_000;
    const N_PER: usize = 10;

    let seed = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_nanos() as u64)
        .unwrap_or(0);
    fastrand::seed(seed);

    let inner_r = disc_radius * 0.02;

    // Generate cluster center positions
    let mut centers: Vec<Vec3> = (0..N_CLUSTERS).map(|_| {
        let a = fastrand::f32() * std::f32::consts::TAU;
        let (s, c) = a.sin_cos();
        let t = inner_r / disc_radius;
        let r = disc_radius * (fastrand::f32() * (1.0 - t * t) + t * t).sqrt();
        center_pos + Vec3::new(c * r, s * r, (fastrand::f32() - 0.5) * r * 0.01)
    }).collect();

    // Sort by distance for Keplerian velocity accumulation
    centers.sort_by(|a, b| {
        let da = (*a - center_pos).mag_sq();
        let db = (*b - center_pos).mag_sq();
        da.partial_cmp(&db).unwrap_or(std::cmp::Ordering::Equal)
    });

    let cluster_mass = N_PER as f32; // each cluster contributes N_PER bodies of mass 1
    let mut mass_acc = 0.0_f32;
    let mut bodies = Vec::with_capacity(N_CLUSTERS * N_PER);

    for cluster_center in &centers {
        mass_acc += cluster_mass;
        let d = *cluster_center - center_pos;
        let dist = d.mag();

        // Keplerian bulk velocity for this cluster in the galactic plane
        let bulk_vel = if dist > 1e-4 {
            let speed = (mass_acc / dist).sqrt() * VELOCITY_DAMPING * g_scale.sqrt();
            let tangent = Vec3::new(-d.y, d.x, 0.0) / dist;
            tangent * speed
        } else {
            Vec3::zero()
        };

        // 10 orbiters in circular orbit around cluster center
        let orb_speed = (cluster_mass / orbit_radius.max(1e-4)).sqrt();
        for k in 0..N_PER {
            let angle = k as f32 * std::f32::consts::TAU / N_PER as f32;
            let (so, co) = angle.sin_cos();
            let pos = *cluster_center + Vec3::new(co * orbit_radius, so * orbit_radius, 0.0);
            let vel = bulk_vel + Vec3::new(-so, co, 0.0) * orb_speed;
            bodies.push(Body::new_with_system(pos, vel, 1.0, 1.0, system_id));
        }
    }

    bodies
}
