#![allow(dead_code)]

use crate::body::Body;
use ultraviolet::Vec3;

pub fn circular_speed(enclosed_mass: f32, radius: f32, g_scale: f32, damping: f32) -> f32 {
    let safe_radius = radius.max(1.0);
    (enclosed_mass.max(0.1) / safe_radius).sqrt() * g_scale.sqrt() * damping
}

pub fn tangent_xy(offset: Vec3, spin: f32) -> Vec3 {
    let planar = Vec3::new(-offset.y, offset.x, 0.0);
    if planar.mag_sq() > 1e-6 {
        planar.normalized() * spin.signum().max(1.0).copysign(spin)
    } else {
        Vec3::new(0.0, spin.signum().max(1.0), 0.0)
    }
}

pub fn push_binary_pair(
    bodies: &mut Vec<Body>,
    pair_center: Vec3,
    pair_center_vel: Vec3,
    separation: f32,
    phase: f32,
    vertical_scale: f32,
    pair_total_mass: f32,
    body_mass: f32,
    body_radius: f32,
    system_id: u16,
    spin: f32,
) {
    let (sin_p, cos_p) = phase.sin_cos();
    let orbital_offset = Vec3::new(
        cos_p * separation * 0.5,
        sin_p * separation * 0.5,
        sin_p * vertical_scale,
    );
    let orbital_speed = (pair_total_mass.max(body_mass * 2.0) / separation.max(1.0)).sqrt() * 0.9;
    let tangent = tangent_xy(orbital_offset, spin);

    bodies.push(Body::new_with_system(
        pair_center + orbital_offset,
        pair_center_vel + tangent * orbital_speed,
        body_mass,
        body_radius,
        system_id,
    ));
    bodies.push(Body::new_with_system(
        pair_center - orbital_offset,
        pair_center_vel - tangent * orbital_speed,
        body_mass,
        body_radius,
        system_id,
    ));
}
