use std::{
    collections::HashSet,
    f32::consts::{PI, TAU, FRAC_PI_2},
    sync::atomic::{AtomicBool, AtomicU8, AtomicU16, AtomicU32, Ordering},
};

use crate::{
    body::Body,
    quadtree::{Node, Octree},
};

use moleculequest::{egui, winit::event::VirtualKeyCode, winit_input_helper::WinitInputHelper};

use palette::{rgb::Rgba, Hsluv, IntoColor};
use ultraviolet::{Vec2, Vec3};

use once_cell::sync::Lazy;
use parking_lot::Mutex;

pub static PAUSED: Lazy<AtomicBool> = Lazy::new(|| false.into());
pub static UPDATE_LOCK: Lazy<Mutex<bool>> = Lazy::new(|| Mutex::new(false));
pub static RESET: Lazy<AtomicBool> = Lazy::new(|| false.into());
pub static NEXT_SYSTEM_ID: Lazy<AtomicU16> = Lazy::new(|| AtomicU16::new(1));

pub static SIM_SPEED_IDX: Lazy<AtomicU8> = Lazy::new(|| AtomicU8::new(0));
pub static DT_SCALE_IDX: Lazy<AtomicU8> = Lazy::new(|| AtomicU8::new(7));
pub static GRAVITY_IDX: Lazy<AtomicU8> = Lazy::new(|| AtomicU8::new(0));
pub static ORBITAL_VEL_IDX: Lazy<AtomicU8> = Lazy::new(|| AtomicU8::new(5)); // idx 0-4: -16..-1, idx 5-9: +1..+16
pub static H_MODE: Lazy<AtomicU8> = Lazy::new(|| AtomicU8::new(0));

pub static BASE_DT_BITS: Lazy<AtomicU32> =
    Lazy::new(|| AtomicU32::new(crate::simulation::BASE_DT.to_bits()));
pub static BASE_GRAVITY_BITS: Lazy<AtomicU32> =
    Lazy::new(|| AtomicU32::new(crate::simulation::BASE_GRAVITY_SCALE.to_bits()));
pub static BASE_SOFTENING_BITS: Lazy<AtomicU32> =
    Lazy::new(|| AtomicU32::new(crate::simulation::BASE_SOFTENING.to_bits()));
pub static DT_FACTOR_BITS: Lazy<AtomicU32> = Lazy::new(|| AtomicU32::new(1.0f32.to_bits()));
pub static GRAVITY_FACTOR_BITS: Lazy<AtomicU32> = Lazy::new(|| AtomicU32::new(1.0f32.to_bits()));
pub static ORBITAL_FACTOR_BITS: Lazy<AtomicU32> = Lazy::new(|| AtomicU32::new(1.0f32.to_bits()));
pub static ARM_RENDER_BOOST_BITS: Lazy<AtomicU32> = Lazy::new(|| AtomicU32::new(1.0f32.to_bits()));
pub static MILKY_ARM_BOOST_BITS: Lazy<AtomicU32> = Lazy::new(|| AtomicU32::new(1.25f32.to_bits()));
pub static UI_CURSOR_X_BITS: Lazy<AtomicU32> = Lazy::new(|| AtomicU32::new(0.0f32.to_bits()));
pub static UI_CURSOR_Y_BITS: Lazy<AtomicU32> = Lazy::new(|| AtomicU32::new(0.0f32.to_bits()));
pub static UI_CURSOR_VALID: Lazy<AtomicBool> = Lazy::new(|| AtomicBool::new(false));

const MAX_PRESET_BODIES: usize = 120_000;

pub static BODIES: Lazy<Mutex<Vec<Body>>> = Lazy::new(|| Mutex::new(Vec::new()));
pub static OCTREE: Lazy<Mutex<Vec<Node>>> = Lazy::new(|| Mutex::new(Vec::new()));

pub static SPAWN: Lazy<Mutex<Vec<Body>>> = Lazy::new(|| Mutex::new(Vec::new()));

#[derive(Clone, Copy)]
struct CenterTarget {
    system_id: u16,
    pos: Vec3,
    mass: f32,
}

pub struct Renderer {
    pos: Vec2,
    scale: f32,

    /// Camera rotation angles for 3-D view.
    /// Yaw rotates around the Y axis; pitch tilts around the X axis.
    /// Default (0, 0) = top-down view of the XY disc plane.
    camera_yaw: f32,
    camera_pitch: f32,

    /// 0 = XY-plane (flat),  1 = XZ-plane,  2 = YZ-plane,  3 = Random
    disc_axis: u8,

    settings_collapsed: bool,

    show_bodies: bool,
    show_octree: bool,

    depth_range: (usize, usize),

    spawn_body: Option<Body>,
    angle: Option<f32>,
    total: Option<f32>,

    confirmed_bodies: Option<Body>,

    // Galaxy-Template Drag-Velocity + Inclination
    pending_galaxy_bodies: Vec<Body>,
    pending_galaxy_bodies_base: Vec<Body>,  // flat disc, before user tilt
    pending_galaxy_incl_yaw: f32,           // user-controlled inclination
    pending_galaxy_incl_pitch: f32,
    pending_galaxy_origin: Option<Vec3>,
    pending_galaxy_mouse: Vec3,
    pending_galaxy_key: Option<VirtualKeyCode>,
    globular_scale_pow: u8,

    bodies: Vec<Body>,
    octree: Vec<Node>,

    overlay_text: String,
    overlay_frames: u32,

    center_targets: Vec<CenterTarget>,
    center_target_idx: usize,
    center_lock: bool,

    milky_way_systems: HashSet<u16>,
}

impl moleculequest::Renderer for Renderer {
    fn new() -> Self {
        Self {
            pos: Vec2::zero(),
            scale: 3600.0,

            camera_yaw: 0.0,
            camera_pitch: 0.0,

            disc_axis: 3,  // default: random inclination

            settings_collapsed: false,

            show_bodies: true,
            show_octree: false,

            depth_range: (0, 0),

            spawn_body: None,
            angle: None,
            total: Some(0.0),

            confirmed_bodies: None,

            pending_galaxy_bodies: Vec::new(),
            pending_galaxy_bodies_base: Vec::new(),
            pending_galaxy_incl_yaw: 0.0,
            pending_galaxy_incl_pitch: 0.0,
            pending_galaxy_origin: None,
            pending_galaxy_mouse: Vec3::zero(),
            pending_galaxy_key: None,
            globular_scale_pow: 0,

            bodies: Vec::new(),
            octree: Vec::new(),

            overlay_text: String::new(),
            overlay_frames: 0,

            center_targets: Vec::new(),
            center_target_idx: 0,
            center_lock: false,

            milky_way_systems: HashSet::new(),
        }
    }

    fn input(&mut self, input: &WinitInputHelper, width: u16, height: u16) {
        if input.key_pressed(VirtualKeyCode::E) || input.key_pressed(VirtualKeyCode::F1) {
            self.settings_collapsed = !self.settings_collapsed;
            self.overlay_text = if self.settings_collapsed {
                "UI: Eingeklappt".to_string()
            } else {
                "UI: Ausgeklappt".to_string()
            };
            self.overlay_frames = 60;
        }

        let cursor_px = || -> (f32, f32) {
            if UI_CURSOR_VALID.load(Ordering::Relaxed) {
                (
                    f32::from_bits(UI_CURSOR_X_BITS.load(Ordering::Relaxed)),
                    f32::from_bits(UI_CURSOR_Y_BITS.load(Ordering::Relaxed)),
                )
            } else {
                input
                    .mouse()
                    .unwrap_or((width as f32 * 0.5, height as f32 * 0.5))
            }
        };

        if input.key_pressed(VirtualKeyCode::Space) {
            let val = PAUSED.load(Ordering::Relaxed);
            PAUSED.store(!val, Ordering::Relaxed)
        }

        if input.key_pressed(VirtualKeyCode::R) {
            RESET.store(true, Ordering::Relaxed);
        }

        if let Some(slot) = Self::pressed_preset_slot(input) {
            let g_idx = GRAVITY_IDX.load(Ordering::Relaxed);
            let g_scale = 64.0 * (1u32 << g_idx) as f32;
            self.apply_numeric_preset(slot, g_scale);
        }

        // '+': cycle multiplier for globular templates (x1 -> x3 -> x9 -> x27 -> x1)
        if input.key_pressed(VirtualKeyCode::Equals) || input.key_pressed(VirtualKeyCode::NumpadAdd) || input.key_pressed(VirtualKeyCode::Plus) {
            self.globular_scale_pow = (self.globular_scale_pow + 1) % 4;
            let mult = 3usize.pow(self.globular_scale_pow as u32);
            self.overlay_text = format!("Kugelhaufen-Skalierung: x{}", mult);
            self.overlay_frames = 90;
        }

        // S: Toggle Octree-Build A (safe) / B (parallel-unsafe)
        if input.key_pressed(VirtualKeyCode::S) {
            let parallel = !crate::quadtree::OCTREE_BUILD_PARALLEL.load(Ordering::Relaxed);
            crate::quadtree::OCTREE_BUILD_PARALLEL.store(parallel, Ordering::Relaxed);
            self.overlay_text = if parallel {
                "Octree-Build: B (parallel)".to_string()
            } else {
                "Octree-Build: A (sicher)".to_string()
            };
            self.overlay_frames = 90;
        }

        // T: Sim-Speed Cycle
        if input.key_pressed(VirtualKeyCode::T) {
            let idx = SIM_SPEED_IDX.load(Ordering::Relaxed);
            let new_idx = if idx >= 6 { 0 } else { idx + 1 };
            SIM_SPEED_IDX.store(new_idx, Ordering::Relaxed);
            self.overlay_text = format!("Speed x{}", 1u32 << new_idx);
            self.overlay_frames = 90;
        }

        // Z: dt-Scale Cycle (-64..−1, 1..64)
        if input.key_pressed(VirtualKeyCode::Z) {
            let idx = DT_SCALE_IDX.load(Ordering::Relaxed);
            let new_idx = if idx >= 13 { 0 } else { idx + 1 };
            DT_SCALE_IDX.store(new_idx, Ordering::Relaxed);
            let factor: i32 = if new_idx < 7 { -((1i32) << (6 - new_idx)) } else { (1i32) << (new_idx - 7) };
            DT_FACTOR_BITS.store((factor as f32).to_bits(), Ordering::Relaxed);
            self.overlay_text = format!("dt x{}", factor);
            self.overlay_frames = 90;
        }

        // G: Gravity-Scale Cycle (1..64)
        if input.key_pressed(VirtualKeyCode::G) {
            let idx = GRAVITY_IDX.load(Ordering::Relaxed);
            let new_idx = if idx >= 6 { 0 } else { idx + 1 };
            GRAVITY_IDX.store(new_idx, Ordering::Relaxed);
            let factor = (1u32 << new_idx) as f32;
            GRAVITY_FACTOR_BITS.store(factor.to_bits(), Ordering::Relaxed);
            self.overlay_text = format!("Gravity x{}", factor as i32);
            self.overlay_frames = 90;
        }

        // O: Orbital-Velocity Cycle (-16,-8,-4,-2,-1, +1,+2,+4,+8,+16)
        if input.key_pressed(VirtualKeyCode::O) {
            let idx = ORBITAL_VEL_IDX.load(Ordering::Relaxed);
            let new_idx = if idx >= 9 { 0 } else { idx + 1 };
            ORBITAL_VEL_IDX.store(new_idx, Ordering::Relaxed);
            let factor: i32 = if new_idx < 5 { -((1i32) << (4 - new_idx as i32)) } else { (1i32) << (new_idx as i32 - 5) };
            ORBITAL_FACTOR_BITS.store((factor as f32).to_bits(), Ordering::Relaxed);
            self.overlay_text = format!("Orbit v x{}", factor);
            self.overlay_frames = 90;
        }

        // H: Color mode cycle
        if input.key_pressed(VirtualKeyCode::H) {
            let mode = H_MODE.load(Ordering::Relaxed);
            let new_mode = if mode >= 2 { 0 } else { mode + 1 };
            H_MODE.store(new_mode, Ordering::Relaxed);
            let label = match new_mode { 0 => "Farbe: Normal", 1 => "Farbe: Hell", _ => "Farbe: Typen" };
            self.overlay_text = label.to_string();
            self.overlay_frames = 90;
        }

        // Q: Toggle template normalization profile (conservative <-> dynamic)
        if input.key_pressed(VirtualKeyCode::Q) {
            let current = crate::simulation::TEMPLATE_PROFILE_IDX.load(Ordering::Relaxed);
            let next = if current >= 1 { 0 } else { current + 1 };
            crate::simulation::TEMPLATE_PROFILE_IDX.store(next, Ordering::Relaxed);
            self.overlay_text = format!(
                "Template-Normierung: {}",
                crate::simulation::template_profile_label(next)
            );
            self.overlay_frames = 90;
        }

        // I: cycle disc spawn axis (XY → XZ → YZ → Random)
        if input.key_pressed(VirtualKeyCode::I) {
            self.disc_axis = (self.disc_axis + 1) % 4;
            self.overlay_text = format!("Scheiben-Ebene: {}", disc_axis_label(self.disc_axis));
            self.overlay_frames = 90;
        }

        if input.key_pressed(VirtualKeyCode::Up) {
            self.center_lock = true;
            self.overlay_text = "Center-Lock: Ein".to_string();
            self.overlay_frames = 90;
        }
        if input.key_pressed(VirtualKeyCode::Down) {
            self.center_lock = false;
            self.overlay_text = "Center-Lock: Aus".to_string();
            self.overlay_frames = 90;
        }
        if !self.center_targets.is_empty() && input.key_pressed(VirtualKeyCode::Right) {
            self.center_target_idx = (self.center_target_idx + 1) % self.center_targets.len();
            self.center_lock = true;
            let c = self.center_targets[self.center_target_idx];
            self.overlay_text = format!("Center System {}", c.system_id);
            self.overlay_frames = 90;
        }
        if !self.center_targets.is_empty() && input.key_pressed(VirtualKeyCode::Left) {
            self.center_target_idx = (self.center_target_idx + self.center_targets.len() - 1) % self.center_targets.len();
            self.center_lock = true;
            let c = self.center_targets[self.center_target_idx];
            self.overlay_text = format!("Center System {}", c.system_id);
            self.overlay_frames = 90;
        }

        if self.center_lock {
            if let Some(center) = self.current_center_target() {
                self.pos = Vec2::new(center.pos.x, center.pos.y);
            }
        }

        let (mx, my) = cursor_px();
        let steps = 5.0;
        let zoom = (-input.scroll_diff() / steps).exp2();
        let target = Vec2::new(mx * 2.0 - width as f32, height as f32 - my * 2.0) / height as f32;
        self.pos += target * self.scale * (1.0 - zoom);
        self.scale *= zoom;

        // Middle mouse:
        //   - while a galaxy key is held → tilt the disc (yaw = left/right, pitch = up/down)
        //   - otherwise → pan the view
        if input.mouse_held(2) {
            let (mdx, mdy) = input.mouse_diff();
            if self.pending_galaxy_key.is_some() {
                self.pending_galaxy_incl_yaw   += mdx / height as f32 * PI;
                self.pending_galaxy_incl_pitch -= mdy / height as f32 * PI;
                // Recompute inclined preview from the stored flat base each time
                // (avoids accumulated floating-point error from incremental rotations)
                if let Some(origin) = self.pending_galaxy_origin {
                    let mut bodies = self.pending_galaxy_bodies_base.clone();
                    crate::utils::apply_inclination(
                        &mut bodies,
                        origin,
                        self.pending_galaxy_incl_yaw,
                        self.pending_galaxy_incl_pitch,
                    );
                    self.pending_galaxy_bodies = bodies;
                }
            } else {
                self.pos.x -= mdx / height as f32 * self.scale * 2.0;
                self.pos.y += mdy / height as f32 * self.scale * 2.0;
            }
        }

        // Left mouse drag: rotate 3-D camera (yaw + pitch)
        if input.mouse_held(0) && !input.mouse_held(1) {
            let (mdx, mdy) = input.mouse_diff();
            self.camera_yaw   += mdx / height as f32 * 2.0;
            self.camera_pitch -= mdy / height as f32 * 2.0;
            self.camera_pitch  = self.camera_pitch.clamp(-FRAC_PI_2 * 0.99, FRAC_PI_2 * 0.99);
        }

        // --- world-space helpers ---
        let world_mouse = || -> Vec2 {
            let (mx, my) = cursor_px();
            let mut mouse = Vec2::new(mx, my);
            mouse *= 2.0 / height as f32;
            mouse.y -= 1.0;
            mouse.y *= -1.0;
            mouse.x -= width as f32 / height as f32;
            mouse * self.scale + self.pos
        };

        // Precompute camera trig once for the unproject closure below.
        let cam_yaw   = self.camera_yaw;
        let cam_pitch = self.camera_pitch;
        let world_mouse_3d = || -> Vec3 {
            let p = world_mouse();
            let (sy, cy) = cam_yaw.sin_cos();
            let (sp, cp) = cam_pitch.sin_cos();
            let wx = if cy.abs() > 1e-3 { p.x / cy } else { p.x };
            let wy = if cp.abs() > 1e-3 { (p.y - sy * sp * wx) / cp } else { p.y };
            Vec3::new(wx, wy, 0.0)
        };

        // --- Galaxy Template Drag-to-Launch ---
        let cursor_ws_3d = world_mouse_3d();

        if self.pending_galaxy_key.is_none() {
            let g_idx = GRAVITY_IDX.load(Ordering::Relaxed);
            let g_scale = 64.0 * (1u32 << g_idx) as f32;

            let galaxy_key_pressed =
            if input.key_pressed(VirtualKeyCode::Y) { Some((VirtualKeyCode::Y, crate::galaxy_templates::GalaxyType::MilkyWay)) }
            else if input.key_pressed(VirtualKeyCode::X) { Some((VirtualKeyCode::X, crate::galaxy_templates::GalaxyType::Triangulum)) }
            else if input.key_pressed(VirtualKeyCode::C) { Some((VirtualKeyCode::C, crate::galaxy_templates::GalaxyType::SmallMagellanicCloud)) }
            else if input.key_pressed(VirtualKeyCode::V) { Some((VirtualKeyCode::V, crate::galaxy_templates::GalaxyType::Whirlpool)) }
            else if input.key_pressed(VirtualKeyCode::B) { Some((VirtualKeyCode::B, crate::galaxy_templates::GalaxyType::Sombrero)) }
            else if input.key_pressed(VirtualKeyCode::N) { Some((VirtualKeyCode::N, crate::galaxy_templates::GalaxyType::ESO383_76)) }
            else if input.key_pressed(VirtualKeyCode::M) { Some((VirtualKeyCode::M, crate::galaxy_templates::GalaxyType::IC1101)) }
            else { None };

            let globular_key_pressed =
            if input.key_pressed(VirtualKeyCode::Comma) { Some((VirtualKeyCode::Comma, crate::galaxy_templates::GlobularTemplate::SparseSmall)) }
            else if input.key_pressed(VirtualKeyCode::Period) { Some((VirtualKeyCode::Period, crate::galaxy_templates::GlobularTemplate::Medium)) }
            else if input.key_pressed(VirtualKeyCode::Minus) { Some((VirtualKeyCode::Minus, crate::galaxy_templates::GlobularTemplate::DenseLarge)) }
            else { None };

            // A: Clustered disc – 10,000 clusters × 10 orbiters = 100,000 bodies
            if input.key_pressed(VirtualKeyCode::A) {
                // 20× the outer radius of the largest template (N=3000, scale=10)
                let disc_radius = 20.0 * (3000_f32).sqrt() * 12.5 * 10.0;
                let orbit_radius = disc_radius * 0.002;
                let sid = NEXT_SYSTEM_ID.fetch_add(1, Ordering::Relaxed);
                let mut bodies = crate::utils::clustered_disc(
                    cursor_ws_3d, disc_radius, orbit_radius, sid, g_scale,
                );
                // orb_factor = 1.0: Kepler-Geschwindigkeiten bleiben erhalten
                self.pending_galaxy_bodies_base = bodies.clone();
                let (iy, ip) = disc_axis_initial_inclination(self.disc_axis);
                self.pending_galaxy_incl_yaw   = iy;
                self.pending_galaxy_incl_pitch = ip;
                crate::utils::apply_inclination(&mut bodies, cursor_ws_3d, iy, ip);
                self.pending_galaxy_bodies = bodies;
                self.pending_galaxy_origin = Some(cursor_ws_3d);
                self.pending_galaxy_key    = Some(VirtualKeyCode::A);
            }

            if let Some((key, galaxy)) = galaxy_key_pressed {
                let sid = NEXT_SYSTEM_ID.fetch_add(1, Ordering::Relaxed);
                let mut bodies = crate::galaxy_templates::create_galaxy_template(
                    galaxy, cursor_ws_3d, sid, g_scale,
                );
                let orb_factor = f32::from_bits(ORBITAL_FACTOR_BITS.load(Ordering::Relaxed));
                for body in &mut bodies {
                    body.vel *= orb_factor;
                }
                if matches!(galaxy, crate::galaxy_templates::GalaxyType::MilkyWay) {
                    self.milky_way_systems.insert(sid);
                }
                self.pending_galaxy_bodies_base = bodies.clone();
                let (iy, ip) = disc_axis_initial_inclination(self.disc_axis);
                self.pending_galaxy_incl_yaw   = iy;
                self.pending_galaxy_incl_pitch = ip;
                crate::utils::apply_inclination(&mut bodies, cursor_ws_3d, iy, ip);
                self.pending_galaxy_bodies = bodies;
                self.pending_galaxy_origin = Some(cursor_ws_3d);
                self.pending_galaxy_key    = Some(key);
            }

            if let Some((key, template)) = globular_key_pressed {
                let sid = NEXT_SYSTEM_ID.fetch_add(1, Ordering::Relaxed);
                let mut bodies = crate::galaxy_templates::create_globular_template(
                    template,
                    cursor_ws_3d,
                    sid,
                    g_scale,
                    self.globular_scale_pow,
                );
                let orb_factor = f32::from_bits(ORBITAL_FACTOR_BITS.load(Ordering::Relaxed));
                for body in &mut bodies {
                    body.vel *= orb_factor;
                }
                self.pending_galaxy_bodies_base = bodies.clone();
                self.pending_galaxy_incl_yaw = 0.0;
                self.pending_galaxy_incl_pitch = 0.0;
                self.pending_galaxy_bodies = bodies;
                self.pending_galaxy_origin = Some(cursor_ws_3d);
                self.pending_galaxy_key = Some(key);
            }
        }

        if self.pending_galaxy_key.is_some() {
            self.pending_galaxy_mouse = cursor_ws_3d;
        }

        if let Some(key) = self.pending_galaxy_key {
            let released = match key {
                VirtualKeyCode::B => input.key_released(VirtualKeyCode::B),
                VirtualKeyCode::N => input.key_released(VirtualKeyCode::N),
                VirtualKeyCode::M => input.key_released(VirtualKeyCode::M),
                VirtualKeyCode::V => input.key_released(VirtualKeyCode::V),
                VirtualKeyCode::C => input.key_released(VirtualKeyCode::C),
                VirtualKeyCode::X => input.key_released(VirtualKeyCode::X),
                VirtualKeyCode::Y => input.key_released(VirtualKeyCode::Y),
                VirtualKeyCode::A => input.key_released(VirtualKeyCode::A),
                VirtualKeyCode::Comma => input.key_released(VirtualKeyCode::Comma),
                VirtualKeyCode::Period => input.key_released(VirtualKeyCode::Period),
                VirtualKeyCode::Minus => input.key_released(VirtualKeyCode::Minus),
                _ => false,
            };
            if released {
                if let Some(origin) = self.pending_galaxy_origin.take() {
                    let vel_delta = self.pending_galaxy_mouse - origin;
                    for body in &mut self.pending_galaxy_bodies {
                        body.vel += vel_delta;
                    }
                    SPAWN.lock().extend(self.pending_galaxy_bodies.drain(..));
                }
                self.pending_galaxy_key = None;
            }
        }

        // Right mouse: spawn single body
        if input.mouse_pressed(1) {
            let mouse = world_mouse_3d();
            self.spawn_body = Some(Body::new(mouse, Vec3::zero(), 1.0, 0.00264));
            self.angle = None;
            self.total = Some(0.0);
        } else if input.mouse_held(1) {
            if let Some(body) = &mut self.spawn_body {
                let mouse = world_mouse_3d();
                if let Some(angle) = self.angle {
                    let d = mouse - body.pos;
                    let angle2 = d.y.atan2(d.x);
                    let a = angle2 - angle;
                    let a = (a + PI).rem_euclid(TAU) - PI;
                    let total = self.total.unwrap() - a;
                    body.mass = (total / TAU).exp2();
                    self.angle = Some(angle2);
                    self.total = Some(total);
                } else {
                    let d = mouse - body.pos;
                    let angle = d.y.atan2(d.x);
                    self.angle = Some(angle);
                }
                body.radius = body.mass.cbrt() / 100.0;
                                body.radius = 0.00264;
                body.vel = mouse - body.pos;
            }
        } else if input.mouse_released(1) {
            self.confirmed_bodies = self.spawn_body.take();
        }
    }

    fn render(&mut self, ctx: &mut moleculequest::RenderContext) {
        {
            let mut lock = UPDATE_LOCK.lock();
            if *lock {
                std::mem::swap(&mut self.bodies, &mut BODIES.lock());
                std::mem::swap(&mut self.octree, &mut OCTREE.lock());
                self.refresh_center_targets();
            }
            if let Some(body) = self.confirmed_bodies.take() {
                self.bodies.push(body);
                SPAWN.lock().push(body);
            }
            *lock = false;
        }

        if self.center_lock {
            if let Some(center) = self.current_center_target() {
                self.pos = Vec2::new(center.pos.x, center.pos.y);
            }
        }

        ctx.clear_circles();
        ctx.clear_lines();
        ctx.clear_rects();
        ctx.set_view_pos(self.pos);
        ctx.set_view_scale(self.scale);

        // Precompute projection components once per frame
        let (sy, cy) = self.camera_yaw.sin_cos();
        let (sp, cp) = self.camera_pitch.sin_cos();
        let pivot = self.current_center_target().map(|c| c.pos).filter(|_| self.center_lock);
        let proj = |p: Vec3| -> Vec2 {
            let local = if let Some(piv) = pivot { p - piv } else { p };
            let px = cy * local.x - sy * local.z;
            let py = sy * sp * local.x + cp * local.y + cy * sp * local.z;
            if let Some(piv) = pivot {
                Vec2::new(px + piv.x, py + piv.y)
            } else {
                Vec2::new(px, py)
            }
        };

        if !self.bodies.is_empty() {
            if self.show_bodies {
                let h_mode = H_MODE.load(Ordering::Relaxed);
                let arm_boost = f32::from_bits(ARM_RENDER_BOOST_BITS.load(Ordering::Relaxed)).max(0.0);
                let milky_boost = f32::from_bits(MILKY_ARM_BOOST_BITS.load(Ordering::Relaxed)).max(0.0);
                // Glow pass first (drawn behind bodies)
                if h_mode == 2 {
                    for i in 0..self.bodies.len() {
                        let body = &self.bodies[i];
                        if body.age >= 10 && body.mass > 0.0 {
                            let [r, g, b, a] = body_color(body, h_mode);
                            let af = a as f32;
                            let p = proj(body.pos);
                            let mw = if self.milky_way_systems.contains(&body.system_id) { milky_boost } else { 1.0 };
                            let glow = 1.0 + body.arm_strength * 1.7 * arm_boost * mw;
                            ctx.draw_circle(p, body.radius * 300.0 * glow, [r, g, b, (af * (0.06 + body.arm_strength * 0.05)) as u8]);
                            ctx.draw_circle(p, body.radius * 183.3 * glow, [r, g, b, (af * (0.14 + body.arm_strength * 0.08)) as u8]);
                            ctx.draw_circle(p, body.radius * 100.0 * glow, [r, g, b, (af * (0.28 + body.arm_strength * 0.10)) as u8]);
                        }
                    }
                }
                for i in 0..self.bodies.len() {
                    let body = &self.bodies[i];
                    if body.age >= 10 && body.mass > 0.0 {
                        let color = body_color(body, h_mode);
                        let mw = if self.milky_way_systems.contains(&body.system_id) { milky_boost } else { 1.0 };
                        let scale = 1.0 + body.arm_strength * 1.35 * arm_boost * mw;
                        ctx.draw_circle(proj(body.pos), body.radius * 33.3 * scale, color);
                    }
                }
            }

            if let Some(body) = &self.confirmed_bodies {
                let p = proj(body.pos);
                let tip = proj(body.pos + body.vel);
                ctx.draw_circle(p, body.radius * 33.3, [0xff; 4]);
                ctx.draw_line(p, tip, [0xff; 4]);
            }

            if let Some(body) = &self.spawn_body {
                let p = proj(body.pos);
                let tip = proj(body.pos + body.vel);
                ctx.draw_circle(p, body.radius * 33.3, [0xff; 4]);
                ctx.draw_line(p, tip, [0xff; 4]);
            }
        }

        // Galaxy preview
        if !self.pending_galaxy_bodies.is_empty() {
            let h_mode = H_MODE.load(Ordering::Relaxed);
            for body in &self.pending_galaxy_bodies {
                let mut color = body_color(body, h_mode);
                color[3] = 0x70;
                let scale = 1.0 + body.arm_strength * 1.35;
                ctx.draw_circle(proj(body.pos), body.radius * 100.0 * scale, color);
            }
            if let Some(origin) = self.pending_galaxy_origin {
                let origin_2d = proj(origin);
                let tip_2d = proj(self.pending_galaxy_mouse);
                ctx.draw_line(origin_2d, tip_2d, [0xff, 0x99, 0x00, 0xff]);
                ctx.draw_circle(origin_2d, self.scale * 0.003, [0xff, 0xff, 0x00, 0xcc]);

                // Disc normal indicator: show the rotation axis as a line
                // Normal of the tilted disc = R_y(yaw)·R_x(pitch) applied to (0,0,1)
                let (sy, cy) = self.camera_yaw.sin_cos();
                let (sp, cp) = self.camera_pitch.sin_cos();
                let (iy, iy_c) = self.pending_galaxy_incl_yaw.sin_cos();
                let (ip, ip_c) = self.pending_galaxy_incl_pitch.sin_cos();
                // Normal after R_y(incl_yaw)·R_x(incl_pitch) of (0,0,1):
                // R_x(p) of (0,0,1) = (0, -sin(p), cos(p))
                // R_y(y) of that    = (cos(p)*sin(y), -sin(p), cos(p)*cos(y))
                let nx = ip_c * iy;
                let ny = -ip;
                let nz = ip_c * iy_c;
                let normal_world = origin + Vec3::new(nx, ny, nz) * 50.0;
                let normal_2d = Vec2::new(
                    cy * normal_world.x - sy * normal_world.z,
                    sy * sp * normal_world.x + cp * normal_world.y + cy * sp * normal_world.z,
                );
                ctx.draw_line(origin_2d, normal_2d, [0x44, 0xcc, 0xff, 0xcc]);
                ctx.draw_circle(normal_2d, self.scale * 0.0015, [0x44, 0xcc, 0xff, 0xff]);
            }
        }

        // Octree debug visualization (simplified: project cube center, draw square)
        if self.show_octree && !self.octree.is_empty() {
            let mut depth_range = self.depth_range;
            if depth_range.0 >= depth_range.1 {
                let mut stack = Vec::new();
                stack.push((Octree::ROOT, 0));

                let mut min_depth = usize::MAX;
                let mut max_depth = 0;
                while let Some((node, depth)) = stack.pop() {
                    let node = &self.octree[node];
                    if node.is_leaf() {
                        if depth < min_depth { min_depth = depth; }
                        if depth > max_depth { max_depth = depth; }
                    } else {
                        for i in 0..8 {
                            stack.push((node.children + i, depth + 1));
                        }
                    }
                }
                depth_range = (min_depth, max_depth);
            }
            let (min_depth, max_depth) = depth_range;

            let mut stack = Vec::new();
            stack.push((Octree::ROOT, 0));
            while let Some((node, depth)) = stack.pop() {
                let node = &self.octree[node];
                if node.is_branch() && depth < max_depth {
                    for i in 0..8 {
                        stack.push((node.children + i, depth + 1));
                    }
                } else if depth >= min_depth {
                    let cube = node.cube;
                    let half = cube.size * 0.5;
                    // Project cube center and approximate as a 2-D square
                    let center_2d = proj(cube.center);
                    let min = center_2d - Vec2::new(half, half);
                    let max = center_2d + Vec2::new(half, half);

                    let t = ((depth - min_depth + !node.is_empty() as usize) as f32)
                        / (max_depth - min_depth + 1) as f32;

                    let start_h = -100.0;
                    let end_h = 80.0;
                    let h = start_h + (end_h - start_h) * t;

                    let c = Hsluv::new(h, 100.0, t * 100.0);
                    let rgba: Rgba = c.into_color();
                    let color = rgba.into_format().into();

                    ctx.draw_rect(min, max, color);
                }
            }
        }
    }

    fn gui(&mut self, ctx: &moleculequest::egui::Context) {
        let pixels_per_point = ctx.pixels_per_point();
        if let Some(pointer_pos) = ctx.input(|i| i.pointer.hover_pos()) {
            // Convert egui logical points to physical pixels to match input width/height.
            UI_CURSOR_X_BITS.store((pointer_pos.x * pixels_per_point).to_bits(), Ordering::Relaxed);
            UI_CURSOR_Y_BITS.store((pointer_pos.y * pixels_per_point).to_bits(), Ordering::Relaxed);
            UI_CURSOR_VALID.store(true, Ordering::Relaxed);
        } else {
            UI_CURSOR_VALID.store(false, Ordering::Relaxed);
        }

        // FPS + particle count
        {
            let body_count = BODIES.lock().len();
            egui::Area::new("stats_overlay")
                .anchor(egui::Align2::LEFT_TOP, [8.0, 8.0])
                .order(egui::Order::Foreground)
                .show(ctx, |ui| {
                    let fps = ctx.input(|i| 1.0 / i.stable_dt);
                    let count_str = {
                        let s = body_count.to_string();
                        let mut out = String::with_capacity(s.len() + s.len() / 3);
                        for (i, c) in s.chars().rev().enumerate() {
                            if i > 0 && i % 3 == 0 { out.push('.'); }
                            out.push(c);
                        }
                        out.chars().rev().collect::<String>()
                    };
                    let pitch_deg = self.camera_pitch.to_degrees();
                    let yaw_deg   = self.camera_yaw.to_degrees();
                    let incl_str = if self.pending_galaxy_key.is_some() {
                        format!(
                            "\nScheibe  Yaw: {:.0}°  Pitch: {:.0}°",
                            self.pending_galaxy_incl_yaw.to_degrees(),
                            self.pending_galaxy_incl_pitch.to_degrees()
                        )
                    } else {
                        String::new()
                    };
                    ui.label(
                        egui::RichText::new(format!(
                            "FPS: {:.0}\nPartikel: {}\nKamera  Yaw: {:.0}°  Pitch: {:.0}°\nEbene: {} (I){}",
                            fps, count_str, yaw_deg, pitch_deg,
                            disc_axis_label(self.disc_axis), incl_str
                        ))
                        .size(14.0)
                        .color(egui::Color32::from_rgb(200, 200, 200)),
                    );
                });
        }

        // Overlay flash
        if self.overlay_frames > 0 {
            let alpha = (self.overlay_frames.min(30) as f32 / 30.0 * 255.0) as u8;
            egui::Area::new("speed_overlay")
                .anchor(egui::Align2::CENTER_CENTER, [0.0, 0.0])
                .order(egui::Order::Foreground)
                .show(ctx, |ui| {
                    ui.label(
                        egui::RichText::new(&self.overlay_text)
                            .size(48.0)
                            .color(egui::Color32::from_rgba_unmultiplied(255, 255, 255, alpha)),
                    );
                });
            self.overlay_frames -= 1;
        }

        let max_h = (ctx.screen_rect().height() - 24.0).max(200.0);
        let panel_width = if self.settings_collapsed { 160.0 } else { 310.0 };
        egui::SidePanel::right("runtime_controls_dock")
            .resizable(false)
            .min_width(panel_width)
            .max_width(panel_width)
            .show(ctx, |ui| {
                ui.horizontal(|ui| {
                    ui.heading("Runtime Controls");
                    ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                        let label = if self.settings_collapsed { "[+]" } else { "[-]" };
                        if ui.button(label).clicked() {
                            self.settings_collapsed = !self.settings_collapsed;
                        }
                    });
                });

                if self.settings_collapsed {
                    ui.label("E/F1 oder [+] zum Ausklappen");
                    return;
                }

                ui.separator();
                egui::ScrollArea::vertical().max_height(max_h - 56.0).show(ui, |ui| {

                        egui::CollapsingHeader::new("Global Physik")
                            .default_open(true)
                            .show(ui, |ui| {
                                let mut base_dt = f32::from_bits(BASE_DT_BITS.load(Ordering::Relaxed));
                                let mut dt_factor = f32::from_bits(DT_FACTOR_BITS.load(Ordering::Relaxed));
                                let mut base_g = f32::from_bits(BASE_GRAVITY_BITS.load(Ordering::Relaxed));
                                let mut g_factor = f32::from_bits(GRAVITY_FACTOR_BITS.load(Ordering::Relaxed));
                                let mut base_soft = f32::from_bits(BASE_SOFTENING_BITS.load(Ordering::Relaxed));
                                let mut orbit_factor = f32::from_bits(ORBITAL_FACTOR_BITS.load(Ordering::Relaxed));

                                ui.label("Alle Slider erlauben positive und negative Werte.");
                                ui.add(egui::Slider::new(&mut base_dt, -0.30..=0.30).text("dt"));
                                ui.add(egui::Slider::new(&mut dt_factor, -64.0..=64.0).text("dt×"));
                                ui.add(egui::Slider::new(&mut base_g, -64.0..=64.0).text("G"));
                                ui.add(egui::Slider::new(&mut g_factor, -64.0..=64.0).text("G×"));
                                ui.add(egui::Slider::new(&mut base_soft, -120.0..=120.0).text("Soft"));
                                ui.add(egui::Slider::new(&mut orbit_factor, -16.0..=16.0).text("Orbit"));

                                BASE_DT_BITS.store(base_dt.to_bits(), Ordering::Relaxed);
                                DT_FACTOR_BITS.store(dt_factor.to_bits(), Ordering::Relaxed);
                                BASE_GRAVITY_BITS.store(base_g.to_bits(), Ordering::Relaxed);
                                GRAVITY_FACTOR_BITS.store(g_factor.to_bits(), Ordering::Relaxed);
                                BASE_SOFTENING_BITS.store(base_soft.to_bits(), Ordering::Relaxed);
                                ORBITAL_FACTOR_BITS.store(orbit_factor.to_bits(), Ordering::Relaxed);

                                if ui.button("Physik zurücksetzen").clicked() {
                                    BASE_DT_BITS.store(crate::simulation::BASE_DT.to_bits(), Ordering::Relaxed);
                                    DT_FACTOR_BITS.store(1.0f32.to_bits(), Ordering::Relaxed);
                                    BASE_GRAVITY_BITS.store(crate::simulation::BASE_GRAVITY_SCALE.to_bits(), Ordering::Relaxed);
                                    GRAVITY_FACTOR_BITS.store(1.0f32.to_bits(), Ordering::Relaxed);
                                    BASE_SOFTENING_BITS.store(crate::simulation::BASE_SOFTENING.to_bits(), Ordering::Relaxed);
                                    ORBITAL_FACTOR_BITS.store(1.0f32.to_bits(), Ordering::Relaxed);
                                }
                            });

                        egui::CollapsingHeader::new("Solver Blend")
                            .default_open(true)
                            .show(ui, |ui| {
                                let mut shield = crate::quadtree::SHIELD_AGE_CTRL.load(Ordering::Relaxed) as f32;
                                let mut blend = crate::quadtree::BLEND_AGE_CTRL.load(Ordering::Relaxed) as f32;
                                ui.add(egui::Slider::new(&mut shield, -128.0..=255.0).text("Shield"));
                                ui.add(egui::Slider::new(&mut blend, -128.0..=255.0).text("Blend"));
                                crate::quadtree::SHIELD_AGE_CTRL.store(shield as i32, Ordering::Relaxed);
                                crate::quadtree::BLEND_AGE_CTRL.store(blend as i32, Ordering::Relaxed);
                            });

                        egui::CollapsingHeader::new("Template & Render")
                            .default_open(true)
                            .show(ui, |ui| {
                                let profile_idx = crate::simulation::TEMPLATE_PROFILE_IDX.load(Ordering::Relaxed);
                                ui.label(format!("Template-Normierung: {} (Q)", crate::simulation::template_profile_label(profile_idx)));

                                let mut arm_boost = f32::from_bits(ARM_RENDER_BOOST_BITS.load(Ordering::Relaxed));
                                let mut mw_boost = f32::from_bits(MILKY_ARM_BOOST_BITS.load(Ordering::Relaxed));
                                ui.add(egui::Slider::new(&mut arm_boost, -2.0..=4.0).text("Arm"));
                                ui.add(egui::Slider::new(&mut mw_boost, -2.0..=4.0).text("MW"));
                                ARM_RENDER_BOOST_BITS.store(arm_boost.to_bits(), Ordering::Relaxed);
                                MILKY_ARM_BOOST_BITS.store(mw_boost.to_bits(), Ordering::Relaxed);
                            });

                        egui::CollapsingHeader::new("Kamera & Center Lock")
                            .default_open(true)
                            .show(ui, |ui| {
                                ui.label("Pfeile: <-/-> Center wechseln, Hoch locken, Runter lösen");
                                ui.checkbox(&mut self.center_lock, "Center Lock aktiv");
                                if let Some(center) = self.current_center_target() {
                                    ui.label(format!("Aktives System: {}", center.system_id));
                                    ui.label(format!("Center-Masse: {:.2}", center.mass));
                                } else {
                                    ui.label("Kein Center verfügbar");
                                }
                                ui.horizontal(|ui| {
                                    ui.label(format!("Yaw: {:.1}°", self.camera_yaw.to_degrees()));
                                    ui.label(format!("Pitch: {:.1}°", self.camera_pitch.to_degrees()));
                                });
                                if ui.button("Kamera zurücksetzen").clicked() {
                                    self.camera_yaw = 0.0;
                                    self.camera_pitch = 0.0;
                                }
                            });

                        egui::CollapsingHeader::new("Darstellung")
                            .default_open(true)
                            .show(ui, |ui| {
                                ui.checkbox(&mut self.show_bodies, "Show Bodies");
                                ui.checkbox(&mut self.show_octree, "Show Octree");
                                if self.show_octree {
                                    let range = &mut self.depth_range;
                                    ui.add(egui::Slider::new(&mut range.0, 0..=64).text("Min"));
                                    ui.add(egui::Slider::new(&mut range.1, 0..=64).text("Max"));
                                }
                                ui.separator();
                                ui.label("Scheiben-Ebene (I)");
                                ui.horizontal(|ui| {
                                    for (idx, label) in ["XY", "XZ", "YZ", "Zufällig"].iter().enumerate() {
                                        if ui.selectable_label(self.disc_axis == idx as u8, *label).clicked() {
                                            self.disc_axis = idx as u8;
                                        }
                                    }
                                });
                            });
                });
            });
    }
}

impl Renderer {
    fn pressed_preset_slot(input: &WinitInputHelper) -> Option<u8> {
        if input.key_pressed(VirtualKeyCode::Key0) || input.key_pressed(VirtualKeyCode::Numpad0) { return Some(0); }
        if input.key_pressed(VirtualKeyCode::Key1) || input.key_pressed(VirtualKeyCode::Numpad1) { return Some(1); }
        if input.key_pressed(VirtualKeyCode::Key2) || input.key_pressed(VirtualKeyCode::Numpad2) { return Some(2); }
        if input.key_pressed(VirtualKeyCode::Key3) || input.key_pressed(VirtualKeyCode::Numpad3) { return Some(3); }
        if input.key_pressed(VirtualKeyCode::Key4) || input.key_pressed(VirtualKeyCode::Numpad4) { return Some(4); }
        if input.key_pressed(VirtualKeyCode::Key5) || input.key_pressed(VirtualKeyCode::Numpad5) { return Some(5); }
        if input.key_pressed(VirtualKeyCode::Key6) || input.key_pressed(VirtualKeyCode::Numpad6) { return Some(6); }
        if input.key_pressed(VirtualKeyCode::Key7) || input.key_pressed(VirtualKeyCode::Numpad7) { return Some(7); }
        if input.key_pressed(VirtualKeyCode::Key8) || input.key_pressed(VirtualKeyCode::Numpad8) { return Some(8); }
        if input.key_pressed(VirtualKeyCode::Key9) || input.key_pressed(VirtualKeyCode::Numpad9) { return Some(9); }
        None
    }

    fn spawn_template_with_velocity(
        &mut self,
        out: &mut Vec<Body>,
        g_scale: f32,
        galaxy: crate::galaxy_templates::GalaxyType,
        center: Vec3,
        drift: Vec3,
    ) {
        let sid = NEXT_SYSTEM_ID.fetch_add(1, Ordering::Relaxed);
        let mut bodies = crate::galaxy_templates::create_galaxy_template(galaxy, center, sid, g_scale);
        let (yaw, pitch) = Self::preset_inclination(center, sid);
        crate::utils::apply_inclination(&mut bodies, center, yaw, pitch);
        for body in &mut bodies {
            body.vel += drift;
        }
        if matches!(galaxy, crate::galaxy_templates::GalaxyType::MilkyWay) {
            self.milky_way_systems.insert(sid);
        }
        out.extend(bodies);
    }

    fn preset_inclination(center: Vec3, sid: u16) -> (f32, f32) {
        let h = center.x * 0.00021 + center.y * 0.00013 + center.z * 0.00017 + sid as f32 * 0.37;
        let yaw = h.rem_euclid(TAU);
        let pitch = (h * 1.73).sin() * (FRAC_PI_2 * 0.72);
        (yaw, pitch)
    }

    fn apply_numeric_preset(&mut self, slot: u8, g_scale: f32) {
        self.pending_galaxy_bodies.clear();
        self.pending_galaxy_bodies_base.clear();
        self.pending_galaxy_origin = None;
        self.pending_galaxy_key = None;
        self.spawn_body = None;
        self.confirmed_bodies = None;
        self.center_lock = false;
        self.center_target_idx = 0;
        self.milky_way_systems.clear();

        RESET.store(true, Ordering::Relaxed);
        let mut spawn = SPAWN.lock();
        spawn.clear();

        if slot == 0 {
            self.overlay_text = "Preset 0: Feld geleert".to_string();
            self.overlay_frames = 120;
            return;
        }

        let mut bodies = Vec::new();
        match slot {
            1 => {
                // Einstieg: zwei kleine Galaxien, frontal kollidierend.
                self.spawn_template_with_velocity(
                    &mut bodies,
                    g_scale,
                    crate::galaxy_templates::GalaxyType::SmallMagellanicCloud,
                    Vec3::new(-1900.0, 0.0, 0.0),
                    Vec3::new(1.20, 0.0, 0.0),
                );
                self.spawn_template_with_velocity(
                    &mut bodies,
                    g_scale,
                    crate::galaxy_templates::GalaxyType::SmallMagellanicCloud,
                    Vec3::new(1900.0, 0.0, 0.0),
                    Vec3::new(-1.20, 0.0, 0.0),
                );
            }
            2 => {
                // Größere kleine Systeme mit leichtem Impact-Parameter.
                self.spawn_template_with_velocity(
                    &mut bodies,
                    g_scale,
                    crate::galaxy_templates::GalaxyType::Triangulum,
                    Vec3::new(-2400.0, -380.0, 0.0),
                    Vec3::new(1.30, 0.18, 0.0),
                );
                self.spawn_template_with_velocity(
                    &mut bodies,
                    g_scale,
                    crate::galaxy_templates::GalaxyType::Triangulum,
                    Vec3::new(2400.0, 380.0, 0.0),
                    Vec3::new(-1.30, -0.18, 0.0),
                );
            }
            3 => {
                // Unterschiedliche Massen/Strukturen, schräger Durchflug.
                self.spawn_template_with_velocity(
                    &mut bodies,
                    g_scale,
                    crate::galaxy_templates::GalaxyType::Whirlpool,
                    Vec3::new(-2800.0, -950.0, 0.0),
                    Vec3::new(1.48, 0.72, 0.0),
                );
                self.spawn_template_with_velocity(
                    &mut bodies,
                    g_scale,
                    crate::galaxy_templates::GalaxyType::SmallMagellanicCloud,
                    Vec3::new(2700.0, 980.0, 90.0),
                    Vec3::new(-1.68, -0.78, -0.02),
                );
            }
            4 => {
                // Dreiweg-Kollision mit großer Zentralstruktur.
                self.spawn_template_with_velocity(&mut bodies, g_scale, crate::galaxy_templates::GalaxyType::MilkyWay, Vec3::new(-3200.0, 0.0, 0.0), Vec3::new(1.25, 0.0, 0.0));
                self.spawn_template_with_velocity(&mut bodies, g_scale, crate::galaxy_templates::GalaxyType::Sombrero, Vec3::new(3200.0, 0.0, 0.0), Vec3::new(-1.25, 0.0, 0.0));
                self.spawn_template_with_velocity(&mut bodies, g_scale, crate::galaxy_templates::GalaxyType::Triangulum, Vec3::new(0.0, 3150.0, 260.0), Vec3::new(0.0, -1.35, -0.03));
            }
            5 => {
                // Vier Systeme, zwei Fronten + zwei Störer.
                self.spawn_template_with_velocity(&mut bodies, g_scale, crate::galaxy_templates::GalaxyType::Whirlpool, Vec3::new(-3500.0, -1200.0, 0.0), Vec3::new(1.52, 0.78, 0.0));
                self.spawn_template_with_velocity(&mut bodies, g_scale, crate::galaxy_templates::GalaxyType::Whirlpool, Vec3::new(3500.0, 1200.0, 0.0), Vec3::new(-1.52, -0.78, 0.0));
                self.spawn_template_with_velocity(&mut bodies, g_scale, crate::galaxy_templates::GalaxyType::SmallMagellanicCloud, Vec3::new(-650.0, 3350.0, 0.0), Vec3::new(0.24, -1.62, 0.0));
                self.spawn_template_with_velocity(&mut bodies, g_scale, crate::galaxy_templates::GalaxyType::Triangulum, Vec3::new(760.0, -3400.0, 0.0), Vec3::new(-0.22, 1.58, 0.0));
            }
            6 => {
                // Schwere Kopf-an-Kopf-Kollision + kleine Begleiter.
                self.spawn_template_with_velocity(&mut bodies, g_scale, crate::galaxy_templates::GalaxyType::MilkyWay, Vec3::new(-3600.0, 0.0, 0.0), Vec3::new(1.45, 0.0, 0.0));
                self.spawn_template_with_velocity(&mut bodies, g_scale, crate::galaxy_templates::GalaxyType::IC1101, Vec3::new(3600.0, 0.0, 0.0), Vec3::new(-1.35, 0.0, 0.0));
                self.spawn_template_with_velocity(&mut bodies, g_scale, crate::galaxy_templates::GalaxyType::SmallMagellanicCloud, Vec3::new(-1100.0, 2500.0, 120.0), Vec3::new(0.42, -1.18, 0.0));
                self.spawn_template_with_velocity(&mut bodies, g_scale, crate::galaxy_templates::GalaxyType::Triangulum, Vec3::new(1200.0, -2450.0, -120.0), Vec3::new(-0.44, 1.16, 0.0));
            }
            7 => {
                // Komplexes Multi-Cluster-Setup mit diagonalen Korridoren.
                self.spawn_template_with_velocity(&mut bodies, g_scale, crate::galaxy_templates::GalaxyType::MilkyWay, Vec3::new(-4200.0, -700.0, 0.0), Vec3::new(1.6, 0.22, 0.0));
                self.spawn_template_with_velocity(&mut bodies, g_scale, crate::galaxy_templates::GalaxyType::ESO383_76, Vec3::new(4100.0, 700.0, 0.0), Vec3::new(-1.45, -0.22, 0.0));
                self.spawn_template_with_velocity(&mut bodies, g_scale, crate::galaxy_templates::GalaxyType::Whirlpool, Vec3::new(0.0, 3900.0, 240.0), Vec3::new(0.0, -1.35, -0.03));
                self.spawn_template_with_velocity(&mut bodies, g_scale, crate::galaxy_templates::GalaxyType::Sombrero, Vec3::new(0.0, -3900.0, -240.0), Vec3::new(0.0, 1.35, 0.03));
                self.spawn_template_with_velocity(&mut bodies, g_scale, crate::galaxy_templates::GalaxyType::SmallMagellanicCloud, Vec3::new(5200.0, 0.0, 0.0), Vec3::new(-0.95, 0.0, 0.0));
            }
            8 => {
                // Nur kleine Galaxien, dafür viele (16 Systeme), großer 3D-Raum.
                let ring = [
                    (Vec3::new(-9800.0,    0.0,  1400.0), Vec3::new( 2.20,  0.18, -0.22)),
                    (Vec3::new( 9800.0,    0.0, -1400.0), Vec3::new(-2.20, -0.18,  0.22)),
                    (Vec3::new(   0.0, -9800.0, -1200.0), Vec3::new( 0.18,  2.20,  0.20)),
                    (Vec3::new(   0.0,  9800.0,  1200.0), Vec3::new(-0.18, -2.20, -0.20)),
                    (Vec3::new(-7600.0, -7600.0,  1700.0), Vec3::new( 1.68,  1.60, -0.24)),
                    (Vec3::new( 7600.0,  7600.0, -1700.0), Vec3::new(-1.68, -1.60,  0.24)),
                    (Vec3::new(-7600.0,  7600.0, -1500.0), Vec3::new( 1.68, -1.60,  0.20)),
                    (Vec3::new( 7600.0, -7600.0,  1500.0), Vec3::new(-1.68,  1.60, -0.20)),
                ];
                let cross = [
                    (Vec3::new(-10400.0,  3400.0,  2100.0), Vec3::new( 2.25, -0.72, -0.26)),
                    (Vec3::new( 10400.0, -3400.0, -2100.0), Vec3::new(-2.25,  0.72,  0.26)),
                    (Vec3::new( 3400.0, -10400.0,  2100.0), Vec3::new(-0.72,  2.25, -0.26)),
                    (Vec3::new(-3400.0,  10400.0, -2100.0), Vec3::new( 0.72, -2.25,  0.26)),
                    (Vec3::new(-9200.0,  5200.0, -2600.0), Vec3::new( 1.95, -1.02,  0.30)),
                    (Vec3::new( 9200.0, -5200.0,  2600.0), Vec3::new(-1.95,  1.02, -0.30)),
                    (Vec3::new( 5200.0,  9200.0, -2600.0), Vec3::new(-1.02, -1.95,  0.30)),
                    (Vec3::new(-5200.0, -9200.0,  2600.0), Vec3::new( 1.02,  1.95, -0.30)),
                ];

                for (i, (p, v)) in ring.iter().enumerate() {
                    let g = if i % 2 == 0 { crate::galaxy_templates::GalaxyType::Triangulum } else { crate::galaxy_templates::GalaxyType::SmallMagellanicCloud };
                    self.spawn_template_with_velocity(&mut bodies, g_scale, g, *p, *v);
                }
                for (i, (p, v)) in cross.iter().enumerate() {
                    let g = if i % 2 == 0 { crate::galaxy_templates::GalaxyType::SmallMagellanicCloud } else { crate::galaxy_templates::GalaxyType::Triangulum };
                    self.spawn_template_with_velocity(&mut bodies, g_scale, g, *p, *v);
                }
            }
            9 => {
                // Maximale Komplexität: ausschließlich kleine Galaxien, doppelte Zahl (24 Systeme), großer 3D-Volumenraum.
                let lanes = [
                    (Vec3::new(-13000.0,     0.0,  2200.0), Vec3::new( 2.45,  0.18, -0.28)),
                    (Vec3::new( 13000.0,     0.0, -2200.0), Vec3::new(-2.45, -0.18,  0.28)),
                    (Vec3::new(    0.0, -13000.0, -2200.0), Vec3::new( 0.18,  2.45,  0.28)),
                    (Vec3::new(    0.0,  13000.0,  2200.0), Vec3::new(-0.18, -2.45, -0.28)),
                    (Vec3::new(-9800.0, -9800.0,  2600.0), Vec3::new( 1.92,  1.86, -0.30)),
                    (Vec3::new( 9800.0,  9800.0, -2600.0), Vec3::new(-1.92, -1.86,  0.30)),
                    (Vec3::new(-9800.0,  9800.0, -2600.0), Vec3::new( 1.92, -1.86,  0.30)),
                    (Vec3::new( 9800.0, -9800.0,  2600.0), Vec3::new(-1.92,  1.86, -0.30)),
                    (Vec3::new(-14100.0,  4600.0,  3200.0), Vec3::new( 2.55, -0.90, -0.34)),
                    (Vec3::new( 14100.0, -4600.0, -3200.0), Vec3::new(-2.55,  0.90,  0.34)),
                    (Vec3::new( 4600.0, -14100.0,  3200.0), Vec3::new(-0.90,  2.55, -0.34)),
                    (Vec3::new(-4600.0,  14100.0, -3200.0), Vec3::new( 0.90, -2.55,  0.34)),
                ];
                let streams = [
                    (Vec3::new(-12000.0,  7400.0, -3600.0), Vec3::new( 2.35, -1.28,  0.36)),
                    (Vec3::new( 12000.0, -7400.0,  3600.0), Vec3::new(-2.35,  1.28, -0.36)),
                    (Vec3::new( 7400.0,  12000.0, -3600.0), Vec3::new(-1.28, -2.35,  0.36)),
                    (Vec3::new(-7400.0, -12000.0,  3600.0), Vec3::new( 1.28,  2.35, -0.36)),
                    (Vec3::new(-15300.0, -2400.0,  2400.0), Vec3::new( 2.75,  0.42, -0.26)),
                    (Vec3::new( 15300.0,  2400.0, -2400.0), Vec3::new(-2.75, -0.42,  0.26)),
                    (Vec3::new(-2400.0,  15300.0,  2400.0), Vec3::new( 0.42, -2.75, -0.26)),
                    (Vec3::new( 2400.0, -15300.0, -2400.0), Vec3::new(-0.42,  2.75,  0.26)),
                    (Vec3::new(-10800.0, 10800.0,  3900.0), Vec3::new( 2.06, -2.06, -0.40)),
                    (Vec3::new( 10800.0,-10800.0, -3900.0), Vec3::new(-2.06,  2.06,  0.40)),
                    (Vec3::new( 10800.0, 10800.0, -3900.0), Vec3::new(-2.06, -2.06,  0.40)),
                    (Vec3::new(-10800.0,-10800.0,  3900.0), Vec3::new( 2.06,  2.06, -0.40)),
                ];

                for (i, (p, v)) in lanes.iter().enumerate() {
                    let g = if i % 2 == 0 { crate::galaxy_templates::GalaxyType::Triangulum } else { crate::galaxy_templates::GalaxyType::SmallMagellanicCloud };
                    self.spawn_template_with_velocity(&mut bodies, g_scale, g, *p, *v);
                }
                for (i, (p, v)) in streams.iter().enumerate() {
                    let g = if i % 2 == 0 { crate::galaxy_templates::GalaxyType::SmallMagellanicCloud } else { crate::galaxy_templates::GalaxyType::Triangulum };
                    self.spawn_template_with_velocity(&mut bodies, g_scale, g, *p, *v);
                }
            }
            _ => {}
        }

        if bodies.len() > MAX_PRESET_BODIES {
            bodies.truncate(MAX_PRESET_BODIES);
        }
        spawn.extend(bodies);
        self.overlay_text = format!("Preset {} geladen", slot);
        self.overlay_frames = 120;
    }

    fn refresh_center_targets(&mut self) {
        use std::collections::HashMap;

        let mut strongest_per_system: HashMap<u16, Body> = HashMap::new();
        for body in &self.bodies {
            if body.mass <= 0.0 || body.system_id == 0 {
                continue;
            }
            strongest_per_system
                .entry(body.system_id)
                .and_modify(|best| {
                    if body.mass > best.mass {
                        *best = *body;
                    }
                })
                .or_insert(*body);
        }

        let mut targets: Vec<CenterTarget> = strongest_per_system
            .into_iter()
            .map(|(system_id, b)| CenterTarget {
                system_id,
                pos: b.pos,
                mass: b.mass,
            })
            .collect();

        targets.sort_by(|a, b| {
            b.mass
                .partial_cmp(&a.mass)
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        self.center_targets = targets;
        if self.center_target_idx >= self.center_targets.len() {
            self.center_target_idx = 0;
        }
    }

    fn current_center_target(&self) -> Option<CenterTarget> {
        self.center_targets.get(self.center_target_idx).copied()
    }
}

fn disc_axis_label(axis: u8) -> &'static str {
    match axis {
        0 => "XY",
        1 => "XZ",
        2 => "YZ",
        _ => "Zufällig",
    }
}

fn disc_axis_initial_inclination(axis: u8) -> (f32, f32) {
    match axis {
        0 => (0.0,          0.0),           // XY plane — flat disc (normal = +Z)
        1 => (0.0,         -FRAC_PI_2),     // XZ plane — disc stands up (normal = +Y)
        2 => (FRAC_PI_2,    0.0),           // YZ plane — disc stands up (normal = +X)
        _ => crate::utils::random_inclination(),
    }
}

fn body_color(body: &crate::body::Body, h_mode: u8) -> [u8; 4] {
    let mut color = match h_mode {
        0 => [0xff, 0xff, 0xff, 0xff],
        1 => [0xcc, 0xe8, 0xff, 0xff],
        _ => {
            let mc = body.merge_count;
            const MASSIVE: f32 = 1e5;

            if body.mass >= MASSIVE {
                [0xff, 0xd7, 0x00, 0xff]
            } else {
                match mc {
                    0..=10 => {
                        let h = ((body.pos.x * 17.3 + body.pos.y * 31.7).abs() as u32) % 5;
                        match h {
                            0 => [0xff, 0x55, 0x55, 0xff],
                            1 => [0x55, 0xcc, 0x55, 0xff],
                            2 => [0x55, 0x88, 0xff, 0xff],
                            3 => [0xff, 0xdd, 0x22, 0xff],
                            _ => [0xaa, 0xaa, 0xaa, 0xff],
                        }
                    },
                    11..=20  => [0xee, 0xee, 0xff, 0xff],
                    21..=39  => [0x88, 0x22, 0x11, 0xff],
                    40..=69  => [0xff, 0xee, 0x44, 0xff],
                    70..=89  => [0xcc, 0xee, 0xff, 0xff],
                    90..=99  => [0xff, 0x44, 0x11, 0xff],
                    _        => [0x44, 0xaa, 0xff, 0xff],
                }
            }
        }
    };

    if body.arm_strength > 0.0 {
        let boost = body.arm_strength.clamp(0.0, 1.0);
        color[0] = ((color[0] as f32) + (255.0 - color[0] as f32) * 0.25 * boost) as u8;
        color[1] = ((color[1] as f32) + (245.0 - color[1] as f32) * 0.32 * boost) as u8;
        color[2] = ((color[2] as f32) + (210.0 - color[2] as f32) * 0.18 * boost) as u8;
    }

    color
}
