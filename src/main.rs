use std::sync::atomic::Ordering;

mod body;
mod constellation;
mod quadtree;
mod renderer;
mod simulation;
mod utils;

use body::Body;
use constellation::Constellation;
use renderer::Renderer;
use simulation::Simulation;
use ultraviolet::Vec2;

/// Rendered radius of each sub-particle.  Kept deliberately small so that
/// the Lissajous figures read as fine, cloud-like dust rather than large blobs.
const SUB_PARTICLE_RADIUS: f32 = 0.35;

/// When a body is manually spawned the orbit radius is scaled from the body's
/// own physics radius.  This factor keeps the cloud a comfortable fraction of
/// the body's own size while still being visually legible.
const SPAWNED_ORBIT_RADIUS_SCALE: f32 = 10.0;

/// Minimum orbit radius for manually-spawned bodies (same as the background
/// default used in utils::ORBIT_RADIUS so lone spawned bodies behave similarly).
const SPAWNED_MIN_ORBIT_RADIUS: f32 = 2.5;

fn main() {
    let config = quarkstrom::Config {
        window_mode: quarkstrom::WindowMode::Windowed(900, 900),
    };

    let mut simulation = Simulation::new();

    std::thread::spawn(move || {
	    loop {
	        if renderer::PAUSED.load(Ordering::Relaxed) {
	            std::thread::yield_now();
	        } else {
	            simulation.step();
	        }
	        render(&mut simulation);
	    }
    });

    quarkstrom::run::<Renderer>(config);
}

fn render(simulation: &mut Simulation) {
    let mut lock = renderer::UPDATE_LOCK.lock();

    // Handle user-spawned bodies: each new mass-center gets a fresh constellation.
    for body in renderer::SPAWN.lock().drain(..) {
        let orbit_radius = (body.radius * SPAWNED_ORBIT_RADIUS_SCALE).max(SPAWNED_MIN_ORBIT_RADIUS);
        simulation.constellations.push(Constellation::new(orbit_radius));
        simulation.bodies.push(body);
    }

    {
        // Collect sub-particle world-positions as lightweight Body entries.
        // Mass-centers themselves are NOT pushed; they remain invisible.
        let mut lock = renderer::BODIES.lock();
        lock.clear();
        for (i, constellation) in simulation.constellations.iter().enumerate() {
            let center_pos = simulation.bodies[i].pos;
            for pos in constellation.world_positions(center_pos) {
                lock.push(Body::new(pos, Vec2::zero(), 0.0, SUB_PARTICLE_RADIUS));
            }
        }
    }

    {
        let mut lock = renderer::QUADTREE.lock();
        lock.clear();
        lock.extend_from_slice(&simulation.quadtree.nodes);
    }
    *lock |= true;
}
