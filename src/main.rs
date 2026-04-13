use std::sync::atomic::Ordering;

mod body;
mod partition;
mod quadtree;
mod renderer;
mod simulation;
mod galaxy_templates;
mod observer;
mod utils;
mod utils_hierarchical;

use renderer::Renderer;
use simulation::Simulation;

fn main() {
    // Keep one core for UI/render thread and use the rest for Rayon tasks.
    let threads = std::thread::available_parallelism()
        .map(|n| n.get().saturating_sub(1).max(1))
        .unwrap_or(1);
    rayon::ThreadPoolBuilder::new()
        .num_threads(threads)
        .build_global()
        .unwrap();

    let config = moleculequest::Config {
        window_mode: moleculequest::WindowMode::Windowed(900, 900),
    };

    let mut simulation = Simulation::new();
    std::thread::spawn(move || {
	    loop {
	        if renderer::PAUSED.load(Ordering::Relaxed) {
	            std::thread::yield_now();
	        } else {
	            // dt-Scale: Faktor aus Z-Taste
            let dt_factor = f32::from_bits(renderer::DT_FACTOR_BITS.load(Ordering::Relaxed));
            let base_dt = f32::from_bits(renderer::BASE_DT_BITS.load(Ordering::Relaxed));
            simulation.dt = base_dt * dt_factor;
	            // Gravity-Scale: Faktor aus G-Taste auf der kalibrierten Basisgravitation.
            let g_factor = f32::from_bits(renderer::GRAVITY_FACTOR_BITS.load(Ordering::Relaxed));
            let base_g = f32::from_bits(renderer::BASE_GRAVITY_BITS.load(Ordering::Relaxed));
            simulation.gravity_scale = base_g * g_factor;
	            // Softening folgt nur schwach der Gravitationsskalierung, damit Startorbits nicht überglättet werden.
                    let base_softening = f32::from_bits(renderer::BASE_SOFTENING_BITS.load(Ordering::Relaxed)).abs();
                    let gravity_ratio = (simulation.gravity_scale.abs() / base_g.abs().max(1e-6)).sqrt();
                    let epsilon_scaled = base_softening * gravity_ratio;
                simulation.octree.e_sq = epsilon_scaled * epsilon_scaled;

	            // Sim-Speed: Mehrere Steps pro Frame aus T-Taste
	            let speed_idx = renderer::SIM_SPEED_IDX.load(Ordering::Relaxed);
	            let steps = 1u32 << speed_idx;
	            for _ in 0..steps {
	                simulation.step();
	            }
	        }
	        render(&mut simulation);
	    }
    });

    moleculequest::run::<Renderer>(config);
}

fn render(simulation: &mut Simulation) {
    let mut lock = renderer::UPDATE_LOCK.lock();
    
    // Reset on R key
    if renderer::RESET.swap(false, std::sync::atomic::Ordering::Relaxed) {
        *simulation = Simulation::new();
    }
    
    for body in renderer::SPAWN.lock().drain(..) {
        simulation.bodies.push(body);
    }
    {
        let mut lock = renderer::BODIES.lock();
        lock.clear();
        lock.extend_from_slice(&simulation.bodies);
    }
    {
        let mut lock = renderer::OCTREE.lock();
        lock.clear();
        lock.extend_from_slice(&simulation.octree.nodes);
    }
    *lock |= true;
}
