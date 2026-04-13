use crate::{body::Body, galaxy_templates::GalaxyTemplate};
use std::collections::HashMap;
use ultraviolet::Vec2;

/// Creates a single spiral galaxy centered at the origin with stable Keplerian orbits.
pub fn uniform_disc(n: usize) -> Vec<Body> {
    fastrand::seed(0);
    GalaxyTemplate::spiral(n).generate(Vec2::zero(), Vec2::zero())
}

/// Detects galaxy centers by locating local maxima in the particle density field.
///
/// The simulation domain is partitioned into a grid with cells of side `cell_size`.
/// Cells containing at least `min_density` particles whose count is greater than or
/// equal to all 8 neighbours are returned as center candidates.  The returned
/// position is the centroid of the particles within that cell.
pub fn find_galaxy_centers(bodies: &[Body], cell_size: f32, min_density: usize) -> Vec<Vec2> {
    if bodies.is_empty() || cell_size <= 0.0 {
        return Vec::new();
    }

    let mut grid: HashMap<(i32, i32), Vec<Vec2>> = HashMap::new();
    for body in bodies {
        let cx = (body.pos.x / cell_size).floor() as i32;
        let cy = (body.pos.y / cell_size).floor() as i32;
        grid.entry((cx, cy)).or_default().push(body.pos);
    }

    let mut centers = Vec::new();
    for (&(cx, cy), positions) in &grid {
        if positions.len() < min_density {
            continue;
        }
        // Only keep cells that are local density maxima.
        let is_local_max = (-1..=1i32)
            .flat_map(|dx| (-1..=1i32).map(move |dy| (dx, dy)))
            .filter(|&(dx, dy)| dx != 0 || dy != 0)
            .all(|(dx, dy)| {
                grid.get(&(cx + dx, cy + dy))
                    .map_or(0, Vec::len)
                    <= positions.len()
            });

        if is_local_max {
            let sum = positions.iter().copied().fold(Vec2::zero(), |a, b| a + b);
            centers.push(sum / positions.len() as f32);
        }
    }

    centers
}
