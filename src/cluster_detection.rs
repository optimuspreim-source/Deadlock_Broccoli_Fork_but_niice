use std::collections::{HashMap, VecDeque};

use crate::body::Body;
use ultraviolet::Vec3;

const DENSITY_GRID_CELL_SIZE: f32 = 420.0;
const DENSE_SEED_CELL_MIN: usize = 16;
const BRIDGE_CELL_MIN: usize = 3;
const MIN_CLUSTER_PARTICLES: usize = 120;

#[derive(Clone)]
pub struct DensityCluster {
    pub body_indices: Vec<usize>,
}

pub struct DensityClusterResult {
    pub clusters: Vec<DensityCluster>,
    pub body_to_cluster: Vec<Option<usize>>,
}

pub fn detect_density_clusters(bodies: &[Body]) -> DensityClusterResult {
    let mut cells: HashMap<(i32, i32, i32), Vec<usize>> = HashMap::new();

    for (idx, body) in bodies.iter().enumerate() {
        if body.mass <= 0.0 {
            continue;
        }
        let key = grid_key(body.pos);
        cells.entry(key).or_default().push(idx);
    }

    let mut visited: HashMap<(i32, i32, i32), bool> = HashMap::with_capacity(cells.len());
    let mut clusters = Vec::new();
    let mut body_to_cluster = vec![None; bodies.len()];

    let cell_keys: Vec<(i32, i32, i32)> = cells.keys().copied().collect();
    for start_key in cell_keys {
        let Some(seed_members) = cells.get(&start_key) else {
            continue;
        };
        if seed_members.len() < DENSE_SEED_CELL_MIN {
            continue;
        }
        if visited.get(&start_key).copied().unwrap_or(false) {
            continue;
        }

        let mut queue = VecDeque::new();
        queue.push_back(start_key);
        visited.insert(start_key, true);

        let mut cluster_indices = Vec::new();

        while let Some(cell) = queue.pop_front() {
            let Some(members) = cells.get(&cell) else {
                continue;
            };
            if members.len() < BRIDGE_CELL_MIN {
                continue;
            }

            cluster_indices.extend(members.iter().copied());

            for neighbor in neighbor_cells(cell) {
                if visited.get(&neighbor).copied().unwrap_or(false) {
                    continue;
                }
                if let Some(neighbor_members) = cells.get(&neighbor) {
                    if neighbor_members.len() >= BRIDGE_CELL_MIN {
                        visited.insert(neighbor, true);
                        queue.push_back(neighbor);
                    }
                }
            }
        }

        if cluster_indices.len() < MIN_CLUSTER_PARTICLES {
            continue;
        }

        let cluster_id = clusters.len();
        for idx in &cluster_indices {
            body_to_cluster[*idx] = Some(cluster_id);
        }
        clusters.push(DensityCluster {
            body_indices: cluster_indices,
        });
    }

    DensityClusterResult {
        clusters,
        body_to_cluster,
    }
}

pub fn cluster_mass_center(bodies: &[Body], indices: &[usize]) -> Option<(Vec3, f32)> {
    let mut mass_sum = 0.0f32;
    let mut weighted_pos = Vec3::zero();
    for &idx in indices {
        let body = bodies[idx];
        if body.mass <= 0.0 {
            continue;
        }
        mass_sum += body.mass;
        weighted_pos += body.pos * body.mass;
    }
    if mass_sum <= 0.0 {
        return None;
    }
    Some((weighted_pos / mass_sum, mass_sum))
}

fn grid_key(pos: Vec3) -> (i32, i32, i32) {
    (
        (pos.x / DENSITY_GRID_CELL_SIZE).floor() as i32,
        (pos.y / DENSITY_GRID_CELL_SIZE).floor() as i32,
        (pos.z / DENSITY_GRID_CELL_SIZE).floor() as i32,
    )
}

fn neighbor_cells(cell: (i32, i32, i32)) -> Vec<(i32, i32, i32)> {
    let mut out = Vec::with_capacity(26);
    for dx in -1..=1 {
        for dy in -1..=1 {
            for dz in -1..=1 {
                if dx == 0 && dy == 0 && dz == 0 {
                    continue;
                }
                out.push((cell.0 + dx, cell.1 + dy, cell.2 + dz));
            }
        }
    }
    out
}
