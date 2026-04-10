use std::fs::{self, OpenOptions};
use std::io::{BufWriter, Write};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::{Duration, Instant};

use ultraviolet::Vec3;

use crate::body::Body;

static UI_AUTO_STREAM_ENABLED: AtomicBool = AtomicBool::new(false);

pub fn set_ui_auto_stream_enabled(enabled: bool) {
    UI_AUTO_STREAM_ENABLED.store(enabled, Ordering::Relaxed);
}

pub fn ui_auto_stream_enabled() -> bool {
    UI_AUTO_STREAM_ENABLED.load(Ordering::Relaxed)
}

pub struct BehaviorObserver {
    env_enabled: bool,
    sample_every: usize,
    rolling_writer: Option<BufWriter<std::fs::File>>,
    rolling_path: PathBuf,
    inbox_dir: PathBuf,
    ui_prev_enabled: bool,
    ui_cycle_started_at: Instant,
    ui_last_snapshot_at: Instant,
    ui_snapshot_idx: usize,
    ui_cycle_id: u64,
    ui_rows_since_snapshot: Vec<String>,
}

impl BehaviorObserver {
    pub fn from_env() -> Self {
        let env_enabled = std::env::var("BH_OBSERVER")
            .map(|v| {
                let t = v.trim().to_ascii_lowercase();
                t == "1" || t == "true" || t == "on" || t == "yes"
            })
            .unwrap_or(false);

        let sample_every = std::env::var("BH_OBSERVER_EVERY")
            .ok()
            .and_then(|v| v.parse::<usize>().ok())
            .unwrap_or(30)
            .max(1);

        let path = std::env::var("BH_OBSERVER_FILE")
            .unwrap_or_else(|_| "galaxy_observer.csv".to_string());
        let rolling_path = PathBuf::from(path);
        let inbox_dir = PathBuf::from("observer_inbox");
        let now = Instant::now();

        Self {
            env_enabled,
            sample_every,
            rolling_writer: None,
            rolling_path,
            inbox_dir,
            ui_prev_enabled: false,
            ui_cycle_started_at: now,
            ui_last_snapshot_at: now,
            ui_snapshot_idx: 0,
            ui_cycle_id: next_cycle_id(),
            ui_rows_since_snapshot: Vec::new(),
        }
    }

    fn csv_header() -> &'static str {
        "frame,dt,system_id,bodies,total_mass,max_r,inner_count,mid_count,outer_count,inner_avg_mass,mid_avg_mass,outer_avg_mass,inner_avg_radius,outer_avg_radius,inner_avg_size,outer_avg_size,inner_osc_ratio,mid_osc_ratio,outer_osc_ratio,merges,elastics"
    }

    fn ensure_rolling_writer(&mut self) {
        if self.rolling_writer.is_some() {
            return;
        }
        if let Some(parent) = self.rolling_path.parent() {
            let _ = fs::create_dir_all(parent);
        }
        let file = OpenOptions::new()
            .create(true)
            .truncate(true)
            .write(true)
            .open(&self.rolling_path)
            .ok();
        self.rolling_writer = file.map(BufWriter::new);
        if let Some(w) = self.rolling_writer.as_mut() {
            let _ = writeln!(w, "{}", Self::csv_header());
            let _ = w.flush();
        }
    }

    fn begin_ui_cycle(&mut self) {
        let now = Instant::now();
        self.ui_cycle_started_at = now;
        self.ui_last_snapshot_at = now;
        self.ui_snapshot_idx = 0;
        self.ui_cycle_id = next_cycle_id();
        self.ui_rows_since_snapshot.clear();
    }

    fn flush_ui_snapshot(&mut self) {
        let _ = fs::create_dir_all(&self.inbox_dir);
        let snapshot_path = self.inbox_dir.join(format!(
            "cycle_{:010}_snapshot_{:02}.csv",
            self.ui_cycle_id,
            self.ui_snapshot_idx + 1
        ));

        if let Ok(mut file) = OpenOptions::new()
            .create(true)
            .truncate(true)
            .write(true)
            .open(snapshot_path)
        {
            let _ = writeln!(file, "{}", Self::csv_header());
            for row in &self.ui_rows_since_snapshot {
                let _ = writeln!(file, "{}", row);
            }
            let _ = file.flush();
        }

        self.ui_rows_since_snapshot.clear();
        self.cleanup_verified_and_stale();
    }

    fn cleanup_verified_and_stale(&self) {
        let Ok(entries) = fs::read_dir(&self.inbox_dir) else {
            return;
        };
        for entry in entries.flatten() {
            let path = entry.path();
            let Some(ext) = path.extension().and_then(|e| e.to_str()) else {
                continue;
            };
            if !ext.eq_ignore_ascii_case("csv") {
                continue;
            }

            let verified_marker = with_verified_extension(&path);
            if verified_marker.exists() || is_stale_csv(&path, Duration::from_secs(600)) {
                let _ = fs::remove_file(&path);
                let _ = fs::remove_file(verified_marker);
            }
        }
    }

    pub fn maybe_record(
        &mut self,
        frame: usize,
        dt: f32,
        bodies: &[Body],
        merges: usize,
        elastics: usize,
    ) {
        let ui_enabled = ui_auto_stream_enabled();
        let effective_enabled = self.env_enabled || ui_enabled;
        if !effective_enabled || frame % self.sample_every != 0 {
            return;
        }
        if self.env_enabled {
            self.ensure_rolling_writer();
        }
        if ui_enabled && !self.ui_prev_enabled {
            self.begin_ui_cycle();
        }
        self.ui_prev_enabled = ui_enabled;

        let mut systems = std::collections::HashSet::new();
        for b in bodies {
            systems.insert(b.system_id);
        }

        let mut rows: Vec<String> = Vec::new();

        for &sid in &systems {
            let mut total_mass = 0.0f32;
            let mut com = Vec3::zero();
            let mut n = 0usize;
            for b in bodies.iter().filter(|b| b.system_id == sid && b.mass > 0.0) {
                n += 1;
                total_mass += b.mass;
                com += b.pos * b.mass;
            }
            if n == 0 || total_mass <= 0.0 {
                continue;
            }
            com /= total_mass;

            let mut max_r = 0.0f32;
            for b in bodies.iter().filter(|b| b.system_id == sid && b.mass > 0.0) {
                let d = b.pos - com;
                let r = (d.x * d.x + d.y * d.y).sqrt();
                if r > max_r {
                    max_r = r;
                }
            }
            if max_r <= 1e-4 {
                continue;
            }

            let r1 = max_r / 3.0;
            let r2 = 2.0 * max_r / 3.0;

            let mut cnt = [0usize; 3];
            let mut mass_sum = [0.0f32; 3];
            let mut rad_sum = [0.0f32; 3];
            let mut size_sum = [0.0f32; 3];
            let mut osc_sum = [0.0f32; 3];

            for b in bodies.iter().filter(|b| b.system_id == sid && b.mass > 0.0) {
                let rel = b.pos - com;
                let r = (rel.x * rel.x + rel.y * rel.y).sqrt();
                let idx = if r < r1 {
                    0
                } else if r < r2 {
                    1
                } else {
                    2
                };

                let radial_dir = if r > 1e-5 {
                    Vec3::new(rel.x / r, rel.y / r, 0.0)
                } else {
                    Vec3::zero()
                };
                let vr = b.vel.dot(radial_dir).abs();
                let vt = (b.vel - radial_dir * b.vel.dot(radial_dir)).mag().max(1e-4);
                let osc_ratio = vr / vt;

                cnt[idx] += 1;
                mass_sum[idx] += b.mass;
                rad_sum[idx] += r;
                size_sum[idx] += b.radius;
                osc_sum[idx] += osc_ratio;
            }

            let avg = |sum: f32, c: usize| if c > 0 { sum / c as f32 } else { 0.0 };
            let row = format!(
                "{},{:.6},{},{},{:.6},{:.6},{},{},{},{:.6},{:.6},{:.6},{:.6},{:.6},{:.6},{:.6},{:.6},{:.6},{:.6},{},{}",
                frame,
                dt,
                sid,
                n,
                total_mass,
                max_r,
                cnt[0],
                cnt[1],
                cnt[2],
                avg(mass_sum[0], cnt[0]),
                avg(mass_sum[1], cnt[1]),
                avg(mass_sum[2], cnt[2]),
                avg(rad_sum[0], cnt[0]),
                avg(rad_sum[2], cnt[2]),
                avg(size_sum[0], cnt[0]),
                avg(size_sum[2], cnt[2]),
                avg(osc_sum[0], cnt[0]),
                avg(osc_sum[1], cnt[1]),
                avg(osc_sum[2], cnt[2]),
                merges,
                elastics
            );
            rows.push(row);
        }

        if let Some(writer) = self.rolling_writer.as_mut() {
            for row in &rows {
                let _ = writeln!(writer, "{}", row);
            }
            let _ = writer.flush();
        }

        if ui_enabled {
            self.ui_rows_since_snapshot.extend(rows);
            if self.ui_last_snapshot_at.elapsed() >= Duration::from_secs(20) {
                self.flush_ui_snapshot();
                self.ui_last_snapshot_at = Instant::now();
                self.ui_snapshot_idx += 1;

                // One cycle = 10 separate snapshots, then a fresh cycle starts.
                if self.ui_snapshot_idx >= 10 {
                    self.begin_ui_cycle();
                }
            }
        }

        self.cleanup_verified_and_stale();
    }
}

fn with_verified_extension(path: &Path) -> PathBuf {
    let mut out = path.to_path_buf();
    out.set_extension("verified");
    out
}

fn is_stale_csv(path: &Path, ttl: Duration) -> bool {
    let Ok(meta) = fs::metadata(path) else {
        return false;
    };
    let Ok(modified) = meta.modified() else {
        return false;
    };
    let Ok(age) = modified.elapsed() else {
        return false;
    };
    age > ttl
}

fn next_cycle_id() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0)
}
