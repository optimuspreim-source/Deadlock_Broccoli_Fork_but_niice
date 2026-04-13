use ultraviolet::Vec3;

#[derive(Clone, Copy)]
pub struct Body {
    pub pos: Vec3,
    pub vel: Vec3,
    pub acc: Vec3,
    pub mass: f32,
    pub radius: f32,
    pub arm_strength: f32,
    pub system_id: u16,
    pub age: u8,
    pub ghost_mass: f32,
    pub merge_count: u8,
    pub template_tag: u8,
    pub accretion_spawn_budget: f32,
}

impl Body {
    pub fn new(pos: Vec3, vel: Vec3, mass: f32, radius: f32) -> Self {
        Self {
            pos,
            vel,
            acc: Vec3::zero(),
            mass,
            radius,
            arm_strength: 0.0,
            system_id: 0,
            age: 255,
            ghost_mass: 0.0,
            merge_count: 0,
            template_tag: 0,
            accretion_spawn_budget: 0.0,
        }
    }

    pub fn new_with_system(pos: Vec3, vel: Vec3, mass: f32, radius: f32, system_id: u16) -> Self {
        Self {
            pos,
            vel,
            acc: Vec3::zero(),
            mass,
            radius,
            arm_strength: 0.0,
            system_id,
            age: 0,
            ghost_mass: 0.0,
            merge_count: 0,
            template_tag: 0,
            accretion_spawn_budget: 0.0,
        }
    }

    pub fn update(&mut self, dt: f32) {
        self.vel += self.acc * dt;
        self.pos += self.vel * dt;
        if self.age < 255 {
            self.age += 1;
        }
    }
}
