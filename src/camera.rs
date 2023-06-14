use instant::Duration;
use nalgebra as na;
use winit::{
    dpi::PhysicalPosition,
    event::{ElementState, MouseScrollDelta, VirtualKeyCode},
};

pub struct Camera {
    pub position: na::Point3<f32>,
    yaw: f32,
    pitch: f32,
}

impl Camera {
    pub fn new<V: Into<na::Point3<f32>>>(position: V, yaw: f32, pitch: f32) -> Self {
        Self {
            position: position.into(),
            yaw: yaw.to_radians(),
            pitch: pitch.to_radians(),
        }
    }
    pub fn calc_view(&self) -> na::Matrix4<f32> {
        let (sin_pitch, cos_pitch) = self.pitch.sin_cos();
        let (sin_yaw, cos_yaw) = self.yaw.sin_cos();

        let rotation = na::UnitQuaternion::look_at_rh(
            &na::Vector3::new(cos_pitch * cos_yaw, sin_pitch, cos_pitch * sin_yaw),
            &na::Vector3::y_axis(),
        );
        let translation = rotation * (-self.position);

        na::Isometry3::from_parts(translation.into(), rotation).to_matrix()
    }
}

pub struct Projection {
    aspect: f32,
    fovy: f32,
    znear: f32,
}

impl Projection {
    pub fn new(width: u32, height: u32, fovy: f32, znear: f32) -> Self {
        Self {
            aspect: width as f32 / height as f32,
            fovy: fovy.to_radians(),
            znear,
        }
    }

    pub fn resize(&mut self, width: u32, height: u32) {
        self.aspect = width as f32 / height as f32;
    }

    pub fn calc_perspective(&self) -> na::Matrix4<f32> {
        nalgebra_glm::reversed_infinite_perspective_rh_zo(self.aspect, self.fovy, self.znear)
    }
}

pub struct CameraController {
    amount_left: f32,
    amount_right: f32,
    amount_forward: f32,
    amount_backward: f32,
    amount_up: f32,
    amount_down: f32,
    rotate_horizontal: f32,
    rotate_vertical: f32,
    scroll: f32,
    speed: f32,
    sensitivity: f32,
}

impl CameraController {
    pub fn new(speed: f32, sensitivity: f32) -> Self {
        Self {
            amount_left: 0.0,
            amount_right: 0.0,
            amount_forward: 0.0,
            amount_backward: 0.0,
            amount_up: 0.0,
            amount_down: 0.0,
            rotate_horizontal: 0.0,
            rotate_vertical: 0.0,
            scroll: 0.0,
            speed,
            sensitivity,
        }
    }

    pub fn process_keyboard(&mut self, key: VirtualKeyCode, state: ElementState) -> bool {
        let amount = if state == ElementState::Pressed {
            1.0
        } else {
            0.0
        };
        match key {
            VirtualKeyCode::W | VirtualKeyCode::Up => {
                self.amount_forward = amount;
                true
            }
            VirtualKeyCode::A | VirtualKeyCode::Left => {
                self.amount_left = amount;
                true
            }
            VirtualKeyCode::S | VirtualKeyCode::Down => {
                self.amount_backward = amount;
                true
            }
            VirtualKeyCode::D | VirtualKeyCode::Right => {
                self.amount_right = amount;
                true
            }
            VirtualKeyCode::Space => {
                self.amount_up = amount;
                true
            }
            VirtualKeyCode::Q => {
                self.amount_down = amount;
                true
            }
            VirtualKeyCode::Plus => {
                const MAX_SENSITIVITY: f32 = 50.0;
                self.sensitivity += 1.0;
                if self.sensitivity > MAX_SENSITIVITY {
                    self.sensitivity = MAX_SENSITIVITY;
                }
                true
            }
            VirtualKeyCode::Minus => {
                const MIN_SENSITIVITY: f32 = 1.0;
                self.sensitivity -= 1.0;
                if self.sensitivity < MIN_SENSITIVITY {
                    self.sensitivity = MIN_SENSITIVITY;
                }
                true
            }
            _ => false,
        }
    }

    pub fn process_mouse(&mut self, mouse_dx: f64, mouse_dy: f64) {
        self.rotate_horizontal = mouse_dx as f32;
        self.rotate_vertical = mouse_dy as f32;
    }

    pub fn process_scroll(&mut self, delta: &MouseScrollDelta) {
        self.scroll = -match delta {
            MouseScrollDelta::LineDelta(_, scroll) => scroll * 100.0,
            MouseScrollDelta::PixelDelta(PhysicalPosition { y: scroll, .. }) => *scroll as f32,
        }
    }

    pub fn update_camera(&mut self, camera: &mut Camera, delta_time: Duration) {
        const MAX_PITCH: f32 = std::f32::consts::FRAC_PI_2 - 0.0001;
        let dt = delta_time.as_secs_f32();
        let speed = self.speed * dt;
        let sensitivity = self.sensitivity * dt;

        // XZ movements
        let (yaw_sin, yaw_cos) = camera.yaw.sin_cos();
        let forward = na::Vector3::new(yaw_cos, 0.0, yaw_sin).normalize();
        let right = na::Vector3::new(-yaw_sin, 0.0, yaw_cos).normalize();
        camera.position += forward * (self.amount_forward - self.amount_backward) * speed;
        camera.position += right * (self.amount_right - self.amount_left) * speed;

        // Zoom (temporarily only moves forward/backward)
        let (pitch_sin, pitch_cos) = camera.pitch.sin_cos();
        let scrollward =
            na::Vector3::new(pitch_cos * yaw_cos, pitch_sin, pitch_cos * yaw_sin).normalize();
        camera.position += scrollward * self.scroll * self.speed * self.sensitivity * dt;
        self.scroll = 0.0;

        // Y movement
        camera.position.y += (self.amount_up - self.amount_down) * speed;

        // Rotation
        camera.yaw += self.rotate_horizontal * sensitivity;
        camera.pitch += -self.rotate_vertical * sensitivity;

        self.rotate_horizontal = 0.0;
        self.rotate_vertical = 0.0;

        // Limit pitch range
        if camera.pitch < -MAX_PITCH {
            camera.pitch = -MAX_PITCH
        } else if camera.pitch > MAX_PITCH {
            camera.pitch = MAX_PITCH
        }
    }
}

#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
pub struct CameraRaw {
    view_position: na::Vector4<f32>,
    view_proj: na::Matrix4<f32>,
}

impl CameraRaw {
    pub fn from_camera_projection(camera: &Camera, projection: &Projection) -> Self {
        CameraRaw {
            view_position: camera.position.to_homogeneous(),
            view_proj: projection.calc_perspective() * camera.calc_view(),
        }
    }
}
