use winit::event::ElementState;
use winit::keyboard::{KeyCode, PhysicalKey};

use crate::effects::EffectUniforms;

/// Actions that can be triggered by keyboard input.
pub enum Action {
    IncreasePixelate,
    DecreasePixelate,
    IncreaseRgbSplit,
    DecreaseRgbSplit,
    ResetEffects,
    TogglePause,
    ToggleFullscreen,
    Quit,
    None,
}

/// Map a physical key press to an action. Shift modifies direction.
pub fn map_key(key: PhysicalKey, state: ElementState, shift: bool) -> Action {
    if state != ElementState::Pressed {
        return Action::None;
    }

    match key {
        PhysicalKey::Code(KeyCode::KeyP) => {
            if shift {
                Action::DecreasePixelate
            } else {
                Action::IncreasePixelate
            }
        }
        PhysicalKey::Code(KeyCode::KeyG) => {
            if shift {
                Action::DecreaseRgbSplit
            } else {
                Action::IncreaseRgbSplit
            }
        }
        PhysicalKey::Code(KeyCode::Digit0) => Action::ResetEffects,
        PhysicalKey::Code(KeyCode::Space) => Action::TogglePause,
        PhysicalKey::Code(KeyCode::KeyF) => Action::ToggleFullscreen,
        PhysicalKey::Code(KeyCode::Escape) => Action::Quit,
        _ => Action::None,
    }
}

/// Apply an action to effect uniforms. Returns control flags.
pub fn apply_action(action: Action, uniforms: &mut EffectUniforms) -> ControlFlow {
    match action {
        Action::IncreasePixelate => uniforms.increase_pixelate(),
        Action::DecreasePixelate => uniforms.decrease_pixelate(),
        Action::IncreaseRgbSplit => uniforms.increase_rgb_split(),
        Action::DecreaseRgbSplit => uniforms.decrease_rgb_split(),
        Action::ResetEffects => uniforms.reset(),
        Action::TogglePause => return ControlFlow::TogglePause,
        Action::ToggleFullscreen => return ControlFlow::ToggleFullscreen,
        Action::Quit => return ControlFlow::Quit,
        Action::None => {}
    }
    ControlFlow::Continue
}

pub enum ControlFlow {
    Continue,
    TogglePause,
    ToggleFullscreen,
    Quit,
}
