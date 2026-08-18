use winit::event::ElementState;
use winit::keyboard::{KeyCode, PhysicalKey};

use crate::effects::EffectUniforms;

/// Actions that can be triggered by keyboard input.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Action {
    IncreasePixelate,
    DecreasePixelate,
    IncreaseRgbSplit,
    DecreaseRgbSplit,
    ResetEffects,
    TogglePause,
    ToggleMediaFreeze,
    ToggleFullscreen,
    ToggleOutputWindow,
    ToggleBlackout,
    /// Ask the host to encode a proxy for the selected layer's verified
    /// content identity. The host answers with HUD status, never a dialog.
    CreateSelectedLayerProxy,
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
        PhysicalKey::Code(KeyCode::KeyM) => Action::ToggleMediaFreeze,
        PhysicalKey::Code(KeyCode::KeyF) => Action::ToggleFullscreen,
        PhysicalKey::Code(KeyCode::KeyO) => Action::ToggleOutputWindow,
        PhysicalKey::Code(KeyCode::KeyB) => Action::ToggleBlackout,
        PhysicalKey::Code(KeyCode::KeyY) => Action::CreateSelectedLayerProxy,
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
        Action::ToggleMediaFreeze => return ControlFlow::ToggleMediaFreeze,
        Action::ToggleFullscreen => return ControlFlow::ToggleFullscreen,
        Action::ToggleOutputWindow => return ControlFlow::ToggleOutputWindow,
        Action::ToggleBlackout => return ControlFlow::ToggleBlackout,
        Action::CreateSelectedLayerProxy => return ControlFlow::CreateSelectedLayerProxy,
        Action::Quit => return ControlFlow::Quit,
        Action::None => {}
    }
    ControlFlow::Continue
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ControlFlow {
    Continue,
    TogglePause,
    ToggleMediaFreeze,
    ToggleFullscreen,
    ToggleOutputWindow,
    ToggleBlackout,
    CreateSelectedLayerProxy,
    Quit,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn y_requests_a_selected_layer_proxy_and_release_is_inert() {
        assert_eq!(
            map_key(
                PhysicalKey::Code(KeyCode::KeyY),
                ElementState::Pressed,
                false
            ),
            Action::CreateSelectedLayerProxy
        );
        assert_eq!(
            map_key(
                PhysicalKey::Code(KeyCode::KeyY),
                ElementState::Released,
                false
            ),
            Action::None
        );
        assert_eq!(
            apply_action(
                Action::CreateSelectedLayerProxy,
                &mut EffectUniforms::default()
            ),
            ControlFlow::CreateSelectedLayerProxy
        );
    }

    #[test]
    fn m_is_the_media_freeze_shortcut_and_release_is_inert() {
        assert_eq!(
            map_key(
                PhysicalKey::Code(KeyCode::KeyM),
                ElementState::Pressed,
                false
            ),
            Action::ToggleMediaFreeze
        );
        assert_eq!(
            map_key(
                PhysicalKey::Code(KeyCode::KeyM),
                ElementState::Released,
                false
            ),
            Action::None
        );
        assert_eq!(
            apply_action(Action::ToggleMediaFreeze, &mut EffectUniforms::default()),
            ControlFlow::ToggleMediaFreeze
        );
    }
}
