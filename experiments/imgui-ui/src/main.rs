//! imgui-rs UI experiment for collide-o-scope
//!
//! Run: cargo run (from experiments/imgui-ui/)
//!
//! Standalone window showing effect controls rendered with Dear ImGui
//! and a dark creative-tool theme for visual comparison with egui.

use std::num::NonZeroU32;

use glow::HasContext;
use glutin::config::ConfigTemplateBuilder;
use glutin::context::ContextAttributesBuilder;
use glutin::display::GetGlDisplay;
use glutin::prelude::*;
use glutin::surface::{SwapInterval, WindowSurface};
use glutin_winit::DisplayBuilder;
use imgui::StyleColor;
use imgui_glow_renderer::AutoRenderer;
use imgui_winit_support::{HiDpiMode, WinitPlatform};
use raw_window_handle::HasRawWindowHandle;
use winit::event::{Event, WindowEvent};
use winit::event_loop::EventLoop;
use winit::window::WindowBuilder;

// Same effect parameters as main app (standalone copy for isolation)
#[derive(Debug)]
struct EffectParams {
    // Digital
    pixelate: f32,
    rgb_split: f32,
    hue_shift: f32,
    saturation: f32,
    brightness: f32,
    contrast: f32,
    posterize: f32,
    invert: bool,
    // Analog
    grain_intensity: f32,
    grain_size: f32,
    grain_algo: usize,
    color_grain: bool,
    vignette: f32,
    color_drift: f32,
    // Motion
    breathe_scale: f32,
    breathe_rotation: f32,
    breathe_position: f32,
}

impl Default for EffectParams {
    fn default() -> Self {
        Self {
            pixelate: 1.0,
            rgb_split: 0.0,
            hue_shift: 0.0,
            saturation: 0.0,
            brightness: 0.0,
            contrast: 0.0,
            posterize: 0.0,
            invert: false,
            grain_intensity: 0.0,
            grain_size: 1.0,
            grain_algo: 0,
            color_grain: false,
            vignette: 0.0,
            color_drift: 0.0,
            breathe_scale: 0.0,
            breathe_rotation: 0.0,
            breathe_position: 0.0,
        }
    }
}

fn apply_creative_theme(imgui: &mut imgui::Context) {
    let style = imgui.style_mut();

    // Compact spacing
    style.window_padding = [8.0, 8.0];
    style.frame_padding = [4.0, 2.0];
    style.item_spacing = [6.0, 4.0];
    style.item_inner_spacing = [4.0, 4.0];

    // Minimal rounding
    style.window_rounding = 0.0;
    style.frame_rounding = 2.0;
    style.grab_rounding = 2.0;
    style.tab_rounding = 2.0;

    // Dark creative palette (Resolume/TouchDesigner inspired)
    let colors = &mut style.colors;
    colors[StyleColor::WindowBg as usize] = [0.06, 0.06, 0.08, 1.0];
    colors[StyleColor::ChildBg as usize] = [0.06, 0.06, 0.08, 1.0];
    colors[StyleColor::PopupBg as usize] = [0.08, 0.08, 0.10, 1.0];
    colors[StyleColor::Border as usize] = [0.20, 0.20, 0.25, 0.5];

    colors[StyleColor::FrameBg as usize] = [0.12, 0.12, 0.16, 1.0];
    colors[StyleColor::FrameBgHovered as usize] = [0.16, 0.16, 0.22, 1.0];
    colors[StyleColor::FrameBgActive as usize] = [0.20, 0.20, 0.28, 1.0];

    colors[StyleColor::TitleBg as usize] = [0.06, 0.06, 0.08, 1.0];
    colors[StyleColor::TitleBgActive as usize] = [0.08, 0.08, 0.12, 1.0];

    colors[StyleColor::Header as usize] = [0.14, 0.14, 0.18, 1.0];
    colors[StyleColor::HeaderHovered as usize] = [0.18, 0.18, 0.24, 1.0];
    colors[StyleColor::HeaderActive as usize] = [0.22, 0.22, 0.30, 1.0];

    colors[StyleColor::SliderGrab as usize] = [0.35, 0.55, 0.90, 1.0];
    colors[StyleColor::SliderGrabActive as usize] = [0.45, 0.65, 1.0, 1.0];

    colors[StyleColor::Button as usize] = [0.14, 0.14, 0.18, 1.0];
    colors[StyleColor::ButtonHovered as usize] = [0.25, 0.40, 0.70, 1.0];
    colors[StyleColor::ButtonActive as usize] = [0.30, 0.50, 0.85, 1.0];

    colors[StyleColor::CheckMark as usize] = [0.35, 0.55, 0.90, 1.0];

    colors[StyleColor::Separator as usize] = [0.20, 0.20, 0.25, 0.5];
    colors[StyleColor::SeparatorHovered as usize] = [0.35, 0.55, 0.90, 0.5];

    colors[StyleColor::Tab as usize] = [0.10, 0.10, 0.14, 1.0];
    colors[StyleColor::TabActive as usize] = [0.20, 0.30, 0.50, 1.0];
    colors[StyleColor::TabHovered as usize] = [0.25, 0.40, 0.70, 1.0];

    colors[StyleColor::Text as usize] = [0.85, 0.85, 0.88, 1.0];
    colors[StyleColor::TextDisabled as usize] = [0.45, 0.45, 0.50, 1.0];
}

fn render_effects(ui: &imgui::Ui, effects: &mut EffectParams) {
    // Digital section
    ui.text_disabled("DIGITAL");
    ui.separator();

    ui.slider_config("Pixelate", 1.0f32, 32.0)
        .display_format("%.0f")
        .build(&mut effects.pixelate);
    ui.slider_config("RGB Split", 0.0f32, 30.0)
        .display_format("%.0f")
        .build(&mut effects.rgb_split);
    ui.slider_config("Hue", -180.0f32, 180.0)
        .display_format("%.0f\u{00b0}")
        .build(&mut effects.hue_shift);
    ui.slider_config("Saturation", -1.0f32, 1.0)
        .display_format("%.2f")
        .build(&mut effects.saturation);
    ui.slider_config("Brightness", -1.0f32, 1.0)
        .display_format("%.2f")
        .build(&mut effects.brightness);
    ui.slider_config("Contrast", -1.0f32, 1.0)
        .display_format("%.2f")
        .build(&mut effects.contrast);
    ui.slider_config("Posterize", 0.0f32, 16.0)
        .display_format("%.0f")
        .build(&mut effects.posterize);
    ui.checkbox("Invert", &mut effects.invert);

    ui.spacing();
    ui.spacing();

    // Analog section
    ui.text_disabled("ANALOG");
    ui.separator();

    ui.slider_config("Grain", 0.0f32, 0.3)
        .display_format("%.3f")
        .build(&mut effects.grain_intensity);

    if effects.grain_intensity > 0.0 {
        ui.indent();
        ui.slider_config("Size", 1.0f32, 4.0)
            .display_format("%.0f")
            .build(&mut effects.grain_size);

        let algo_items = ["Gaussian", "Perlin", "Salt&Pepper", "Blue"];
        ui.combo_simple_string("Algorithm", &mut effects.grain_algo, &algo_items);

        ui.checkbox("Color Grain", &mut effects.color_grain);
        ui.unindent();
    }

    ui.slider_config("Vignette", 0.0f32, 1.5)
        .display_format("%.2f")
        .build(&mut effects.vignette);
    ui.slider_config("Color Drift", 0.0f32, 0.02)
        .display_format("%.4f")
        .build(&mut effects.color_drift);

    ui.spacing();
    ui.spacing();

    // Motion section
    ui.text_disabled("MOTION");
    ui.separator();

    ui.slider_config("Breathe Scale", 0.0f32, 0.05)
        .display_format("%.4f")
        .build(&mut effects.breathe_scale);
    ui.slider_config("Breathe Rotate", 0.0f32, 2.0)
        .display_format("%.2f\u{00b0}")
        .build(&mut effects.breathe_rotation);
    ui.slider_config("Breathe Drift", 0.0f32, 0.02)
        .display_format("%.4f")
        .build(&mut effects.breathe_position);
}

fn main() {
    let event_loop = EventLoop::new().unwrap();

    let window_builder = WindowBuilder::new()
        .with_title("collide-o-scope \u{2014} imgui experiment")
        .with_inner_size(winit::dpi::LogicalSize::new(380u32, 720u32));

    // Create OpenGL context via glutin
    let template = ConfigTemplateBuilder::new();
    let display_builder = DisplayBuilder::new().with_window_builder(Some(window_builder));

    let (window, gl_config) = display_builder
        .build(&event_loop, template, |configs| {
            configs
                .reduce(|a, b| {
                    if a.num_samples() > b.num_samples() { a } else { b }
                })
                .unwrap()
        })
        .expect("Failed to create display");

    let window = window.unwrap();
    let raw_window_handle = window.raw_window_handle();

    let context_attrs = ContextAttributesBuilder::new().build(Some(raw_window_handle));
    let gl_display = gl_config.display();

    let gl_context = unsafe {
        gl_display
            .create_context(&gl_config, &context_attrs)
            .expect("Failed to create GL context")
    };

    let size = window.inner_size();
    let surface_attrs = glutin::surface::SurfaceAttributesBuilder::<WindowSurface>::new()
        .build(
            raw_window_handle,
            NonZeroU32::new(size.width).unwrap(),
            NonZeroU32::new(size.height).unwrap(),
        );

    let gl_surface = unsafe {
        gl_display
            .create_window_surface(&gl_config, &surface_attrs)
            .expect("Failed to create surface")
    };

    let gl_context = gl_context
        .make_current(&gl_surface)
        .expect("Failed to make context current");

    let _ = gl_surface.set_swap_interval(
        &gl_context,
        SwapInterval::Wait(NonZeroU32::new(1).unwrap()),
    );

    let glow_context = unsafe {
        glow::Context::from_loader_function_cstr(|s| gl_display.get_proc_address(s))
    };

    // imgui setup
    let mut imgui = imgui::Context::create();
    imgui.set_ini_filename(None);
    apply_creative_theme(&mut imgui);

    let mut platform = WinitPlatform::init(&mut imgui);
    platform.attach_window(imgui.io_mut(), &window, HiDpiMode::Default);

    let mut renderer = AutoRenderer::initialize(glow_context, &mut imgui)
        .expect("Failed to create imgui renderer");

    let mut effects = EffectParams::default();

    // Event loop
    event_loop
        .run(move |event, elwt| {
            match event {
                Event::WindowEvent { event: WindowEvent::CloseRequested, .. } => {
                    elwt.exit();
                }
                Event::WindowEvent { event: WindowEvent::Resized(size), .. } => {
                    if size.width > 0 && size.height > 0 {
                        gl_surface.resize(
                            &gl_context,
                            NonZeroU32::new(size.width).unwrap(),
                            NonZeroU32::new(size.height).unwrap(),
                        );
                    }
                }
                Event::AboutToWait => {
                    window.request_redraw();
                }
                Event::WindowEvent { event: WindowEvent::RedrawRequested, .. } => {
                    platform.prepare_frame(imgui.io_mut(), &window).unwrap();

                    {
                        let ui = imgui.frame();
                        let size = window.inner_size();

                        ui.window("Effects")
                            .position([0.0, 0.0], imgui::Condition::Always)
                            .size(
                                [size.width as f32, size.height as f32],
                                imgui::Condition::Always,
                            )
                            .flags(
                                imgui::WindowFlags::NO_TITLE_BAR
                                    | imgui::WindowFlags::NO_RESIZE
                                    | imgui::WindowFlags::NO_MOVE
                                    | imgui::WindowFlags::NO_COLLAPSE,
                            )
                            .build(|| {
                                if let Some(_tab_bar) = ui.tab_bar("tabs") {
                                    if let Some(_tab) = ui.tab_item("Master") {
                                        ui.spacing();
                                        render_effects(ui, &mut effects);
                                    }
                                    if let Some(_tab) = ui.tab_item("Layer 1") {
                                        ui.spacing();
                                        ui.text_disabled("LAYER");
                                        ui.separator();

                                        let mut speed = 1.0f32;
                                        ui.slider_config("Speed", 0.25f32, 4.0)
                                            .display_format("%.2fx")
                                            .build(&mut speed);

                                        let mut fps = 30.0f32;
                                        ui.slider_config("FPS", 1.0f32, 60.0)
                                            .display_format("%.0f")
                                            .build(&mut fps);

                                        let mut opacity = 1.0f32;
                                        ui.slider_config("Opacity", 0.0f32, 1.0)
                                            .display_format("%.2f")
                                            .build(&mut opacity);

                                        let blend_items = [
                                            "Normal", "Screen", "Multiply", "Difference",
                                        ];
                                        let mut blend = 0usize;
                                        ui.combo_simple_string("Blend", &mut blend, &blend_items);

                                        ui.spacing();
                                        ui.spacing();
                                        render_effects(ui, &mut effects);
                                    }
                                }
                            });
                    }

                    let draw_data = imgui.render();

                    unsafe {
                        renderer.gl_context().clear(glow::COLOR_BUFFER_BIT);
                    }
                    renderer.render(draw_data).unwrap();

                    gl_surface.swap_buffers(&gl_context).unwrap();
                }
                ref e => {
                    platform.handle_event(imgui.io_mut(), &window, e);
                }
            }
        })
        .unwrap();
}
