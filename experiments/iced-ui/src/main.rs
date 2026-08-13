//! iced UI experiment for collide-o-scope
//!
//! Run: cargo run (from experiments/iced-ui/)
//!
//! Standalone window showing effect controls rendered with iced
//! and a dark creative-tool theme. Proper HiDPI and resize handling
//! via wgpu backend.

use iced::widget::{
    checkbox, column, container, horizontal_rule, pick_list, row, scrollable, slider, text,
    toggler, Column,
};
use iced::{color, Element, Length, Theme};

fn main() -> iced::Result {
    iced::application("collide-o-scope — iced experiment", App::update, App::view)
        .theme(|_| custom_theme())
        .window_size((380.0, 720.0))
        .run()
}

// Same effect parameters as main app
#[derive(Debug, Clone)]
struct App {
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
    grain_algo: GrainAlgo,
    color_grain: bool,
    vignette: f32,
    color_drift: f32,
    // Motion
    breathe_scale: f32,
    breathe_rotation: f32,
    breathe_position: f32,
    // Tab
    active_tab: Tab,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Tab {
    Master,
    Layer1,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum GrainAlgo {
    Gaussian,
    Perlin,
    SaltPepper,
    Blue,
}

impl std::fmt::Display for GrainAlgo {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            GrainAlgo::Gaussian => write!(f, "Gaussian"),
            GrainAlgo::Perlin => write!(f, "Perlin"),
            GrainAlgo::SaltPepper => write!(f, "Salt&Pepper"),
            GrainAlgo::Blue => write!(f, "Blue"),
        }
    }
}

impl Default for App {
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
            grain_algo: GrainAlgo::Gaussian,
            color_grain: false,
            vignette: 0.0,
            color_drift: 0.0,
            breathe_scale: 0.0,
            breathe_rotation: 0.0,
            breathe_position: 0.0,
            active_tab: Tab::Master,
        }
    }
}

#[derive(Debug, Clone)]
enum Message {
    TabChanged(Tab),
    Pixelate(f32),
    RgbSplit(f32),
    HueShift(f32),
    Saturation(f32),
    Brightness(f32),
    Contrast(f32),
    Posterize(f32),
    Invert(bool),
    GrainIntensity(f32),
    GrainSize(f32),
    GrainAlgoChanged(GrainAlgo),
    ColorGrain(bool),
    Vignette(f32),
    ColorDrift(f32),
    BreatheScale(f32),
    BreatheRotation(f32),
    BreathePosition(f32),
}

impl App {
    fn update(&mut self, message: Message) {
        match message {
            Message::TabChanged(tab) => self.active_tab = tab,
            Message::Pixelate(v) => self.pixelate = v,
            Message::RgbSplit(v) => self.rgb_split = v,
            Message::HueShift(v) => self.hue_shift = v,
            Message::Saturation(v) => self.saturation = v,
            Message::Brightness(v) => self.brightness = v,
            Message::Contrast(v) => self.contrast = v,
            Message::Posterize(v) => self.posterize = v,
            Message::Invert(v) => self.invert = v,
            Message::GrainIntensity(v) => self.grain_intensity = v,
            Message::GrainSize(v) => self.grain_size = v,
            Message::GrainAlgoChanged(v) => self.grain_algo = v,
            Message::ColorGrain(v) => self.color_grain = v,
            Message::Vignette(v) => self.vignette = v,
            Message::ColorDrift(v) => self.color_drift = v,
            Message::BreatheScale(v) => self.breathe_scale = v,
            Message::BreatheRotation(v) => self.breathe_rotation = v,
            Message::BreathePosition(v) => self.breathe_position = v,
        }
    }

    fn view(&self) -> Element<'_, Message> {
        // Tab bar
        let tab_bar = row![
            tab_button("Master", Tab::Master, self.active_tab),
            tab_button("Layer 1", Tab::Layer1, self.active_tab),
        ]
        .spacing(4);

        let content: Element<Message> = match self.active_tab {
            Tab::Master => self.effects_panel(),
            Tab::Layer1 => {
                let layer_controls = column![
                    section_label("LAYER"),
                    horizontal_rule(1),
                    labeled_slider_row("Speed", self.pixelate, 0.25, 4.0, |_| Message::Pixelate(1.0)),
                    labeled_slider_row("FPS", 30.0, 1.0, 60.0, |_| Message::Pixelate(1.0)),
                    labeled_slider_row("Opacity", 1.0, 0.0, 1.0, |_| Message::Pixelate(1.0)),
                ]
                .spacing(4);

                let effects = self.effects_controls();

                column![layer_controls, effects].spacing(8).into()
            }
        };

        let body = scrollable(
            container(
                column![tab_bar, horizontal_rule(1), content]
                    .spacing(8)
                    .padding(12),
            )
            .width(Length::Fill),
        );

        container(body)
            .width(Length::Fill)
            .height(Length::Fill)
            .into()
    }

    fn effects_panel(&self) -> Element<'_, Message> {
        self.effects_controls()
    }

    fn effects_controls(&self) -> Element<'_, Message> {
        let digital = column![
            section_label("DIGITAL"),
            horizontal_rule(1),
            labeled_slider_row("Pixelate", self.pixelate, 1.0, 32.0, Message::Pixelate),
            labeled_slider_row("RGB Split", self.rgb_split, 0.0, 30.0, Message::RgbSplit),
            labeled_slider_row("Hue", self.hue_shift, -180.0, 180.0, Message::HueShift),
            labeled_slider_row("Saturation", self.saturation, -1.0, 1.0, Message::Saturation),
            labeled_slider_row("Brightness", self.brightness, -1.0, 1.0, Message::Brightness),
            labeled_slider_row("Contrast", self.contrast, -1.0, 1.0, Message::Contrast),
            labeled_slider_row("Posterize", self.posterize, 0.0, 16.0, Message::Posterize),
            row![
                text("Invert").size(13).width(80),
                toggler(self.invert).on_toggle(Message::Invert),
            ]
            .spacing(8)
            .align_y(iced::Alignment::Center),
        ]
        .spacing(4);

        let mut analog = Column::new().spacing(4);
        analog = analog
            .push(section_label("ANALOG"))
            .push(horizontal_rule(1))
            .push(labeled_slider_row(
                "Grain",
                self.grain_intensity,
                0.0,
                0.3,
                Message::GrainIntensity,
            ));

        if self.grain_intensity > 0.0 {
            analog = analog
                .push(labeled_slider_row(
                    "  Size",
                    self.grain_size,
                    1.0,
                    4.0,
                    Message::GrainSize,
                ))
                .push(
                    row![
                        text("  Algo").size(13).width(80),
                        pick_list(
                            &[
                                GrainAlgo::Gaussian,
                                GrainAlgo::Perlin,
                                GrainAlgo::SaltPepper,
                                GrainAlgo::Blue,
                            ][..],
                            Some(self.grain_algo),
                            Message::GrainAlgoChanged,
                        )
                        .width(Length::Fill),
                    ]
                    .spacing(8)
                    .align_y(iced::Alignment::Center),
                )
                .push(
                    row![
                        text("  Color").size(13).width(80),
                        checkbox("", self.color_grain).on_toggle(Message::ColorGrain),
                    ]
                    .spacing(8)
                    .align_y(iced::Alignment::Center),
                );
        }

        analog = analog
            .push(labeled_slider_row(
                "Vignette",
                self.vignette,
                0.0,
                1.5,
                Message::Vignette,
            ))
            .push(labeled_slider_row(
                "Drift",
                self.color_drift,
                0.0,
                0.02,
                Message::ColorDrift,
            ));

        let motion = column![
            section_label("MOTION"),
            horizontal_rule(1),
            labeled_slider_row("Bth Scale", self.breathe_scale, 0.0, 0.05, Message::BreatheScale),
            labeled_slider_row(
                "Bth Rotate",
                self.breathe_rotation,
                0.0,
                2.0,
                Message::BreatheRotation
            ),
            labeled_slider_row(
                "Bth Drift",
                self.breathe_position,
                0.0,
                0.02,
                Message::BreathePosition
            ),
        ]
        .spacing(4);

        column![digital, analog, motion].spacing(12).into()
    }
}

// --- Helpers ---

fn section_label(label: &str) -> Element<'_, Message> {
    text(label.to_string())
        .size(11)
        .color(color!(0x667788))
        .into()
}

fn labeled_slider_row<'a>(
    label: &'a str,
    value: f32,
    min: f32,
    max: f32,
    on_change: impl Fn(f32) -> Message + 'a,
) -> Element<'a, Message> {
    let value_text = if max <= 1.0 && min >= -1.0 {
        format!("{:.2}", value)
    } else if max > 100.0 {
        format!("{:.0}", value)
    } else {
        format!("{:.1}", value)
    };

    row![
        text(label).size(13).width(80),
        slider(min..=max, value, on_change).width(Length::Fill),
        text(value_text).size(11).width(50),
    ]
    .spacing(8)
    .align_y(iced::Alignment::Center)
    .into()
}

fn tab_button(label: &str, tab: Tab, active: Tab) -> Element<'_, Message> {
    let style = if tab == active {
        iced::widget::button::primary
    } else {
        iced::widget::button::secondary
    };

    iced::widget::button(text(label).size(13))
        .style(style)
        .on_press(Message::TabChanged(tab))
        .into()
}

fn custom_theme() -> Theme {
    Theme::Dark
}
