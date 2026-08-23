//! Embedded static files for the web control panel.
//! Files are included at compile time so the binary is self-contained.

use axum::http::{header, StatusCode};
use axum::response::{IntoResponse, Response};

const INDEX_HTML: &str = include_str!("../../static/index.html");
const STYLE_CSS: &str = include_str!("../../static/style.css");
const APP_JS: &str = include_str!("../../static/app.js");

/// The program icon, at the four sizes a browser and a desktop ask for.
/// Embedded like every other panel asset so the binary stays
/// self-contained.
const ICON_16: &[u8] = include_bytes!("../../assets/icon/collide-o-scope-16.png");
const ICON_32: &[u8] = include_bytes!("../../assets/icon/collide-o-scope-32.png");
const ICON_48: &[u8] = include_bytes!("../../assets/icon/collide-o-scope-48.png");
const ICON_256: &[u8] = include_bytes!("../../assets/icon/collide-o-scope-256.png");

/// The B15 per-control help table, generated once from the Rust source of
/// truth in `crate::control_help`. The panel never carries its own copy, so
/// the browser and the native editor's tooltips cannot disagree.
static HELP_JS: std::sync::LazyLock<String> =
    std::sync::LazyLock::new(crate::control_help::panel_javascript);

pub async fn serve(uri: axum::http::Uri) -> Response {
    let path = uri.path().trim_start_matches('/');

    let (content, mime) = match path {
        "" | "index.html" => (INDEX_HTML, "text/html; charset=utf-8"),
        "style.css" => (STYLE_CSS, "text/css; charset=utf-8"),
        "app.js" => (APP_JS, "text/javascript; charset=utf-8"),
        // Icons are bytes rather than text, so they return directly.
        "icon-16.png" | "icon-32.png" | "icon-48.png" | "icon-256.png" => {
            let bytes = match path {
                "icon-16.png" => ICON_16,
                "icon-32.png" => ICON_32,
                "icon-48.png" => ICON_48,
                _ => ICON_256,
            };
            return ([(header::CONTENT_TYPE, "image/png")], bytes).into_response();
        }
        "help.js" => {
            return (
                [(header::CONTENT_TYPE, "text/javascript; charset=utf-8")],
                HELP_JS.as_str(),
            )
                .into_response();
        }
        _ => {
            return (StatusCode::NOT_FOUND, "Not found").into_response();
        }
    };

    ([(header::CONTENT_TYPE, mime)], content).into_response()
}

#[cfg(test)]
mod tests {
    use super::STYLE_CSS;

    #[test]
    fn panel_columns_share_the_scroll_contract_without_magnifying_range_thumbs() {
        assert!(
            STYLE_CSS.contains("min-height: 0;\n  overflow-y: auto;"),
            "the first desktop column must own its vertical scroll"
        );
        assert!(
            STYLE_CSS.contains("overflow: visible;\n  padding: 6px;\n  flex: 0 0 auto;"),
            "the library grid must participate in the column scroll"
        );
        let thumb_hover = STYLE_CSS
            .split_once("input[type=\"range\"]::-webkit-slider-thumb:hover {")
            .and_then(|(_, rest)| rest.split_once('}'))
            .map(|(rule, _)| rule)
            .expect("missing range-thumb hover rule");
        assert!(
            !thumb_hover.contains("transform"),
            "hovering a range cursor must not change its size"
        );
    }
}
