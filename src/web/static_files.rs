//! Embedded static files for the web control panel.
//! Files are included at compile time so the binary is self-contained.

use axum::http::{header, StatusCode};
use axum::response::{IntoResponse, Response};

const INDEX_HTML: &str = include_str!("../../static/index.html");
const STYLE_CSS: &str = include_str!("../../static/style.css");
const APP_JS: &str = include_str!("../../static/app.js");

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
