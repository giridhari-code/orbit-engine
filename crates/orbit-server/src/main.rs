//! Orbit Server — HTTP entry point for the Orbit Inference Engine.

use axum::{routing::get, Router};
use tracing_subscriber::EnvFilter;

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(EnvFilter::from_default_env().add_directive("info".parse()?))
        .init();

    let app = Router::new().route("/health", get(|| async { "orbit-engine: ok" }));

    let listener = tokio::net::TcpListener::bind("0.0.0.0:8080").await?;
    tracing::info!("Orbit server listening on http://0.0.0.0:8080");
    axum::serve(listener, app).await?;

    Ok(())
}
