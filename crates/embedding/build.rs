use std::env;
use std::fs;
use std::io;
use std::path::{Path, PathBuf};

mod build_assets;

const DEFAULT_RELEASE_BASE_URL: &str =
    "https://github.com/NomanworkGroup/erio/releases/download/models-v1";

fn main() {
    println!("cargo:rerun-if-changed=build.rs");
    println!("cargo:rerun-if-env-changed=ERIO_MODEL_DIR");
    println!("cargo:rerun-if-env-changed=ERIO_MODEL_RELEASE_BASE_URL");
    println!("cargo:rerun-if-env-changed=DOCS_RS");
    println!("cargo:rerun-if-env-changed=CARGO_CFG_DOCSRS");

    if is_docs_build() {
        let out_dir = PathBuf::from(env::var("OUT_DIR").expect("OUT_DIR must be set"));
        let dummy_dir = out_dir.join("docsrs-model-dummy");
        fs::create_dir_all(&dummy_dir).unwrap_or_else(|err| panic!("{err}"));
        println!("cargo:rustc-env=ERIO_MODEL_DIR={}", dummy_dir.display());
        return;
    }

    let model_dir = if let Ok(path) = env::var("ERIO_MODEL_DIR") {
        let path = PathBuf::from(path);
        build_assets::validate_model_dir(&path).unwrap_or_else(|err| panic!("{err}"));
        path
    } else {
        let out_dir = PathBuf::from(env::var("OUT_DIR").expect("OUT_DIR must be set"));
        let model_dir = out_dir.join("models");
        ensure_model_assets(&model_dir).unwrap_or_else(|err| panic!("{err}"));
        model_dir
    };

    println!("cargo:rustc-env=ERIO_MODEL_DIR={}", model_dir.display());
}

fn ensure_model_assets(model_dir: &Path) -> Result<(), String> {
    fs::create_dir_all(model_dir).map_err(|e| format!("failed to create model directory: {e}"))?;

    if build_assets::has_all_assets(model_dir) {
        return Ok(());
    }

    let base_url = env::var("ERIO_MODEL_RELEASE_BASE_URL")
        .unwrap_or_else(|_| DEFAULT_RELEASE_BASE_URL.to_owned());

    for asset in build_assets::ASSETS {
        let dest = model_dir.join(asset.relative_path);
        if dest.exists() {
            continue;
        }

        if let Some(parent) = dest.parent() {
            fs::create_dir_all(parent).map_err(|e| {
                format!(
                    "failed to create parent directory {}: {e}",
                    parent.display()
                )
            })?;
        }

        let url = format!("{base_url}/{}", asset.asset_name);
        download_file(&url, &dest).map_err(|e| {
            format!(
                "Failed to download embedding model files from GitHub Release assets.\n\nThis usually means:\n- No internet connection during build\n- The release assets haven't been published yet (maintainer action needed)\n\nFor maintainers: run the model-publish CI workflow first.\nFor offline builds: set ERIO_MODEL_DIR to a directory containing the model files.\n\nDownload error ({url}): {e}"
            )
        })?;
    }

    build_assets::validate_model_dir(model_dir)
}

fn is_docs_build() -> bool {
    env::var_os("DOCS_RS").as_deref() == Some("1".as_ref())
        || env::var_os("CARGO_CFG_DOCSRS").is_some()
}

fn download_file(url: &str, dest: &Path) -> Result<(), io::Error> {
    let response = ureq::get(url)
        .call()
        .map_err(|e| io::Error::other(format!("request failed: {e}")))?;

    let mut reader = response.into_reader();
    let mut writer = fs::File::create(dest)?;
    io::copy(&mut reader, &mut writer)?;
    Ok(())
}
