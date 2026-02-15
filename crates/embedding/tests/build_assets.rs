#[path = "../build_assets.rs"]
mod build_assets;

use std::fs;
use std::time::{SystemTime, UNIX_EPOCH};

fn unique_temp_dir(prefix: &str) -> std::path::PathBuf {
    let unique = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .expect("clock should be after unix epoch")
        .as_nanos();
    std::env::temp_dir().join(format!("{prefix}-{}-{unique}", std::process::id()))
}

#[test]
fn validate_model_dir_ok_when_all_files_exist() {
    let dir = unique_temp_dir("erio-embedding-build-assets-ok");
    fs::create_dir_all(dir.join("2_Dense")).unwrap();
    fs::create_dir_all(dir.join("3_Dense")).unwrap();

    for asset in build_assets::ASSETS {
        assert!(!asset.asset_name.is_empty());
        let path = dir.join(asset.relative_path);
        if let Some(parent) = path.parent() {
            fs::create_dir_all(parent).unwrap();
        }
        fs::write(path, b"").unwrap();
    }

    assert!(build_assets::has_all_assets(&dir));
    build_assets::validate_model_dir(&dir).unwrap();

    fs::remove_dir_all(&dir).unwrap();
}

#[test]
fn validate_model_dir_errors_when_any_file_missing() {
    let dir = unique_temp_dir("erio-embedding-build-assets-missing");
    fs::create_dir_all(&dir).unwrap();

    let err = build_assets::validate_model_dir(&dir).unwrap_err();
    assert!(err.contains("Missing required model file"), "{err}");
    assert!(!build_assets::has_all_assets(&dir));

    fs::remove_dir_all(&dir).unwrap();
}
