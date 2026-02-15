use std::path::Path;

pub(crate) struct Asset {
    pub(crate) asset_name: &'static str,
    pub(crate) relative_path: &'static str,
}

pub(crate) const ASSETS: &[Asset] = &[
    Asset {
        asset_name: "embeddinggemma-300M-Q8_0.gguf",
        relative_path: "embeddinggemma-300M-Q8_0.gguf",
    },
    Asset {
        asset_name: "tokenizer.json",
        relative_path: "tokenizer.json",
    },
    Asset {
        asset_name: "2_Dense_model.safetensors",
        relative_path: "2_Dense/model.safetensors",
    },
    Asset {
        asset_name: "3_Dense_model.safetensors",
        relative_path: "3_Dense/model.safetensors",
    },
];

pub(crate) fn validate_model_dir(model_dir: &Path) -> Result<(), String> {
    for asset in ASSETS {
        let expected = model_dir.join(asset.relative_path);
        if !expected.exists() {
            return Err(format!(
                "Missing required model file: {}\nSet ERIO_MODEL_DIR to a directory containing all model files.",
                expected.display()
            ));
        }
    }
    Ok(())
}

pub(crate) fn has_all_assets(model_dir: &Path) -> bool {
    ASSETS
        .iter()
        .all(|asset| model_dir.join(asset.relative_path).exists())
}
