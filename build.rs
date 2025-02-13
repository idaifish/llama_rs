use std::env;
use std::path::PathBuf;

fn main() {
    println!("cargo:rerun-if-changed=build.rs");

    let out_dir = PathBuf::from(env::var("OUT_DIR").unwrap());
    let llama_cpp_src = PathBuf::from(env::var("CARGO_MANIFEST_DIR").unwrap()).join("llama.cpp");

    let _binding = bindgen::Builder::default()
        .header("llama.cpp/include/llama.h")
        .clang_arg(format!(
            "-I{}",
            llama_cpp_src.join("include").as_path().display()
        ))
        .clang_arg(format!(
            "-I{}",
            llama_cpp_src.join("ggml/include").as_path().display()
        ))
        .parse_callbacks(Box::new(bindgen::CargoCallbacks::new()))
        .allowlist_function("llama_.*")
        .allowlist_type("llama_.*")
        .generate()
        .expect("Unable to generate binding to llama.cpp")
        .write_to_file(out_dir.join("binding.rs"))
        .expect("Failed to write binding");

    let mut llamalib_conf = cmake::Config::new("llama.cpp");
    llamalib_conf.define("BUILD_SHARED_LIBS", "OFF");

    if cfg!(feature = "openblas") {
        llamalib_conf.define("GGML_BLAS", "ON");
        llamalib_conf.define("GGML_BLAS_VENDOR", "OpenBLAS");
        println!("cargo::rustc-link-lib=dylib=openblas");
        println!("cargo::rustc-link-lib=static=ggml-blas");
    }

    if cfg!(feature = "cuda") {
        llamalib_conf.define("GGML_CUDA", "ON");
        println!("cargo::rustc-link-lib=dylib=cuda");
        println!("cargo::rustc-link-lib=dylib=cudart");
        println!("cargo::rustc-link-lib=dylib=cublas");
        println!("cargo::rustc-link-lib=static=ggml-cuda");
    }

    let llamalib_dst= llamalib_conf.build();

    println!("cargo::rustc-link-search=native={}/lib", llamalib_dst.display());
    println!("cargo::rustc-link-lib=dylib=stdc++");
    println!("cargo::rustc-link-lib=dylib=gomp");
    println!("cargo::rustc-link-lib=static=ggml");
    println!("cargo::rustc-link-lib=static=ggml-cpu");
    println!("cargo::rustc-link-lib=static=ggml-base");
    println!("cargo::rustc-link-lib=static=llama");
}
