use std::env;
use std::path::Path;
use std::time::Instant;

use llama_rs::LLM;

fn main() {
    env::set_var("GGML_CUDA_ENABLE_UNIFIED_MEMORY", "1");
    
    let start = Instant::now();

    let mut llm = LLM::new(Path::new("/tmp/qwen2.5-coder-3b.gguf"));
    let response = llm.completion("hi, who are you").unwrap();
    println!("response => {}", response.trim());

    println!("{:?} elapsed", start.elapsed());
}