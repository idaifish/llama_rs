use std::env;
use std::path::Path;
use std::time::Instant;

use llama_rs::{LLMOptions, LLM};

fn main() {
    env::set_var("GGML_CUDA_ENABLE_UNIFIED_MEMORY", "1");

    let start = Instant::now();

    let options = LLMOptions {
        n_ctx: 4096,
        n_threads: 8,
        n_threads_batch: 8,
        temperature: 0.1,
        stop_words: vec![String::from(".")],
        ..Default::default()
    };
    println!("LLM Options => {:#?}", &options);
    let mut llm = LLM::new(Path::new("/tmp/qwen2.5-coder-3b.gguf"), options);
    let response = llm.completion("hi, who are you").unwrap();
    println!("response => {}", response);

    println!("{:?} elapsed", start.elapsed());
}
