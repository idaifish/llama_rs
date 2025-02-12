use std::path::Path;
use std::time::Instant;

use llama_rs::LLM;

fn dot_product(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b.iter()).map(|(&x, &y)| x * y).sum()
}

fn magnitude(v: &[f32]) -> f32 {
    (v.iter().map(|&x| x.powi(2)).sum::<f32>()).sqrt()
}

fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
    dot_product(a, b) / (magnitude(a) * magnitude(b))
}

fn main() {
    let start = Instant::now();
    let mut llm = LLM::new(Path::new("/tmp/nomic-embed-text.gguf"));

    let tokens = llm.tokenize("hello llama, hello llama.cpp").unwrap();
    println!("tokenize: {:?}", tokens);

    let text = llm.detokenize(tokens).unwrap();
    println!("detokenize: {}", text);

    let embedding1 = llm.embed("i love Rust").unwrap();
    println!("Embedding1: {:?}", &embedding1);

    let embedding2 = llm.embed("Python is the best").unwrap();
    println!("Embedding2: {:?}", &embedding2);

    let similarity = cosine_similarity(&embedding1, &embedding2);
    println!("cosine similarity: {}", similarity);

    println!("{:?} elapsed", start.elapsed());
}