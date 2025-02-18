use std::ffi::{c_char, c_void, CStr, CString};
use std::path::Path;
use std::ptr;

mod llama_cpp;

#[derive(Debug, Clone)]
pub struct LLMOptions {
    // model
    pub n_gpu_layers: i32,
    // context
    pub n_ctx: u32,
    pub n_max_tokens: usize,
    pub n_batch: u32,
    pub n_ubatch: u32,
    pub n_threads: i32,       // number of threads to use for generation
    pub n_threads_batch: i32, // number of threads to use for batch processing
    // sampling
    pub temperature: f32,
    pub seed: u32,
    pub top_k: i32,
    pub top_p: f32,
    pub min_p: f32,
    pub stop_words: Vec<String>,
    pub penalty_last_n: i32,
    pub penalty_repeat: f32,
    pub penalty_freq: f32,
    pub penalty_present: f32,
}

impl Default for LLMOptions {
    fn default() -> Self {
        Self {
            n_gpu_layers: 0,
            n_ctx: 0, // use context size from GGUF file.
            n_max_tokens: 512,
            n_batch: 512,
            n_ubatch: 512,
            n_threads: 1,
            n_threads_batch: 1,
            temperature: 0.8,
            seed: 1337,
            top_k: 40,
            top_p: 0.95,
            min_p: 0.05,
            stop_words: Vec::new(),
            penalty_last_n: 0, // (0 = disable penalty, -1 = context size)
            penalty_repeat: 1.0,
            penalty_freq: 0.0,
            penalty_present: 0.0,
        }
    }
}

impl LLMOptions {
    pub fn n_gpu_layers(&mut self, n_gpu_layers: i32) -> &mut Self {
        self.n_gpu_layers = n_gpu_layers;
        self
    }

    pub fn n_ctx(&mut self, n_ctx: u32) -> &mut Self {
        self.n_ctx = n_ctx;
        self
    }

    pub fn n_max_tokens(&mut self, n_max_tokens: usize) -> &mut Self {
        self.n_max_tokens = n_max_tokens;
        self
    }

    pub fn n_batch(&mut self, n_batch: u32) -> &mut Self {
        self.n_batch = n_batch;
        self
    }

    pub fn n_ubatch(&mut self, n_ubatch: u32) -> &mut Self {
        self.n_ubatch = n_ubatch;
        self
    }

    pub fn n_threads(&mut self, n_threads: i32) -> &mut Self {
        self.n_threads = n_threads;
        self
    }

    pub fn n_threads_batch(&mut self, n_threads_batch: i32) -> &mut Self {
        self.n_threads_batch = n_threads_batch;
        self
    }

    pub fn temperature(&mut self, temperature: f32) -> &mut Self {
        self.temperature = temperature;
        self
    }

    pub fn seed(&mut self, seed: u32) -> &mut Self {
        self.seed = seed;
        self
    }

    pub fn top_k(&mut self, top_k: i32) -> &mut Self {
        self.top_k = top_k;
        self
    }

    pub fn top_p(&mut self, top_p: f32) -> &mut Self {
        self.top_p = top_p;
        self
    }

    pub fn min_p(&mut self, min_p: f32) -> &mut Self {
        self.min_p = min_p;
        self
    }

    pub fn stop_words(&mut self, stop_words: Vec<String>) -> &mut Self {
        self.stop_words = stop_words;
        self
    }

    pub fn penalty_last_n(&mut self, penalty_last_n: i32) -> &mut Self {
        self.penalty_last_n = penalty_last_n;
        self
    }

    pub fn penalty_freq(&mut self, penalty_freq: f32) -> &mut Self {
        self.penalty_freq = penalty_freq;
        self
    }

    pub fn penalty_repeat(&mut self, penalty_repeat: f32) -> &mut Self {
        self.penalty_repeat = penalty_repeat;
        self
    }

    pub fn penalty_present(&mut self, penalty_present: f32) -> &mut Self {
        self.penalty_present = penalty_present;
        self
    }
}

#[derive(Debug, Clone)]
pub struct LLM {
    pub model: *mut llama_cpp::llama_model,
    pub ctx: *mut llama_cpp::llama_context,
    pub sampler: *mut llama_cpp::llama_sampler,
    options: LLMOptions,
}

impl LLM {
    pub fn new(model_path: &Path, options: LLMOptions) -> Self {
        unsafe extern "C" fn disable_log_callback(
            level: llama_cpp::ggml_log_level,
            text: *const c_char,
            user_data: *mut c_void,
        ) {
            if level == llama_cpp::ggml_log_level_GGML_LOG_LEVEL_ERROR {
                panic!("{:?}", CStr::from_ptr(text).to_str().unwrap());
            }
        }

        unsafe {
            // disable log
            if std::env::var("LLAMA_DEBUG").is_err() {
                llama_cpp::llama_log_set(Some(disable_log_callback), ptr::null_mut());
            }

            // model
            let mut model_params = llama_cpp::llama_model_default_params();
            model_params.n_gpu_layers = options.n_gpu_layers;
            let model = llama_cpp::llama_load_model_from_file(
                CString::new(model_path.to_str().unwrap()).unwrap().as_ptr(),
                model_params,
            );
            if model.is_null() {
                panic!("failed to load model from file");
            }

            // context
            let mut ctx_params = llama_cpp::llama_context_default_params();
            ctx_params.n_ctx = options.n_ctx;
            ctx_params.embeddings = false;
            ctx_params.no_perf = true;
            ctx_params.n_batch = options.n_batch;
            ctx_params.n_ubatch = options.n_ubatch;
            ctx_params.n_threads = options.n_threads;
            ctx_params.n_threads_batch = options.n_threads_batch;
            ctx_params.pooling_type = llama_cpp::llama_pooling_type_LLAMA_POOLING_TYPE_UNSPECIFIED;
            let ctx = llama_cpp::llama_init_from_model(model, ctx_params);

            if ctx.is_null() {
                panic!("failed to create llama context");
            }

            // sampler
            let mut sampler_params = llama_cpp::llama_sampler_chain_default_params();
            sampler_params.no_perf = true;
            let sampler = llama_cpp::llama_sampler_chain_init(sampler_params);
            llama_cpp::llama_sampler_chain_add(
                sampler,
                llama_cpp::llama_sampler_init_min_p(options.min_p, 1),
            );
            llama_cpp::llama_sampler_chain_add(
                sampler,
                llama_cpp::llama_sampler_init_top_p(options.top_p, 1),
            );
            llama_cpp::llama_sampler_chain_add(
                sampler,
                llama_cpp::llama_sampler_init_top_k(options.top_k),
            );
            llama_cpp::llama_sampler_chain_add(
                sampler,
                llama_cpp::llama_sampler_init_temp(options.temperature),
            );
            llama_cpp::llama_sampler_chain_add(
                sampler,
                llama_cpp::llama_sampler_init_penalties(
                    options.penalty_last_n,
                    options.penalty_repeat,
                    options.penalty_freq,
                    options.penalty_present,
                ),
            );
            llama_cpp::llama_sampler_chain_add(
                sampler,
                llama_cpp::llama_sampler_init_dist(options.seed),
            );

            Self {
                model,
                ctx,
                sampler,
                options,
            }
        }
    }

    fn chat_completions() {}

    pub fn completion(&mut self, prompt: &str) -> Result<String, &'static str> {
        let role = CString::new("user").unwrap();
        let content = CString::new(prompt).unwrap();

        let user_prompt = llama_cpp::llama_chat_message {
            role: role.as_ptr(),
            content: content.as_ptr(),
        };

        let mut formatted_msg = Vec::<c_char>::with_capacity(prompt.len() * 2 + 100);

        unsafe {
            let msg_len = llama_cpp::llama_chat_apply_template(
                ptr::null(),
                &user_prompt,
                1,
                false,
                formatted_msg.as_mut_ptr(),
                formatted_msg.capacity().try_into().unwrap(),
            );

            if msg_len < 0 {
                return Err("failed to apply chat template");
            }

            formatted_msg.set_len(msg_len.try_into().unwrap());
            let formatted_msg = formatted_msg.iter().map(|&b| b as u8).collect::<Vec<u8>>();
            match std::str::from_utf8(&formatted_msg) {
                Ok(msg) => {
                    let mut prompt_tokens = self.tokenize(msg).expect("failed to tokenize prompt");
                    let mut batch = llama_cpp::llama_batch_get_one(
                        prompt_tokens.as_mut_ptr(),
                        prompt_tokens.len().try_into().unwrap(),
                    );
                    let mut predicted = Vec::<i32>::new();
                    let n_ctx = llama_cpp::llama_n_ctx(self.ctx);
                    let vocab = llama_cpp::llama_model_get_vocab(self.model);

                    loop {
                        let n_ctx_used = llama_cpp::llama_get_kv_cache_used_cells(self.ctx);
                        if (n_ctx_used + batch.n_tokens > n_ctx.try_into().unwrap()) {
                            return Err("context size exceeded");
                        }

                        if (llama_cpp::llama_decode(self.ctx, batch) < 0) {
                            return Err("failed to decode user prompt");
                        }

                        let mut predicted_token =
                            llama_cpp::llama_sampler_sample(self.sampler, self.ctx, -1);
                        if llama_cpp::llama_vocab_is_eog(vocab, predicted_token) {
                            break;
                        } else {
                            predicted.push(predicted_token);

                            if predicted.len() >= self.options.n_max_tokens {
                                break;
                            }

                            let result = self.detokenize(predicted.clone()).unwrap();
                            if self.options.stop_words.iter().any(|s| result.ends_with(s)) {
                                break;
                            }
                        }

                        batch = llama_cpp::llama_batch_get_one(&mut predicted_token as *mut i32, 1)
                    }

                    self.detokenize(predicted)
                }
                Err(_) => return Err("failed to apply chat template"),
            }
        }
    }

    pub fn embed(&mut self, prompt: &str) -> Result<Vec<f32>, &'static str> {
        fn batch_decode(
            ctx: *mut llama_cpp::llama_context,
            batch: llama_cpp::llama_batch,
        ) -> Vec<f32> {
            unsafe {
                llama_cpp::llama_kv_cache_clear(ctx);

                // run
                let model = llama_cpp::llama_get_model(ctx);
                let n_embd = llama_cpp::llama_model_n_embd(model);
                if llama_cpp::llama_model_has_encoder(model)
                    && !llama_cpp::llama_model_has_decoder(model)
                {
                    if llama_cpp::llama_encode(ctx, batch) < 0 {
                        panic!("failed to encode batch");
                    }
                }

                if !llama_cpp::llama_model_has_encoder(model)
                    && llama_cpp::llama_model_has_decoder(model)
                {
                    if llama_cpp::llama_decode(ctx, batch) < 0 {
                        panic!("failed to decode batch");
                    }
                }

                // get embedding
                let embedding_ptr: *mut f32;
                if llama_cpp::llama_pooling_type(ctx)
                    != llama_cpp::llama_pooling_type_LLAMA_POOLING_TYPE_NONE
                {
                    embedding_ptr = llama_cpp::llama_get_embeddings_seq(ctx, 0);
                } else {
                    embedding_ptr = llama_cpp::llama_get_embeddings(ctx);
                }

                if embedding_ptr.is_null() {
                    return Vec::new();
                }
                return std::slice::from_raw_parts(embedding_ptr, n_embd.try_into().unwrap())
                    .to_vec();
            }
        }

        unsafe {
            // embedding mode
            llama_cpp::llama_set_embeddings(self.ctx, true);

            let mut tokens = self.tokenize(prompt).unwrap().clone();

            // truncate
            let n_ctx = llama_cpp::llama_model_n_ctx_train(self.model);
            if tokens.len() > n_ctx.try_into().unwrap() {
                tokens.truncate(n_ctx.try_into().unwrap());
            }

            let batch = llama_cpp::llama_batch_get_one(tokens.as_mut_ptr(), tokens.len() as i32);
            let embedding = batch_decode(self.ctx, batch);

            // normalize
            let euclidean_norm = |e: Vec<f32>| -> Vec<f32> {
                let mut sum = 0.0;
                for &v in e.iter() {
                    sum += v * v;
                }
                sum = sum.sqrt();
                let norm = if sum > 0.0 { 1.0 / sum } else { 0.0 };

                e.into_iter().map(|v| v * norm).collect()
            };

            Ok(euclidean_norm(embedding))
        }
    }

    pub fn tokenize(&mut self, prompt: &str) -> Result<Vec<i32>, &'static str> {
        unsafe {
            let vocab = llama_cpp::llama_model_get_vocab(self.model);
            let n_ctx = llama_cpp::llama_model_n_ctx_train(self.model);
            let text = CString::new(prompt).unwrap();
            let mut tokens =
                Vec::<llama_cpp::llama_token>::with_capacity(n_ctx.try_into().unwrap());
            let n_tokens = llama_cpp::llama_tokenize(
                vocab,
                text.as_ptr(),
                text.as_bytes().len() as i32,
                tokens.as_mut_ptr(),
                n_ctx,
                llama_cpp::llama_vocab_get_add_bos(vocab),
                false,
            );

            if n_tokens > 0 {
                tokens.set_len(n_tokens.try_into().unwrap());
                Ok(tokens.into_iter().map(|t| t as i32).collect())
            } else {
                Err("failed to tokenize prompt")
            }
        }
    }

    pub fn detokenize(&mut self, tokens: Vec<i32>) -> Result<String, &'static str> {
        unsafe {
            let vocab = llama_cpp::llama_model_get_vocab(self.model);
            let n_ctx = llama_cpp::llama_model_n_ctx_train(self.model);
            let n_tokens = tokens.len();
            let text_len_max = n_tokens * 5;
            let mut text = Vec::<c_char>::with_capacity(text_len_max + 1);

            let n_text = llama_cpp::llama_detokenize(
                vocab,
                tokens.as_ptr(),
                tokens.len() as i32,
                text.as_mut_ptr(),
                text.capacity() as i32 - 1,
                true,
                false,
            );

            if n_text >= 0 {
                text.set_len(n_text.try_into().unwrap());
                let text = text.iter().map(|&b| b as u8).collect::<Vec<u8>>();
                match std::str::from_utf8(&text) {
                    Ok(text_str) => Ok(text_str.to_string()),
                    Err(e) => Err("Invalid utf-8 sequences"),
                }
            } else {
                Err("failed to detokenize tokens")
            }
        }
    }
}

impl Drop for LLM {
    fn drop(&mut self) {
        unsafe {
            // if !self.ctx.is_null() {
            //     llama_cpp::llama_free(self.ctx);
            // }

            if !self.model.is_null() {
                llama_cpp::llama_model_free(self.model);
            }

            if !self.sampler.is_null() {
                llama_cpp::llama_sampler_free(self.sampler);
            }
        }
    }
}
