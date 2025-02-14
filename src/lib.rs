use std::ffi::{c_char, c_void, CStr, CString};
use std::path::Path;
use std::ptr::{self, slice_from_raw_parts};

mod llama_cpp;

#[derive(Debug, Clone)]
pub struct LLM {
    pub model: *mut llama_cpp::llama_model,
    pub ctx: *mut llama_cpp::llama_context,
    pub sampler: *mut llama_cpp::llama_sampler,
}

impl LLM {
    pub fn new(model_path: &Path) -> Self {
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
            llama_cpp::llama_log_set(Some(disable_log_callback), ptr::null_mut());

            // model
            let mut model_params = llama_cpp::llama_model_default_params();
            if cfg!(feature = "cuda") {
                model_params.n_gpu_layers = 99;
            }

            let model = llama_cpp::llama_load_model_from_file(
                CString::new(model_path.to_str().unwrap()).unwrap().as_ptr(),
                model_params,
            );

            // context
            let mut ctx_params = llama_cpp::llama_context_default_params();
            ctx_params.n_ctx = 0; // use context size from GGUF file.
            ctx_params.embeddings = false;
            ctx_params.no_perf = true;
            ctx_params.n_batch = 2048;
            ctx_params.n_ubatch = ctx_params.n_batch;
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
                llama_cpp::llama_sampler_init_min_p(0.05f32, 1),
            );
            llama_cpp::llama_sampler_chain_add(sampler, llama_cpp::llama_sampler_init_temp(0.8f32));
            llama_cpp::llama_sampler_chain_add(sampler, llama_cpp::llama_sampler_init_dist(1337));

            Self {
                model,
                ctx,
                sampler,
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

                        let mut predicted_token = llama_cpp::llama_sampler_sample(self.sampler, self.ctx, -1);
                        if llama_cpp::llama_token_is_eog(vocab, predicted_token) {
                            break;
                        } else {
                            predicted.push(predicted_token);
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
                if llama_cpp::llama_pooling_type(ctx) != llama_cpp::llama_pooling_type_LLAMA_POOLING_TYPE_NONE {
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

            if n_text > 0 {
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
