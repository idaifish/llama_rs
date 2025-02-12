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
        }

        unsafe {
            // disable log
            llama_cpp::llama_log_set(Some(disable_log_callback), ptr::null_mut());

            // model
            let model_params = llama_cpp::llama_model_default_params();
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
            let ctx = llama_cpp::llama_new_context_with_model(model, ctx_params);

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
                let embedding_ptr = llama_cpp::llama_get_embeddings_seq(ctx, 0);
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
                true,
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
