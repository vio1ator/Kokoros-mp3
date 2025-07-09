//! OpenAI-compatible TTS HTTP server for Kokoros
//!
//! This module provides an HTTP API that is compatible with OpenAI's text-to-speech endpoints.
//! It implements streaming and non-streaming audio generation with multiple format support.
//!
//! ## Implemented Features
//! - `/v1/audio/speech` - Text-to-speech generation with streaming support
//! - `/v1/audio/voices` - List available voices
//! - `/v1/models` - List available models (static dummy list)
//! - Multiple audio formats: MP3, WAV, PCM, OPUS, AAC, FLAC
//! - Streaming audio generation for low-latency responses
//!
//! ## OpenAI API Compatibility Limitations
//! - `return_download_link`: Not implemented (files are streamed directly)
//! - `lang_code`: Not implemented (language auto-detected from voice prefix)
//! - `volume_multiplier`: Not implemented (audio returned at original levels)
//! - `download_format`: Not implemented (only response_format used)
//! - `normalization_options`: Not implemented (basic text processing only)
//! - Streaming only supports PCM format (other formats fall back to PCM)

use std::error::Error;
use std::io;
use std::sync::Arc;
use std::time::Instant;

use axum::{
    body::Body,
    extract::{Path, State},
    http::{header, StatusCode},
    response::{IntoResponse, Response},
    routing::{get, post},
    Json, Router,
};
use futures::stream::StreamExt;
use kokoros::{
    tts::koko::{InitConfig as TTSKokoInitConfig, TTSKoko},
    utils::mp3::pcm_to_mp3,
    utils::wav::{write_audio_chunk, WavHeader},
};
use serde::{Deserialize, Serialize};
use tokio::sync::mpsc;
use tower_http::cors::CorsLayer;
use uuid::Uuid;


#[derive(Deserialize, Default, Debug)]
#[serde(rename_all = "lowercase")]
enum AudioFormat {
    #[default]
    Mp3,
    Wav,
    Opus,
    Aac,
    Flac,
    Pcm,
}

#[derive(Deserialize)]
struct Voice(String);

impl Default for Voice {
    fn default() -> Self {
        Self("af_sky".into())
    }
}

#[derive(Deserialize)]
struct Speed(f32);

impl Default for Speed {
    fn default() -> Self {
        Self(1.)
    }
}

#[derive(Deserialize)]
struct SpeechRequest {
    // Only one Kokoro model exists
    #[allow(dead_code)]
    model: String,

    input: String,

    #[serde(default)]
    voice: Voice,

    #[serde(default)]
    response_format: AudioFormat,

    #[serde(default)]
    speed: Speed,

    #[serde(default)]
    initial_silence: Option<usize>,

    /// Enable streaming audio generation (implemented)
    #[serde(default)]
    stream: Option<bool>,

    // OpenAI API compatibility parameters - accepted but not implemented
    // These fields ensure request parsing compatibility with OpenAI clients
    
    /// Return download link after generation (not implemented)
    #[serde(default)]
    #[allow(dead_code)]
    return_download_link: Option<bool>,
    
    /// Language code for text processing (not implemented)
    #[serde(default)]
    #[allow(dead_code)]
    lang_code: Option<String>,
    
    /// Volume multiplier for output audio (not implemented)
    #[serde(default)]
    #[allow(dead_code)]
    volume_multiplier: Option<f32>,
    
    /// Format for download when different from response_format (not implemented)
    #[serde(default)]
    #[allow(dead_code)]
    download_format: Option<String>,
    
    /// Text normalization options (not implemented)
    #[serde(default)]
    #[allow(dead_code)]
    normalization_options: Option<serde_json::Value>,
}

/// Ordered chunk for async streaming
#[derive(Debug, Clone)]
struct OrderedChunk {
    id: usize,
    text: String,
    audio_data: Option<Vec<u8>>,
    processing_time: Option<std::time::Duration>,
}

/// Async TTS worker task
#[derive(Debug)]
struct TTSTask {
    id: usize,
    chunk: String,
    voice: String,
    speed: f32,
    initial_silence: Option<usize>,
    result_tx: mpsc::UnboundedSender<OrderedChunk>,
}

/// Streaming session manager
#[derive(Debug)]
struct StreamingSession {
    session_id: Uuid,
    chunk_count: usize,
    start_time: Instant,
}

/// TTS worker pool manager
#[derive(Clone)]
struct TTSWorkerPool {
    tts: Arc<TTSKoko>,
    tts_lock: Arc<tokio::sync::Mutex<()>>,
    max_concurrent: usize,
}

impl TTSWorkerPool {
    fn new(tts: TTSKoko, max_concurrent: usize) -> Self {
        Self {
            tts: Arc::new(tts),
            tts_lock: Arc::new(tokio::sync::Mutex::new(())),
            max_concurrent,
        }
    }

    async fn process_chunk(&self, task: TTSTask) {
        let tts_clone = Arc::clone(&self.tts);
        let tts_lock = Arc::clone(&self.tts_lock);
        let start_time = Instant::now();
        
        // Clone data for the closure
        let chunk_text = task.chunk.clone();
        let voice = task.voice.clone();
        let speed = task.speed;
        let initial_silence = task.initial_silence;
        let chunk_id = task.id;
        
        eprintln!("Starting TTS processing for chunk {} with text: '{}'", chunk_id, &chunk_text);
        
        // Run TTS inference in a blocking thread with comprehensive error handling
        // Actually use the tts_lock to prevent concurrent model access
        let result = tokio::task::spawn_blocking(move || {
            // Introduce small delay based on chunk ID to stagger starts
            if chunk_id > 0 {
                std::thread::sleep(std::time::Duration::from_millis(chunk_id as u64 * 40));
            }
            
            // Acquire the lock and process TTS - remove panic handler to avoid UnwindSafe issues
            let _lock_guard = tts_lock.blocking_lock();
            eprintln!("Acquired TTS lock for chunk {}", chunk_id);
            
            eprintln!("Calling tts_raw_audio for chunk {}", chunk_id);
            let audio_result = tts_clone.tts_raw_audio(&chunk_text, "en-us", &voice, speed, initial_silence);
            eprintln!("Completed tts_raw_audio call for chunk {}", chunk_id);
            
            audio_result.map(|audio| {
                eprintln!("TTS success for chunk {}: {} samples", chunk_id, audio.len());
                audio
            }).map_err(|e| {
                eprintln!("TTS error for chunk {}: {:?}", chunk_id, e);
                format!("TTS processing error: {:?}", e)
            })
        }).await;
        
        let processing_time = start_time.elapsed();
        
        let mut ordered_chunk = OrderedChunk {
            id: task.id,
            text: task.chunk,
            audio_data: None,
            processing_time: Some(processing_time),
        };
        
        match result {
            Ok(Ok(audio_samples)) => {
                // Convert f32 samples to 16-bit PCM
                let mut pcm_data = Vec::with_capacity(audio_samples.len() * 2);
                for sample in audio_samples {
                    let pcm_sample = (sample * 32767.0).clamp(-32768.0, 32767.0) as i16;
                    pcm_data.extend_from_slice(&pcm_sample.to_le_bytes());
                }
                ordered_chunk.audio_data = Some(pcm_data);
            }
            Ok(Err(e)) => {
                eprintln!("TTS processing error for chunk {}: {}", task.id, e);
            }
            Err(e) => {
                eprintln!("Task execution error for chunk {}: {:?}", task.id, e);
            }
        }
        
        // Send result back
        if let Err(_) = task.result_tx.send(ordered_chunk) {
            eprintln!("Failed to send result for chunk {}", task.id);
        }
    }
}

#[derive(Serialize)]
struct VoicesResponse {
    voices: Vec<String>,
}

#[derive(Serialize)]
struct ModelObject {
    id: String,
    object: String,
    created: u64,
    owned_by: String,
}

#[derive(Serialize)]
struct ModelsResponse {
    object: String,
    data: Vec<ModelObject>,
}

pub async fn create_server(tts: TTSKoko) -> Router {
    println!("create_server()");

    Router::new()
        .route("/", get(handle_home))
        .route("/v1/audio/speech", post(handle_tts))
        .route("/v1/audio/voices", get(handle_voices))
        .route("/v1/models", get(handle_models))
        .route("/v1/models/{model}", get(handle_model))
        .layer(CorsLayer::permissive())
        .with_state(tts)
}

pub use axum::serve;

#[derive(Debug)]
enum SpeechError {
    // Deciding to modify this example in order to see errors
    // (e.g. with tracing) is up to the developer
    #[allow(dead_code)]
    Koko(Box<dyn Error>),

    #[allow(dead_code)]
    Header(io::Error),

    #[allow(dead_code)]
    Chunk(io::Error),

    #[allow(dead_code)]
    Mp3Conversion(std::io::Error),
}

impl std::fmt::Display for SpeechError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            SpeechError::Koko(e) => write!(f, "Koko TTS error: {}", e),
            SpeechError::Header(e) => write!(f, "Header error: {}", e),
            SpeechError::Chunk(e) => write!(f, "Chunk error: {}", e),
            SpeechError::Mp3Conversion(e) => write!(f, "MP3 conversion error: {}", e),
        }
    }
}

impl IntoResponse for SpeechError {
    fn into_response(self) -> Response {
        // None of these errors make sense to expose to the user of the API
        StatusCode::INTERNAL_SERVER_ERROR.into_response()
    }
}

/// Returns a 200 OK response to make it easier to check if the server is
/// running.
async fn handle_home() -> &'static str {
    "OK"
}

async fn handle_tts(
    State(tts): State<TTSKoko>,
    request: axum::extract::Request,
) -> Result<Response, SpeechError> {
    // Check if this is a streaming request by examining headers
    // The OpenAI SDK's with_streaming_response may set specific headers
    let headers = request.headers();
    let is_streaming_request = 
        // Check Accept header for streaming indicators
        headers.get("accept")
            .and_then(|v| v.to_str().ok())
            .map(|accept| accept.contains("text/event-stream") || 
                         accept.contains("application/octet-stream") ||
                         accept.contains("*/*"))  // OpenAI SDK often uses */*
            .unwrap_or(false)
        ||
        // Check for Transfer-Encoding expectations
        headers.get("te")
            .and_then(|v| v.to_str().ok())
            .map(|te| te.contains("chunked"))
            .unwrap_or(false)
        ||
        // Check Connection header for streaming indicators
        headers.get("connection")
            .and_then(|v| v.to_str().ok())
            .map(|conn| conn.contains("keep-alive"))
            .unwrap_or(false);
    
    // Log all headers for debugging
    eprintln!("=== Request Headers Debug ===");
    for (name, value) in headers.iter() {
        if let Ok(value_str) = value.to_str() {
            eprintln!("  {}: {}", name, value_str);
        }
    }
    eprintln!("  Streaming detected: {}", is_streaming_request);
    eprintln!("=============================");
    eprintln!("About to parse request body...");
    
    // Parse the JSON body
    let bytes = axum::body::to_bytes(request.into_body(), usize::MAX)
        .await
        .map_err(|e| {
            eprintln!("Error reading request body: {:?}", e);
            SpeechError::Mp3Conversion(std::io::Error::new(std::io::ErrorKind::InvalidInput, e))
        })?;
    
    eprintln!("Successfully read {} bytes from request body", bytes.len());
    eprintln!("Request body content: {}", String::from_utf8_lossy(&bytes));
    eprintln!("About to parse JSON...");
    
    let speech_request: SpeechRequest = serde_json::from_slice(&bytes)
        .map_err(|e| {
            eprintln!("JSON parsing error: {:?}", e);
            SpeechError::Mp3Conversion(std::io::Error::new(std::io::ErrorKind::InvalidInput, e))
        })?;

    eprintln!("Successfully parsed JSON request");

    let SpeechRequest {
        input,
        voice: Voice(voice),
        response_format,
        speed: Speed(speed),
        initial_silence,
        stream,
        ..
    } = speech_request;

    eprintln!("Successfully destructured request: input='{}', voice='{}', stream={:?}", input, voice, stream);

    // Debug streaming decision
    eprintln!("Streaming decision: stream={:?}, is_streaming_request={}", stream, is_streaming_request);
    
    // Use async streaming only when:
    // 1. Explicitly requested via "stream": true, OR
    // 2. Voice-mode style streaming request (detected via specific headers)
    let should_stream = stream.unwrap_or(false) || is_streaming_request;
    
    eprintln!("Final should_stream decision: {}", should_stream);
    
    if should_stream {
        eprintln!("Using async streaming: explicit={}, detected_streaming_headers={}", 
                  stream.unwrap_or(false), is_streaming_request);
        return handle_tts_streaming(tts, input, voice, response_format, speed, initial_silence).await;
    }

    // Non-streaming mode (existing implementation)
    let raw_audio = tts
        .tts_raw_audio(&input, "en-us", &voice, speed, initial_silence)
        .map_err(SpeechError::Koko)?;

    let sample_rate = TTSKokoInitConfig::default().sample_rate;

    let (content_type, audio_data) = match response_format {
        AudioFormat::Wav => {
            let mut wav_data = Vec::default();
            let header = WavHeader::new(1, sample_rate, 32);
            header
                .write_header(&mut wav_data)
                .map_err(SpeechError::Header)?;
            write_audio_chunk(&mut wav_data, &raw_audio).map_err(SpeechError::Chunk)?;

            ("audio/wav", wav_data)
        }
        AudioFormat::Mp3 => {
            let mp3_data =
                pcm_to_mp3(&raw_audio, sample_rate).map_err(|e| SpeechError::Mp3Conversion(e))?;

            ("audio/mpeg", mp3_data)
        }
        AudioFormat::Pcm => {
            // For PCM, we return the raw audio data directly
            // Convert f32 samples to 16-bit PCM
            let mut pcm_data = Vec::with_capacity(raw_audio.len() * 2);
            for sample in raw_audio {
                let pcm_sample = (sample * 32767.0).clamp(-32768.0, 32767.0) as i16;
                pcm_data.extend_from_slice(&pcm_sample.to_le_bytes());
            }
            ("audio/pcm", pcm_data)
        }
        // For now, unsupported formats fall back to MP3
        _ => {
            let mp3_data =
                pcm_to_mp3(&raw_audio, sample_rate).map_err(|e| SpeechError::Mp3Conversion(e))?;

            ("audio/mpeg", mp3_data)
        }
    };

    Ok(Response::builder()
        .header(header::CONTENT_TYPE, content_type)
        .body(audio_data.into())
        .map_err(|e| {
            SpeechError::Mp3Conversion(std::io::Error::new(std::io::ErrorKind::Other, e))
        })?)
}

/// Handle streaming TTS requests with true async processing
/// 
/// Uses micro-chunking and parallel processing for low-latency streaming.
/// Maintains speech order while allowing out-of-order chunk completion.
async fn handle_tts_streaming(
    tts: TTSKoko,
    input: String,
    voice: String,
    response_format: AudioFormat,
    speed: f32,
    initial_silence: Option<usize>,
) -> Result<Response, SpeechError> {
    // Streaming implementation: PCM format for optimal performance
    let content_type = match response_format {
        AudioFormat::Pcm => "audio/pcm",
        _ => "audio/pcm", // Force PCM for optimal streaming performance
    };

    // Create worker pool with controlled concurrency
    // Using staggered starts to prevent memory corruption while allowing parallelism
    let worker_pool = TTSWorkerPool::new(tts.clone(), 4); // Increased to 4 concurrent workers for faster processing
    
    // Only chunk if message is longer than 500 words
    let word_count = input.split_whitespace().count();
    let chunks = if word_count < 500 {
        vec![input.clone()]
    } else {
        // Create speech chunks based on word count and punctuation
        tts.split_text_into_speech_chunks(&input, 10) // ~10 words per chunk for faster streaming
    };
    let total_chunks = chunks.len();
    
    eprintln!("Input text: '{}'", input);
    eprintln!("Generated {} chunks:", total_chunks);
    for (i, chunk) in chunks.iter().enumerate() {
        eprintln!("  Chunk {}: '{}'", i, chunk);
    }
    
    if chunks.is_empty() {
        return Err(SpeechError::Mp3Conversion(std::io::Error::new(
            std::io::ErrorKind::InvalidInput,
            "No text to process",
        )));
    }

    // Create channels for chunk processing
    let (result_tx, mut result_rx) = mpsc::unbounded_channel::<OrderedChunk>();
    let (audio_tx, audio_rx) = mpsc::unbounded_channel::<(usize, Vec<u8>)>(); // Tag chunks with order ID
    
    // Create session for tracking
    let session = StreamingSession {
        session_id: Uuid::new_v4(),
        chunk_count: total_chunks,
        start_time: Instant::now(),
    };
    
    eprintln!("Starting async streaming session {} with {} chunks", session.session_id, total_chunks);

    // Create semaphore to enforce max_concurrent limit
    let semaphore = Arc::new(tokio::sync::Semaphore::new(worker_pool.max_concurrent));

    // Spawn worker tasks for all chunks with proper concurrency control
    for (id, chunk) in chunks.into_iter().enumerate() {
        let task = TTSTask {
            id,
            chunk,
            voice: voice.clone(),
            speed,
            initial_silence: if id == 0 { initial_silence } else { None },
            result_tx: result_tx.clone(),
        };
        
        let worker_pool_clone = worker_pool.clone();
        let semaphore_clone = Arc::clone(&semaphore);
        tokio::spawn(async move {
            // Acquire semaphore permit to limit concurrency
            let _permit = semaphore_clone.acquire().await.unwrap();
            let task_id = task.id; // Extract ID before moving task
            eprintln!("Acquired semaphore permit for chunk {}", task_id);
            worker_pool_clone.process_chunk(task).await;
            eprintln!("Released semaphore permit for chunk {}", task_id);
        });
    }
    
    // Drop the main result_tx to allow shutdown
    drop(result_tx);

    // Spawn ordering and streaming task
    let audio_tx_clone = audio_tx.clone();
    tokio::spawn(async move {
        let mut completed_chunks = std::collections::HashMap::new();
        let mut next_chunk_id = 0;
        let mut total_processed = 0;
        
        // Collect results and stream in order
        // Start streaming when 40% of text length is processed
        let input_length = input.len();
        let early_streaming_threshold = (input_length as f32 * 0.40) as usize;
        let mut early_streaming_started = false;
        
        while let Some(chunk) = result_rx.recv().await {
            total_processed += 1;
            
            eprintln!("Received chunk {} (processed: {}/{})", chunk.id, total_processed, total_chunks);
            
            if let Some(ref processing_time) = chunk.processing_time {
                eprintln!("Chunk {} processed in {:?}", chunk.id, processing_time);
            } else {
                eprintln!("Chunk {} had no processing time (likely error)", chunk.id);
            }
            
            if chunk.audio_data.is_none() {
                eprintln!("WARNING: Chunk {} has no audio data", chunk.id);
            }
            
            // Extract chunk_id before moving the chunk
            let chunk_id = chunk.id;
            completed_chunks.insert(chunk_id, chunk);
            
            // Start early streaming if we have enough text processed
            let processed_text_length: usize = completed_chunks.values()
                .map(|c| c.text.len())
                .sum();
            
            if !early_streaming_started && processed_text_length >= early_streaming_threshold {
                eprintln!("Early streaming triggered at {}% text completion ({}/{} chars)", 
                         (processed_text_length as f32 / input_length as f32 * 100.0) as u32,
                         processed_text_length, input_length);
                early_streaming_started = true;
            }
            
            // Send all consecutive chunks that are ready, starting from next_chunk_id
            while let Some(ready_chunk) = completed_chunks.remove(&next_chunk_id) {
                eprintln!("Streaming chunk {} in order immediately (text: '{}')", ready_chunk.id, &ready_chunk.text[..ready_chunk.text.len().min(30)]);
                
                if let Some(audio_data) = ready_chunk.audio_data {
                    if audio_data.is_empty() {
                        eprintln!("WARNING: Chunk {} has empty audio data", ready_chunk.id);
                    } else {
                        eprintln!("Sending audio data for chunk {} immediately: {} bytes", ready_chunk.id, audio_data.len());
                        // Send chunk with its actual ID for proper ordering
                        match audio_tx_clone.send((ready_chunk.id, audio_data)) {
                            Ok(_) => eprintln!("Successfully sent chunk {} to HTTP stream", ready_chunk.id),
                            Err(_) => {
                                eprintln!("Failed to send audio chunk {} - receiver closed", ready_chunk.id);
                                return;
                            }
                        }
                    }
                } else {
                    eprintln!("Skipping chunk {} due to missing audio data", ready_chunk.id);
                }
                next_chunk_id += 1;
            }
            
            // Only log buffering if the chunk is still in the buffer (not sent immediately)
            if completed_chunks.contains_key(&chunk_id) {
                eprintln!("Buffering chunk {} - waiting for chunk {} to complete first", chunk_id, next_chunk_id);
            }
            
            // Check if we've processed all chunks
            if total_processed >= total_chunks {
                eprintln!("All chunks processed, exiting ordering loop");
                break;
            }
        }
        
        let session_time = session.start_time.elapsed();
        eprintln!("Async streaming session {} completed in {:?}", session.session_id, session_time);
        
        // Send termination signal and close the audio stream
        eprintln!("All chunks sent to HTTP stream, sending termination signal");
        // Send a special termination marker with empty data to signal end of stream
        let _ = audio_tx_clone.send((total_chunks, vec![])); // Empty data as termination signal
        drop(audio_tx_clone);
    });

    // Create immediate streaming - chunks are already sent in order from TTS processing
    let stream = tokio_stream::wrappers::UnboundedReceiverStream::new(audio_rx)
        .map(|(chunk_id, data)| -> Result<Vec<u8>, std::io::Error> {
            // Check for termination signal (empty data)
            if data.is_empty() {
                eprintln!("HTTP stream received termination signal (chunk {}) - ending stream", chunk_id);
                return Err(std::io::Error::new(std::io::ErrorKind::UnexpectedEof, "Stream complete"));
            }
            
            eprintln!("HTTP stream delivering chunk {} immediately: {} bytes", chunk_id, data.len());
            Ok(data)
        })
        .take_while(|result| {
            // Continue until we hit an error (termination signal)
            std::future::ready(result.is_ok())
        });

    // Convert to HTTP body with explicit ordering
    let body = Body::from_stream(stream);

    Ok(Response::builder()
        .header(header::CONTENT_TYPE, content_type)
        .header(header::CONNECTION, "keep-alive")
        .header(header::CACHE_CONTROL, "no-cache")
        .header("X-Accel-Buffering", "no") // Disable nginx buffering
        .header("Transfer-Encoding", "chunked") // Enable HTTP chunked transfer encoding
        .header("Access-Control-Allow-Origin", "*") // CORS for browser clients
        .body(body)
        .map_err(|e| {
            SpeechError::Mp3Conversion(std::io::Error::new(std::io::ErrorKind::Other, e))
        })?)
}

async fn handle_voices(State(tts): State<TTSKoko>) -> Json<VoicesResponse> {
    let voices = tts.get_available_voices();
    Json(VoicesResponse { voices })
}

/// Handle /v1/models endpoint
/// 
/// Returns a static list of models for OpenAI API compatibility.
/// Note: All models use the same underlying Kokoro TTS engine.
async fn handle_models() -> Json<ModelsResponse> {
    let models = vec![
        ModelObject {
            id: "tts-1".to_string(),
            object: "model".to_string(),
            created: 1686935002,
            owned_by: "kokoro".to_string(),
        },
        ModelObject {
            id: "tts-1-hd".to_string(),
            object: "model".to_string(),
            created: 1686935002,
            owned_by: "kokoro".to_string(),
        },
        ModelObject {
            id: "kokoro".to_string(),
            object: "model".to_string(),
            created: 1686935002,
            owned_by: "kokoro".to_string(),
        },
    ];

    Json(ModelsResponse {
        object: "list".to_string(),
        data: models,
    })
}

async fn handle_model(Path(model_id): Path<String>) -> Result<Json<ModelObject>, StatusCode> {
    let model = match model_id.as_str() {
        "tts-1" => ModelObject {
            id: "tts-1".to_string(),
            object: "model".to_string(),
            created: 1686935002,
            owned_by: "kokoro".to_string(),
        },
        "tts-1-hd" => ModelObject {
            id: "tts-1-hd".to_string(),
            object: "model".to_string(),
            created: 1686935002,
            owned_by: "kokoro".to_string(),
        },
        "kokoro" => ModelObject {
            id: "kokoro".to_string(),
            object: "model".to_string(),
            created: 1686935002,
            owned_by: "kokoro".to_string(),
        },
        _ => return Err(StatusCode::NOT_FOUND),
    };

    Ok(Json(model))
}
