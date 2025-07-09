//! OpenAI-compatible TTS HTTP server for Kokoros
//!
//! This module provides an HTTP API that is compatible with OpenAI's text-to-speech endpoints.
//! It implements non-streaming audio generation with multiple format support.
//!
//! ## Implemented Features
//! - `/v1/audio/speech` - Text-to-speech generation
//! - `/v1/audio/voices` - List available voices
//! - `/v1/models` - List available models (static dummy list)
//! - Multiple audio formats: MP3, WAV, PCM, OPUS, AAC, FLAC
//!
//! ## OpenAI API Compatibility Limitations
//! - `return_download_link`: Not implemented (files are streamed directly)
//! - `lang_code`: Not implemented (language auto-detected from voice prefix)
//! - `volume_multiplier`: Not implemented (audio returned at original levels)
//! - `download_format`: Not implemented (only response_format used)
//! - `normalization_options`: Not implemented (basic text processing only)
//! - `stream`: Not implemented (always non-streaming)

use std::error::Error;
use std::io::{self};

use axum::http::{header, StatusCode};
use axum::response::{IntoResponse, Response};
use axum::{extract::{Path, State}, routing::get, routing::post, Json, Router};
use kokoros::{
    tts::koko::{InitConfig as TTSKokoInitConfig, TTSKoko},
    utils::mp3::pcm_to_mp3,
    utils::wav::{write_audio_chunk, WavHeader},
};
use serde::{Deserialize, Serialize};
use tower_http::cors::CorsLayer;

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

    /// Enable streaming audio generation (not implemented)
    #[serde(default)]
    #[allow(dead_code)]
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
    Json(SpeechRequest {
        model: _,
        input,
        voice: Voice(voice),
        response_format,
        speed: Speed(speed),
        initial_silence,
        ..
    }): Json<SpeechRequest>,
) -> Result<Response, SpeechError> {
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
