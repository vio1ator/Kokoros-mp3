mod onn;
mod tts;
mod utils;

fn main() {
    println!("Hello, world!");

    let tts = tts::koko::TTSKoko::new("checkpoints/koko.onnx");

    tts.tts("hello, world.");
}

fn test() {
    let array: [[i32; 3]; 2] = [[1, 2, 3], [4, 5, 6]];
}
