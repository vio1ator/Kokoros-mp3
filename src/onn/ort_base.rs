use ort::session::builder::SessionBuilder;
use ort::session::Session;

pub trait OrtBase {
    fn load_model(&mut self, model_path: String) -> Result<(), String> {
        match SessionBuilder::new() {
            Ok(builder) => {
                let session = builder
                    .commit_from_file(model_path)
                    .map_err(|e| format!("Failed to commit from file: {}", e))?;
                self.set_sess(session);
                Ok(())
            }
            Err(e) => Err(format!("Failed to create session builder: {}", e)),
        }
    }

    fn print_info(&self) {
        if let Some(session) = self.sess() {
            println!("Input names:");
            for input in &session.inputs {
                println!("  - {}", input.name);
            }
            println!("Output names:");
            for output in &session.outputs {
                println!("  - {}", output.name);
            }
        } else {
            println!("Session is not initialized.");
        }
    }

    fn set_sess(&mut self, sess: Session);
    fn sess(&self) -> Option<&Session>;
}
