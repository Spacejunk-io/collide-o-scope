//! Spout receiver probe: connects to the collide-o-scope sender and
//! verifies real frames arrive. Used to test the Spout output end-to-end
//! without OBS/Resolume installed.
//!
//! Usage: cargo run --bin spout_probe   (while collide-o-scope runs with
//! Spout output enabled). Prints sender info and pixel statistics.

#[cfg(windows)]
fn main() {
    let mut receiver = match spout2::dx::Receiver::new(Some("collide-o-scope")) {
        Ok(r) => r,
        Err(e) => {
            eprintln!("PROBE FAIL: receiver init: {e}");
            std::process::exit(1);
        }
    };

    // Canonical Spout receive loop: call receive_image; when is_updated
    // fires, resize to the sender's dimensions and keep going.
    let (mut w, mut h) = (16u32, 16u32);
    let mut pixels = vec![0u8; (w * h * 4) as usize];
    let mut connected = false;
    let mut nonzero_frames = 0;
    let mut received_frames = 0;

    for _ in 0..200 {
        match receiver.receive_image(&mut pixels, w, h, false, false) {
            Ok(true) => {
                if receiver.is_updated() {
                    let (sw, sh) = receiver.sender_size();
                    if sw > 0 && sh > 0 {
                        w = sw;
                        h = sh;
                        pixels = vec![0u8; (w * h * 4) as usize];
                        if !connected {
                            connected = true;
                            println!("connected: sender='{}' {}x{}", receiver.sender_name(), w, h);
                        }
                    }
                    continue;
                }
                if connected && receiver.is_frame_new() {
                    received_frames += 1;
                    if pixels.iter().any(|&b| b != 0) {
                        nonzero_frames += 1;
                    }
                    if received_frames >= 30 {
                        break;
                    }
                }
            }
            Ok(false) => {}
            Err(e) => {
                eprintln!("PROBE FAIL: receive_image: {e}");
                std::process::exit(1);
            }
        }
        std::thread::sleep(std::time::Duration::from_millis(30));
    }

    if !connected {
        eprintln!("PROBE FAIL: no sender named 'collide-o-scope' found");
        std::process::exit(1);
    }

    println!("frames received: {received_frames}, non-black: {nonzero_frames}");
    if nonzero_frames > 0 {
        println!("PROBE OK");
    } else {
        eprintln!("PROBE FAIL: connected but all frames black");
        std::process::exit(1);
    }
}

#[cfg(not(windows))]
fn main() {
    eprintln!("spout_probe is Windows-only");
}
