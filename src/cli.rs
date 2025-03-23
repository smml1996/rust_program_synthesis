use clap::{Parser, Subcommand, command};

#[derive(Parser)]
#[command(
    name = "qsynthesis",
    version = env!("CARGO_PKG_VERSION"),
    author = env!("CARGO_PKG_AUTHORS"),
    about = env!("CARGO_PKG_DESCRIPTION")
)]
pub struct Cli {
    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand)]
enum Commands {
    // cargo run -- hardware --name name
    Hardware {
        /// Name to greet
        #[arg(short, long)]
        name: String,
    },

    /// Perform addition
    Add {
        /// First number
        a: i32,
        /// Second number
        b: i32,
    },
}
