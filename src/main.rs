use clap::Parser;
mod cli;
use cli::Cli;

fn main() {
    let args = Cli::parse();
}
