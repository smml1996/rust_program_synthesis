use clap::Parser;
mod cli;
use cli::Cli;

fn main() {
    let _args = Cli::parse();
}
