mod quantum_hardware {
    use std::collections::HashMap;
    use std::fmt;
    use std::iter::zip;
    use std::path::PathBuf;

    use serde_json::{Value, from_reader};
    use std::fs::File;
    use std::io::BufReader;

    use strum::IntoEnumIterator;
    use strum_macros::EnumIter;

    use crate::utils::channels::{Channel, MeasurementChannel, QuantumChannel};
    use crate::utils::gates::BasisGates;
    use crate::utils::instructions::Instruction;

    #[derive(PartialEq, Eq, Hash, Debug, EnumIter)]
    pub enum QuantumHardware {
        Algiers,
        Almaden,
        Armonk,
        Athens,
        Auckland,
        Belem,
        Boeblingen,
        Bogota,
        Brisbane,
        Brooklyn,
        Burlington,
        Cairo,
        Cambridge,
        Casablanca,
        Cusco,
        Essex,
        Fez,
        Geneva,
        Guadalupe,
        Hanoi,
        Jakarta,
        Johannesburg,
        Kawasaki,
        Kolkata,
        Kyiv,
        Kyoto,
        Lagos,
        Lima,
        London,
        Makarresh,
        Melbourne,
        Manhattan,
        Manila,
        Montreal,
        Mumbai,
        Nairobi,
        Osaka,
        Oslo,
        Ourense,
        Paris,
        Perth,
        Poughkeepsie,
        Prague,
        Quito,
        Rochester,
        Rome,
        Santiago,
        Singapore,
        Sydney,
        Torino,
        Toronto,
        Valencia,
        Vigo,
        Washington,
        Yorktown,
    }

    impl fmt::Display for QuantumHardware {
        fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
            write!(f, "{:?}", self)
        }
    }

    impl QuantumHardware {
        fn get_enum_val(s: &str) -> QuantumHardware {
            for quantum_hardware in QuantumHardware::iter() {
                let str_quantum_hardware =
                    "fake_".to_owned() + &quantum_hardware.to_string().to_lowercase();

                if s == str_quantum_hardware {
                    return quantum_hardware;
                }
            }
            panic!("failed to get quantum hardware enum for string {}", s);
        }

        fn get_str_repr(&self) -> String {
            "fake_".to_string() + &self.to_string().to_lowercase()
        }
    }

    struct HardwareSpecification {
        quantum_hardware: QuantumHardware,
        basis_gates: BasisGates,
        num_qubits: u16, // number of qubits that a quantum hardware possess
        thermal_relaxation: bool,
        instructions_to_channels: HashMap<Instruction, Channel>,
    }

    impl HardwareSpecification {
        fn new(qhardware_name: &str, prec: Option<u32>, with_thermalization: Option<bool>) -> Self {
            let prec = prec.unwrap_or(8);
            let quantum_hardware = QuantumHardware::get_enum_val(qhardware_name);
            let thermal_relaxation = with_thermalization.unwrap_or(false);

            let path = if thermal_relaxation {
                PathBuf::from(env!("CARGO_MANIFEST_DIR"))
                    .join("hardware_specs")
                    .join("with_thermalization")
                    .join("fake_".to_owned() + qhardware_name)
            } else {
                PathBuf::from(env!("CARGO_MANIFEST_DIR"))
                    .join("hardware_specs")
                    .join("no_thermalization")
                    .join("fake_".to_owned() + qhardware_name)
            };

            // opening and parsing JSON file
            let file = File::open(path).unwrap_or_else(|e| {
                panic!("{}", e);
            });
            let reader = BufReader::new(file);
            // Parse JSON into the struct
            let json_hardware_spec: Value = from_reader(reader).unwrap_or_else(|error| {
                panic!("{}", error);
            });

            let num_qubits: u16 = json_hardware_spec["num_qubits"]
                .as_u64()
                .unwrap()
                .try_into()
                .unwrap_or_else(|e| {
                    panic!("Cannot cast number of qubits ({:?})", e);
                });

            let mut raw_basis_gates: Vec<&str> = Vec::new();

            for gate in json_hardware_spec["basis_gates"]
                .as_array()
                .unwrap_or_else(|| {
                    panic!("cannot convert basis gates to array");
                })
            {
                let str_g: &str = gate.as_str().unwrap_or_else(|| {
                    panic!("cannot convert {:?} to &str", gate);
                });
                raw_basis_gates.push(str_g);
            }

            let basis_gates = BasisGates::find_basis_gates(raw_basis_gates);

            // building instructions_to_channels
            let instructions = json_hardware_spec["instructions"].as_array().unwrap_or_else(|| {
                panic!("could not convert harwdare spec instruction given in hardware specification{:?}", quantum_hardware);
            });

            let channels = json_hardware_spec["channels"]
                .as_array()
                .unwrap_or_else(|| {
                    panic!(
                        "could not convert channels for hardware specification {:?}",
                        quantum_hardware
                    );
                });

            let mut instructions_to_channels = HashMap::new();
            for (instruction_, channel_) in zip(instructions, channels) {
                let instruction = Instruction::new(instruction_, prec);
                let channel = match instruction {
                    Instruction::Meas { .. } => {
                        Channel::Measurement(MeasurementChannel::new(channel_, prec))
                    }
                    _ => Channel::Quantum(QuantumChannel::new(channel_, prec)),
                };

                instructions_to_channels.insert(instruction, channel);
            }

            Self {
                quantum_hardware,
                basis_gates,
                num_qubits,
                thermal_relaxation,
                instructions_to_channels,
            }
        }
    }
}
