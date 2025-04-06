pub mod gates {
    use lazy_static::lazy_static;
    use std::{collections::HashSet, fmt};
    use strum::IntoEnumIterator;
    use strum_macros::EnumIter;
    #[derive(PartialEq, Eq, Hash, Debug, EnumIter)]
    pub enum GateName {
        // this enum just helps to keep track of the basis gates of the quantum hardware
        // one-qubit ops
        I,
        X,
        Y,
        Z,
        H,
        S,
        Sd,
        Sx,
        Sxd,
        U1,
        U2,
        U3,
        T,
        Td,
        Rz,
        Ry,
        Rx,
        Reset,
        Meas,
        Custom,

        // MULTI-QUBIT GATES
        Cnot,
        Ecr,
        Rzx,
        Cz,
        Ch,
        Swap,
        Toffoli,
    }

    impl fmt::Display for GateName {
        fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
            write!(f, "{:?}", self)
        }
    }

    impl GateName {
        fn get_enum_val(s: &str) -> GateName {
            let s = s.to_lowercase();
            for gate_name in GateName::iter() {
                let str_gate_name = gate_name.to_string().to_lowercase();

                if s == str_gate_name {
                    return gate_name;
                }
            }
            panic!("failed to get GateName enum for string {}", s)
        }
    }

    /// Basis gates sets that different hardware specifications utilize
    #[derive(Debug, PartialEq, Eq, Hash, EnumIter)]
    pub enum BasisGates {
        TYPE1,
        TYPE2,
        TYPE3,
        TYPE4,
        TYPE5,
        TYPE6,
        TYPE7,
        TYPE8,
        TYPE9,
        TYPE10,
        TYPE11,
    }

    lazy_static! {
        // TODO: Write tests to check that hashmap maps to all different values and that all types are present in the hashmap
        static ref BASIS_GATES_MAP: std::collections::HashMap<BasisGates, HashSet<GateName>> = {
            use BasisGates::*;
            use GateName::*;

            let mut map = std::collections::HashMap::new();
            map.insert(TYPE1, HashSet::from([
                Cnot, // multiqubit gates
                U1,
                U2,
                U3, // single qubit gates
            ]));
            map.insert(TYPE2, HashSet::from([
                Cnot, // multiqubit gate
                Rz,
                Sx,
                X, // single qubit gates
                Meas,
                Reset, // non-unitary
            ]));
            map.insert(TYPE3, HashSet::from([
                // No meas errors?
                Cnot, // multiqubit gate
                Rz,
                Sx,
                X,     // single qubit gates
                Reset, // non-unitary
            ]));
            map.insert(TYPE4, HashSet::from([
                Cz, //multiqubit gate
                Rz,
                Sx,
                X, // unitary gates
                Meas,
                Reset, // non-unitary gates
            ]));
            map.insert(TYPE5, HashSet::from([
                Rz,
                Sx,
                X, // unitary gates
            ]));
            map.insert(TYPE6, HashSet::from([
                Cnot, // multiqubit gates
                Sx,
                X,
                U1,
                U2,
                U3, // single qubit gates
            ]));
            map.insert(TYPE7, HashSet::from([
                Cnot, Rz, Sx, X
            ]));
            map.insert(TYPE8, HashSet::from([
                Cnot, Sx, X, Meas, Reset
            ]));
            map.insert(TYPE9, HashSet::from([
                Rz, Sx, X, Meas, Reset
            ]));
            map.insert(TYPE10, HashSet::from([
                Ecr, Cnot, Rz, Sx, X, Meas, Reset
            ]));
            map.insert(TYPE11, HashSet::from([Ecr, Cnot, Rz, Sx, X, Meas, Reset]));
            map
        };
    }

    impl BasisGates {
        pub fn find_basis_gates(raw_basis_gates: Vec<&str>) -> BasisGates {
            let current_gates: HashSet<GateName> = raw_basis_gates
                .iter()
                .copied()
                .map(GateName::get_enum_val)
                .collect();

            // convert raw_basis_gates to a hashset that contains gates
            for basis_gate_type in BasisGates::iter() {
                if let Some(basis_gates) = BASIS_GATES_MAP.get(&basis_gate_type) {
                    if *basis_gates == current_gates {
                        return basis_gate_type;
                    }
                } else {
                    panic!(
                        "This should not happen: static variable BASIS_GATES_MAP does not contains {:?}",
                        basis_gate_type
                    );
                }
            }
            panic!("basis gates not found for {:?}", raw_basis_gates);
        }
    }
}

pub mod instructions {
    use rug::{Complex, Float};
    use serde_json::Value;
    use std::hash::{Hash, Hasher};

    #[derive(Debug)]
    pub enum Instruction {
        // this enum just helps to keep track of the basis gates of the quantum hardware
        // one-qubit ops
        I {
            target: u16,
        },
        X {
            target: u16,
        },
        Y {
            target: u16,
        },
        Z {
            target: u16,
        },
        H {
            target: u16,
        },
        S {
            target: u16,
        },
        Sd {
            target: u16,
        },
        Sx {
            target: u16,
        },
        Sxd {
            target: u16,
        },
        U1 {
            target: u16,
            theta: Float,
        },
        U2 {
            target: u16,
            phi: Float,
            lambda: Float,
        },
        U3 {
            target: u16,
            theta: Float,
            phi: Float,
            lambda: Float,
        },
        T {
            target: u16,
        },
        Td {
            target: u16,
        },
        Rz {
            target: u16,
            phi: Float,
        },
        Ry {
            target: u16,
            theta: Float,
        },
        Rx {
            target: u16,
            theta: Float,
        },
        Reset {
            target: u16,
        },
        Meas {
            target: u16,
        },
        Custom {
            target: u16,
            matrix: [[Complex; 2]; 2],
        }, // this instruction is for applying arbitrary matrices to a qubit (hence 2x2 matrix)

        // MULTI-QUBIT GATES
        Cnot {
            control: u16,
            target: u16,
        },
        Ecr {
            control: u16,
            target: u16,
        },
        Rzx {
            control: u16,
            target: u16,
            theta: Float,
        },
        Cz {
            control: u16,
            target: u16,
        },
        Ch {
            control: u16,
            target: u16,
        },
        Swap {
            qubit1: u16,
            qubit2: u16,
        },
        Toffoli {
            control1: u16,
            control2: u16,
            target: u16,
        },
    }

    impl Instruction {
        fn get_control(json_val: &Value) -> u16 {
            json_val["control"]
                .as_u64()
                .unwrap()
                .try_into()
                .unwrap_or_else(|x| panic!("could not convert control {:?}", json_val))
        }

        pub fn new(json_val: &Value, prec: u32) -> Self {
            let op_name: &str = json_val["op"].as_str().unwrap_or_else(|| {
                panic!("could not convert op {:?}", json_val);
            });

            let target: u16 = json_val["target"]
                .as_u64()
                .unwrap()
                .try_into()
                .unwrap_or_else(|_e| {
                    panic!("target could not be converted ({:?})", json_val);
                });

            let (theta, phi, lambda) = (Float::new(prec), Float::new(prec), Float::new(prec));

            match op_name {
                "RESET" => Instruction::Reset { target },
                "CNOT" => {
                    let control = Instruction::get_control(json_val);
                    Instruction::Cnot { control, target }
                }
                "U1" => Instruction::U1 { target, theta },
                "U2" => Instruction::U2 {
                    target,
                    phi,
                    lambda,
                },
                "U3" => Instruction::U3 {
                    target,
                    theta,
                    phi,
                    lambda,
                },
                "RZ" => Instruction::Rz { target, phi },
                "SX" => Instruction::Sx { target },
                "X" => Instruction::X { target },
                "MEAS" => Instruction::Meas { target },
                "CZ" => {
                    let control = Instruction::get_control(json_val);
                    Instruction::Cz { control, target }
                }
                "ECR" => {
                    let control: u16 = Instruction::get_control(json_val);
                    Instruction::Ecr { control, target }
                }
                _ => panic!("op {} could not be converted to Instruction", op_name),
            }
        }
    }

    impl PartialEq for Instruction {
        fn eq(&self, other: &Self) -> bool {
            match (self, other) {
                (Self::I { target: l_target }, Self::I { target: r_target }) => {
                    l_target == r_target
                }
                (Self::X { target: l_target }, Self::X { target: r_target }) => {
                    l_target == r_target
                }
                (Self::Y { target: l_target }, Self::Y { target: r_target }) => {
                    l_target == r_target
                }
                (Self::Z { target: l_target }, Self::Z { target: r_target }) => {
                    l_target == r_target
                }
                (Self::H { target: l_target }, Self::H { target: r_target }) => {
                    l_target == r_target
                }
                (Self::S { target: l_target }, Self::S { target: r_target }) => {
                    l_target == r_target
                }
                (Self::Sd { target: l_target }, Self::Sd { target: r_target }) => {
                    l_target == r_target
                }
                (Self::Sx { target: l_target }, Self::Sx { target: r_target }) => {
                    l_target == r_target
                }
                (Self::Sxd { target: l_target }, Self::Sxd { target: r_target }) => {
                    l_target == r_target
                }
                (
                    Self::U1 {
                        target: l_target, ..
                    },
                    Self::U1 {
                        target: r_target, ..
                    },
                ) => l_target == r_target,
                (
                    Self::U2 {
                        target: l_target, ..
                    },
                    Self::U2 {
                        target: r_target, ..
                    },
                ) => l_target == r_target,
                (
                    Self::U3 {
                        target: l_target, ..
                    },
                    Self::U3 {
                        target: r_target, ..
                    },
                ) => l_target == r_target,
                (Self::T { target: l_target }, Self::T { target: r_target }) => {
                    l_target == r_target
                }
                (Self::Td { target: l_target }, Self::Td { target: r_target }) => {
                    l_target == r_target
                }
                (
                    Self::Rz {
                        target: l_target, ..
                    },
                    Self::Rz {
                        target: r_target, ..
                    },
                ) => l_target == r_target,
                (
                    Self::Ry {
                        target: l_target, ..
                    },
                    Self::Ry {
                        target: r_target, ..
                    },
                ) => l_target == r_target,
                (
                    Self::Rx {
                        target: l_target, ..
                    },
                    Self::Rx {
                        target: r_target, ..
                    },
                ) => l_target == r_target,
                (Self::Reset { target: l_target }, Self::Reset { target: r_target }) => {
                    l_target == r_target
                }
                (Self::Meas { target: l_target }, Self::Meas { target: r_target }) => {
                    l_target == r_target
                }
                (
                    Self::Custom {
                        target: l_target, ..
                    },
                    Self::Custom {
                        target: r_target, ..
                    },
                ) => l_target == r_target,
                (
                    Self::Cnot {
                        control: l_control,
                        target: l_target,
                    },
                    Self::Cnot {
                        control: r_control,
                        target: r_target,
                    },
                ) => l_control == r_control && l_target == r_target,
                (
                    Self::Ecr {
                        control: l_control,
                        target: l_target,
                    },
                    Self::Ecr {
                        control: r_control,
                        target: r_target,
                    },
                ) => l_control == r_control && l_target == r_target,
                (
                    Self::Rzx {
                        control: l_control,
                        target: l_target,
                        ..
                    },
                    Self::Rzx {
                        control: r_control,
                        target: r_target,
                        ..
                    },
                ) => l_control == r_control && l_target == r_target,
                (
                    Self::Cz {
                        control: l_control,
                        target: l_target,
                    },
                    Self::Cz {
                        control: r_control,
                        target: r_target,
                    },
                ) => l_control == r_control && l_target == r_target,
                (
                    Self::Ch {
                        control: l_control,
                        target: l_target,
                    },
                    Self::Ch {
                        control: r_control,
                        target: r_target,
                    },
                ) => l_control == r_control && l_target == r_target,
                (
                    Self::Swap {
                        qubit1: l_qubit1,
                        qubit2: l_qubit2,
                    },
                    Self::Swap {
                        qubit1: r_qubit1,
                        qubit2: r_qubit2,
                    },
                ) => l_qubit1 == r_qubit1 && l_qubit2 == r_qubit2,
                (
                    Self::Toffoli {
                        control1: l_control1,
                        control2: l_control2,
                        target: l_target,
                    },
                    Self::Toffoli {
                        control1: r_control1,
                        control2: r_control2,
                        target: r_target,
                    },
                ) => l_control1 == r_control1 && l_control2 == r_control2 && l_target == r_target,
                _ => false,
            }
        }
    }

    impl Eq for Instruction {}

    impl Hash for Instruction {
        fn hash<H: Hasher>(&self, state: &mut H) {
            std::mem::discriminant(self).hash(state); // Hash enum variant
            // we will not hash parameters Float and Complex, only target since this is what we need for the hashMap of the hardware
            match self {
                // One-qubit gates
                Instruction::I { target }
                | Instruction::X { target }
                | Instruction::Y { target }
                | Instruction::Z { target }
                | Instruction::H { target }
                | Instruction::S { target }
                | Instruction::Sd { target }
                | Instruction::Sx { target }
                | Instruction::Sxd { target }
                | Instruction::T { target }
                | Instruction::Td { target }
                | Instruction::Reset { target }
                | Instruction::Meas { target }
                | Instruction::U1 { target, .. }
                | Instruction::Rz { target, .. }
                | Instruction::Ry { target, .. }
                | Instruction::Rx { target, .. }
                | Instruction::U2 { target, .. }
                | Instruction::U3 { target, .. }
                | Instruction::Custom { target, .. } => {
                    target.hash(state);
                }
                // Multi-qubit gates
                Instruction::Cnot { control, target }
                | Instruction::Ecr { control, target }
                | Instruction::Cz { control, target }
                | Instruction::Ch { control, target }
                | Instruction::Rzx {
                    control, target, ..
                } => {
                    control.hash(state);
                    target.hash(state);
                }
                Instruction::Swap { qubit1, qubit2 } => {
                    qubit1.hash(state);
                    qubit2.hash(state);
                }
                Instruction::Toffoli {
                    control1,
                    control2,
                    target,
                } => {
                    control1.hash(state);
                    control2.hash(state);
                    target.hash(state);
                }
            }
        }
    }
}

pub mod channels {
    use core::panic;
    use std::iter::zip;

    use rug::{Float, ops::CompleteRound};
    use serde_json::Value;

    use super::instructions::Instruction;

    #[derive(Debug)]
    pub enum Channel {
        Quantum(QuantumChannel),
        Measurement(MeasurementChannel),
    }

    #[derive(Debug)]
    pub struct QuantumChannel {
        errors_to_probs: Vec<(Vec<Instruction>, Float)>,
    }

    impl QuantumChannel {
        pub fn new(json_val: &Value, prec: u32) -> Self {
            let mut errors_to_probs: Vec<(Vec<Instruction>, Float)> = Vec::new();

            let probabilities = json_val["probabilities"].as_array().unwrap();
            let all_errors = json_val["errors"].as_array().unwrap();

            for (vec_errors_, probability_) in zip(all_errors, probabilities) {
                let mut final_errors: Vec<Instruction> = Vec::new();
                let probability = Float::parse(probability_.as_str().unwrap())
                    .unwrap()
                    .complete(prec);

                if let Some(vec_errors) = vec_errors_.as_array() {
                    for json_instruction in vec_errors {
                        let instruction = Instruction::new(json_instruction, prec);
                        final_errors.push(instruction);
                    }
                } else {
                    panic!(
                        "[QuantumChannel] could not convert vector {:?}",
                        vec_errors_
                    );
                }
                errors_to_probs.push((final_errors, probability));
            }

            Self { errors_to_probs }
        }
    }

    #[derive(Debug)]
    pub struct MeasurementChannel {
        correct_0: Float,   // probability of receiving 0 and that it is actually 0
        correct_1: Float,   // probability of receiving 1 and that it is actually 1
        incorrect_0: Float, // probability of receiving 0 and that it is actually 1
        incorrect_1: Float, // probability of receiving 1 and that it is actually 0
    }

    impl MeasurementChannel {
        pub fn new(json_val: &Value, prec: u32) -> Self {
            let str_correct_0 = json_val["0"]["0"].as_str().unwrap_or_else(|| {
                panic!(
                    "[MeasurementChannel] could not convert float 00 {:?}",
                    json_val
                );
            });
            let str_incorrect_0 = json_val["0"]["1"].as_str().unwrap_or_else(|| {
                panic!(
                    "[MeasurementChannel] could not convert float 01 {:?}",
                    json_val
                );
            });
            let str_correct_1 = json_val["1"]["1"].as_str().unwrap_or_else(|| {
                panic!(
                    "[MeasurementChannel] could not convert float 11 {:?}",
                    json_val
                );
            });
            let str_incorrect_1 = json_val["1"]["0"].as_str().unwrap_or_else(|| {
                panic!(
                    "[MeasurementChannel] could not convert float 10 {:?}",
                    json_val
                );
            });

            let correct_0: Float = Float::parse(str_correct_0).unwrap().complete(prec);
            let correct_1: Float = Float::parse(str_correct_1).unwrap().complete(prec);
            let incorrect_0: Float = Float::parse(str_incorrect_0).unwrap().complete(prec);
            let incorrect_1 = Float::parse(str_incorrect_1).unwrap().complete(prec);
            Self {
                correct_0,
                correct_1,
                incorrect_0,
                incorrect_1,
            }
        }

        fn get_probability(&self, received_val: bool, actual_val: bool) -> &Float {
            if received_val == actual_val {
                if !received_val {
                    &self.correct_0
                } else {
                    &self.correct_1
                }
            } else if !received_val {
                &self.incorrect_0
            } else {
                &self.incorrect_1
            }
        }
    }
}
