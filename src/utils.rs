pub mod instructions {
    use std::collections::HashSet;
    #[derive(PartialEq, Eq, Hash, Debug)]
    pub enum Gate {
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

    /// Basis gates sets that different hardware specifications utilize
    #[derive(Debug, PartialEq, Eq, Hash)]
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

    impl BasisGates {
        fn value(basis_gates: BasisGates) -> HashSet<Gate> {
            match basis_gates {
                BasisGates::TYPE1 => HashSet::from([
                    Gate::Cnot, // multiqubit gates
                    Gate::U1,
                    Gate::U2,
                    Gate::U3, // single qubit gates
                ]),
                BasisGates::TYPE2 => HashSet::from([
                    Gate::Cnot, // multiqubit gate
                    Gate::Rz,
                    Gate::Sx,
                    Gate::X, // single qubit gates
                    Gate::Meas,
                    Gate::Reset, // non-unitary
                ]),
                BasisGates::TYPE3 => HashSet::from([
                    // No meas errors?
                    Gate::Cnot, // multiqubit gate
                    Gate::Rz,
                    Gate::Sx,
                    Gate::X,     // single qubit gates
                    Gate::Reset, // non-unitary
                ]),
                BasisGates::TYPE4 => HashSet::from([
                    Gate::Cz, //multiqubit gate
                    Gate::Rz,
                    Gate::Sx,
                    Gate::X, // unitary gates
                    Gate::Meas,
                    Gate::Reset, // non-unitary gates
                ]),
                BasisGates::TYPE5 => HashSet::from([
                    Gate::Rz,
                    Gate::Sx,
                    Gate::X, // unitary gates
                ]),
                BasisGates::TYPE6 => HashSet::from([
                    Gate::Cnot, // multiqubit gates
                    Gate::Sx,
                    Gate::X,
                    Gate::U1,
                    Gate::U2,
                    Gate::U3, // single qubit gates
                ]),
                BasisGates::TYPE7 => HashSet::from([
                    Gate::Cnot, // multiqubit gates
                    Gate::Rz,
                    Gate::Sx,
                    Gate::X, // unitary gates
                ]),
                BasisGates::TYPE8 => HashSet::from([
                    Gate::Cnot, // multiqubit gates
                    Gate::U1,
                    Gate::U2,
                    Gate::U3, // single qubit gates
                    Gate::Meas,
                    Gate::Reset, // non-unitary gates
                ]),
                BasisGates::TYPE9 => HashSet::from([
                    Gate::Rz,
                    Gate::Sx,
                    Gate::X, // unitary gates
                    Gate::Meas,
                    Gate::Reset, // non-unitary gates
                ]),
                BasisGates::TYPE10 => HashSet::from([
                    Gate::Ecr, // multiqubit gates
                    Gate::Rz,
                    Gate::Sx,
                    Gate::X, // unitary gates
                    Gate::Meas,
                    Gate::Reset, // non-unitary gates
                ]),
                BasisGates::TYPE11 => HashSet::from([
                    Gate::Ecr,
                    Gate::Cnot, // multiqubit gates
                    Gate::Rz,
                    Gate::Sx,
                    Gate::X, // unitary gates
                    Gate::Meas,
                    Gate::Reset, // non-unitary gates
                ]),
            }
        }
    }

    pub enum GateType {
        CLASSIC,     // only classical reads and writes
        MEASUREMENT, // must return classical output to fall into this category
        MULTIQUBIT,  // must be unitary
        SINGLEQUBIT, // must be unitary
        NONUNITARY,  // reset falls here, anything that does not return an observable
    }

    impl Gate {
        pub fn get_gate_type(gate: &Gate) -> GateType {
            match gate {
                Gate::I
                | Gate::X
                | Gate::Y
                | Gate::Z
                | Gate::H
                | Gate::S
                | Gate::Sd
                | Gate::Sx
                | Gate::Sxd
                | Gate::U1
                | Gate::U2
                | Gate::U3
                | Gate::T
                | Gate::Td
                | Gate::Rz
                | Gate::Ry
                | Gate::Rx => GateType::SINGLEQUBIT,

                Gate::Reset | Gate::Custom => GateType::NONUNITARY, // Custom goes here cause kraus matrices are not always unitary

                Gate::Meas => GateType::MEASUREMENT,

                Gate::Cnot
                | Gate::Ecr
                | Gate::Rzx
                | Gate::Cz
                | Gate::Ch
                | Gate::Swap
                | Gate::Toffoli => GateType::MULTIQUBIT,
            }
        }
    }

    struct Instruction {
        target: u16, // use this whenever possible: for classical read and writes, as well as quantum instructions
        gate: Gate,
        controls: Option<Vec<u16>>,    // used for multiqubit gates
        params: Option<Vec<f64>>,      // used for parametric gates
        matrix: Option<[[f64; 2]; 2]>, // only used for Op::Custom and is a 1-qubit matrix
        classical_target: Option<u16>, // used only for measurements. The classical output is written at this index in the classical state
    }

    impl Instruction {
        fn new(
            target: u16,
            gate: Gate,
            controls: Option<Vec<u16>>,
            params: Option<Vec<f64>>,
            matrix: Option<[[f64; 2]; 2]>,
            classical_target: Option<u16>,
        ) -> Self {
            Self {
                target,
                gate,
                controls,
                params,
                matrix,
                classical_target,
            }
        }

        #[cfg(debug_assertions)] // this function only exists in debug mode
        fn check_instruction(self) {
            match Gate::get_gate_type(&self.gate) {
                GateType::CLASSIC => {
                    debug_assert!(
                        self.controls.is_none(),
                        "classical instructions do not need controls"
                    );
                    debug_assert!(
                        self.params.is_none(),
                        "no parameters needed for classical instructions"
                    );
                    debug_assert!(
                        self.matrix.is_none(),
                        "matrix not needed for classical instruction"
                    );
                    debug_assert!(
                        self.classical_target.is_none(),
                        "we only need to set target for classical instructions (not classical target)"
                    );
                }
                GateType::MEASUREMENT => {
                    debug_assert!(
                        self.classical_target.is_some(),
                        "classical target needed for measurement instruction"
                    );
                    debug_assert!(
                        self.controls.is_none(),
                        "controls are not needed for measurement instruction"
                    );
                    debug_assert!(
                        self.params.is_none(),
                        "parametric gate is invalid for measurement gates"
                    );
                    debug_assert!(
                        self.matrix.is_none(),
                        "matrix should not be specified for measurement instructions"
                    );
                }
                GateType::MULTIQUBIT => {
                    debug_assert!(
                        self.controls.is_some(),
                        "controls in multiqubit gate is none"
                    );
                    debug_assert!(
                        self.classical_target.is_none(),
                        "multiqubit gate does not needs classical target"
                    );

                    //assumptions that I might need to remove later:
                    debug_assert!(
                        self.matrix.is_none(),
                        "multiqubit gate cannot be specified with a matrix"
                    );
                }
                GateType::SINGLEQUBIT => {
                    debug_assert!(
                        self.controls.is_none(),
                        "single qubit gates should not have controls"
                    );
                    debug_assert!(
                        self.classical_target.is_none(),
                        "single qubit gates does not needs classical target"
                    );
                }
                GateType::NONUNITARY => {
                    // the following might be assumptions I might need to remove later:
                    debug_assert!(
                        self.controls.is_none(),
                        "non-unitary gate is multiqubit gate (with controls)"
                    );
                    debug_assert!(self.params.is_none(), "non-unitary gate has parameters");
                    debug_assert!(self.classical_target.is_none(), "classical target is not ");
                }
            }
        }
    }
}
