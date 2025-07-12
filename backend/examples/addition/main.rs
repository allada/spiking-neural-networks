use rand::seq::SliceRandom;
use rand::Rng;
use spiking_neural_networks::{
    error::SpikingNeuralNetworksError,
    neuron::{
        integrate_and_fire::IzhikevichNeuron,
        iterate_and_spike::{
            ApproximateNeurotransmitter, ApproximateReceptor,
            IonotropicNeurotransmitterType, GaussianParameters,
        },
        plasticity::STDP,
        spike_train::{DeltaDiracRefractoriness, PresetSpikeTrain},
        GridVoltageHistory, Lattice, LatticeNetwork, RunNetwork, SpikeTrainGridHistory,
        SpikeTrainLattice,
    },
};

fn generate_dataset(size: usize, test_split: f32) -> (Vec<(u8, u8, u8)>, Vec<(u8, u8, u8)>) {
    let mut rng = rand::thread_rng();
    let mut data = Vec::with_capacity(size);
    for _ in 0..size {
        let a = rng.gen_range(0..=9);
        let b = rng.gen_range(0..=9);
        data.push((a, b, a + b));
    }
    let split = ((size as f32) * (1.0 - test_split)) as usize;
    let test = data.split_off(split);
    (data, test)
}

fn set_inputs(lattice: &mut SpikeTrainLattice<IonotropicNeurotransmitterType, PresetSpikeTrain<IonotropicNeurotransmitterType, ApproximateNeurotransmitter, DeltaDiracRefractoriness>, SpikeTrainGridHistory>, a: u8, b: u8) {
    lattice.apply(|st| {
        st.firing_times = vec![1000.0];
        st.counter = 0;
    });
    lattice.apply_given_position(|(idx, _), st| {
        if idx == a as usize || idx == 10 + b as usize {
            st.firing_times = vec![0.0];
        }
    });
}

fn teacher_force(lattice: &mut Lattice<IzhikevichNeuron<ApproximateNeurotransmitter, ApproximateReceptor>, spiking_neural_networks::graph::AdjacencyMatrix<(usize, usize), f32>, GridVoltageHistory, STDP, IonotropicNeurotransmitterType>, target: usize) {
    lattice.apply(|n| {
        n.current_voltage = n.v_init;
        n.is_spiking = false;
        n.last_firing_time = None;
    });
    lattice.apply_given_position(|(idx, _), n| {
        if idx == target {
            n.current_voltage = n.v_th + 5.0;
        }
    });
}

fn main() -> Result<(), SpikingNeuralNetworksError> {
    type SpikeTrainType = PresetSpikeTrain<IonotropicNeurotransmitterType, ApproximateNeurotransmitter, DeltaDiracRefractoriness>;
    let stdp = STDP::default();
    let weight_params = GaussianParameters { mean: 1.5, std: 0.1, min: 1.0, max: 2.0 };

    let (mut train, test) = generate_dataset(2000, 0.2);

    let preset_spike = SpikeTrainType::default_impl();
    let mut spike_trains: SpikeTrainLattice<_, _, _> = SpikeTrainLattice::default();
    spike_trains.populate(&preset_spike, 20, 1)?;
    spike_trains.set_id(0);

    let neuron = IzhikevichNeuron::default_impl();
    let mut outputs: Lattice<_, _, _, _, _> = Lattice::default_impl();
    outputs.populate(&neuron, 19, 1)?;
    outputs.plasticity = stdp;
    outputs.do_plasticity = true;
    outputs.set_id(1);

    let lattices = vec![outputs];
    let spike_lattices = vec![spike_trains];
    let mut network: LatticeNetwork<_, _, _, _, _, _, _, _> = LatticeNetwork::generate_network(lattices, spike_lattices)?;
    network.connect(0, 1, &(|_, _| true), Some(&(|_, _| weight_params.get_random_number())))?;
    network.electrical_synapse = true;
    network.chemical_synapse = false;

    for epoch in 0..1000 {
        for (i, &(a, b, sum)) in train.iter().enumerate() {
            set_inputs(network.get_mut_spike_train_lattice(&0).unwrap(), a, b);
            teacher_force(network.get_mut_lattice(&1).unwrap(), sum as usize);
            network.run_lattices(1)?;
            if (i + 1) % 1000 == 0 {
                network.reset_timing();
            }
        }
        train.shuffle(&mut rand::thread_rng());
        println!("finished epoch {}", epoch + 1);
    }

    let mut correct = 0usize;
    for &(a, b, sum) in &test {
        set_inputs(network.get_mut_spike_train_lattice(&0).unwrap(), a, b);
        network.get_mut_lattice(&1).unwrap().apply(|n| { n.is_spiking = false; });
        network.run_lattices(1)?;
        let lattice = network.get_lattice(&1).unwrap();
        let mut best_idx = 0usize;
        let mut best_voltage = f32::MIN;
        for (idx, row) in lattice.cell_grid().iter().enumerate() {
            let neuron = &row[0];
            let v = neuron.current_voltage;
            if neuron.is_spiking || v > best_voltage {
                best_voltage = v;
                best_idx = idx;
            }
        }
        if best_idx as u8 == sum { correct += 1; }
    }
    let accuracy = correct as f32 / test.len() as f32;
    println!("Test accuracy: {}", accuracy);

    Ok(())
}
