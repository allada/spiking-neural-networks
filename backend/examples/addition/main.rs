use rand::Rng;
extern crate spiking_neural_networks;
use spiking_neural_networks::{
    error::SpikingNeuralNetworksError,
    graph::AdjacencyMatrix,
    neuron::{
        integrate_and_fire::IzhikevichNeuron,
        iterate_and_spike::{
            ApproximateNeurotransmitter, GaussianParameters, NeurotransmitterType,
        },
        plasticity::STDP,
        spike_train::{DeltaDiracRefractoriness, PresetSpikeTrain},
        Lattice, LatticeNetwork, RunNetwork, SpikeTrainLattice,
    },
};

#[derive(Clone, Copy)]
struct Sample {
    a: usize,
    b: usize,
    sum: usize,
}

fn generate_dataset(n: usize) -> (Vec<Sample>, Vec<Sample>) {
    let mut rng = rand::thread_rng();
    let mut data: Vec<Sample> = (0..n)
        .map(|_| {
            let a = rng.gen_range(0..10);
            let b = rng.gen_range(0..10);
            Sample { a, b, sum: a + b }
        })
        .collect();
    let split = (n as f32 * 0.8) as usize;
    let test = data.split_off(split);
    (data, test)
}

fn build_network() -> Result<
    LatticeNetwork<
        IzhikevichNeuron<
            ApproximateNeurotransmitter,
            spiking_neural_networks::neuron::iterate_and_spike::ApproximateReceptor,
        >,
        AdjacencyMatrix<(usize, usize), f32>,
        spiking_neural_networks::neuron::GridVoltageHistory,
        PresetSpikeTrain<
            spiking_neural_networks::neuron::IonotropicNeurotransmitterType,
            ApproximateNeurotransmitter,
            DeltaDiracRefractoriness,
        >,
        spiking_neural_networks::neuron::SpikeTrainGridHistory,
        AdjacencyMatrix<spiking_neural_networks::neuron::GraphPosition, f32>,
        STDP,
        spiking_neural_networks::neuron::IonotropicNeurotransmitterType,
    >,
> {
    type NType = spiking_neural_networks::neuron::IonotropicNeurotransmitterType;
    type Spike = PresetSpikeTrain<NType, ApproximateNeurotransmitter, DeltaDiracRefractoriness>;

    let mut spike_trains: SpikeTrainLattice<
        NType,
        Spike,
        spiking_neural_networks::neuron::SpikeTrainGridHistory,
    > = SpikeTrainLattice::default();
    spike_trains.populate(&Spike::default_impl(), 2, 10)?;
    spike_trains.set_id(0);

    let neuron = IzhikevichNeuron {
        gap_conductance: 10.,
        ..IzhikevichNeuron::default_impl()
    };
    let mut lattice: Lattice<
        _,
        AdjacencyMatrix<(usize, usize), f32>,
        spiking_neural_networks::neuron::GridVoltageHistory,
        STDP,
        NType,
    > = Lattice::default();
    lattice.populate(&neuron, 19, 1)?;
    lattice.plasticity = STDP::default();
    lattice.do_plasticity = true;
    lattice.set_id(1);

    let mut network = LatticeNetwork::generate_network(vec![lattice], vec![spike_trains])?;
    let params = GaussianParameters {
        mean: 0.5,
        std: 0.1,
        min: 0.,
        max: 1.,
    };
    network.connect(0, 1, &|_, _| true, Some(&|_, _| params.get_random_number()))?;
    Ok(network)
}

fn main() -> Result<(), SpikingNeuralNetworksError> {
    let (train, test) = generate_dataset(1200);
    let mut network = build_network()?;

    for (n, sample) in train.iter().enumerate() {
        // clear spikes
        network
            .get_mut_spike_train_lattice(&0)
            .unwrap()
            .apply(|st| {
                st.firing_times.clear();
            });
        let clock = network.internal_clock as f32;
        network
            .get_mut_spike_train_lattice(&0)
            .unwrap()
            .apply_given_position(|pos, st| {
                if pos == (0, sample.a) || pos == (1, sample.b) {
                    st.firing_times = vec![clock];
                }
            });

        network.run_lattices(1)?;
        {
            let mut lattice = network.get_mut_lattice(&1).unwrap();
            lattice.apply_given_position(|pos, neuron| {
                if pos == (sample.sum, 0) {
                    neuron.current_voltage = neuron.v_th;
                }
            });
        }
        network.run_lattices(1)?;

        if (n + 1) % 1000 == 0 {
            network.reset_timing();
        }
    }

    let mut correct = 0usize;
    for sample in test.iter() {
        network.reset_timing();
        network
            .get_mut_spike_train_lattice(&0)
            .unwrap()
            .apply(|st| st.firing_times.clear());
        network
            .get_mut_spike_train_lattice(&0)
            .unwrap()
            .apply_given_position(|pos, st| {
                if pos == (0, sample.a) || pos == (1, sample.b) {
                    st.firing_times = vec![0.];
                }
            });
        network.run_lattices(2)?;
        let lattice = network.get_lattice(&1).unwrap();
        let mut best = 0usize;
        let mut best_v = f32::MIN;
        for (i, row) in lattice.cell_grid().iter().enumerate() {
            let v = row[0].current_voltage;
            if v > best_v {
                best_v = v;
                best = i;
            }
        }
        if best == sample.sum {
            correct += 1;
        }
    }
    println!(
        "Accuracy: {:.2}%",
        correct as f32 / test.len() as f32 * 100.
    );
    Ok(())
}
