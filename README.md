# ReadoutWithRL

This is the Github for the research project, Using Deep Reinforcement Learning for Pulse Optimization in Superconducting Qubit Readout. In this project, DRL techniques are used to study the optimal measurement pulse waveform that maximises fidelity in qubit state discrimination as well as minimizes the total durations of measurement.

In Part 1, the focus is on resonator pulse optimization in single qubit-resonator systems. I first construct a Langevin Dynamics based simulation of the resonator state dynamics under an input drive (the measurement), and use the simulated dynamics of the resonator state to obtain the approximate maximum fidelity, as well as the total duration till the resonator returns to vacuum state. I pair this simulation with various on-policy DRL algorithms (PPO, TRPO, A2C, RecurrentPPO) implemented using StableBaselines3. To test this out yourself, ensure the necessary dependencies are installed in the requirements.txt file, and then run any of the algo files in the command line. While the algorithm trains, to measure the training performance, in a seperate terminal type "tensorboard --logdir logs" to view the mean episodic rewards, learning rates and more in real time. Finally, once a sufficient training time has passed, you can run a specific version of the DRL model using the instructions on the grapher.py file.

In Part 2, extensions will be made to simulataneous qubit and resonator pulse operations. This will primarily be to actively reset the qubit during measurement such that a delay isn't needed to account for the Qubit Lifetimes.

In Part 3, experiments will be run on the ibmq_jakarta backend to verify the performance of RL techniques for readout optimization. All details of device charecterization and experiments to quantify the performance of the RL pulses will be included here.

In Part 4, the single qubit-resonator optimizations will be extended to frequency multiplexed systems, including muliplexed qubit-resonator systems, as well as multiple qubits to one resonator systems where selective and simultaneous qubit measurements are important to achieve.
