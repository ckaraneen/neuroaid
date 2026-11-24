# neuroaid

:zap: :books: Papers, courses, datasets and tools for getting started with **Neuro-AI**! :brain: :boom:

![neuroaid](./brain.gif)

> New here? Start with the [Top 12](#1-start-here-top-12), then pick an [entry track](#3-entry-tracks).
>
> *Updated November 2026. The previous list lives in [`ARCHIVE.md`](ARCHIVE.md).*

---

## What is NeuroAI?

People use the term in a few different ways. This list covers all of them:

1. Using trained neural networks as models of the brain (§6–9, §12–14, §19).
2. Using machine learning to analyze and decode neural data (§5, §15, §16, §18).
3. Using ideas from neuroscience to build better AI (§10, §11, §24).
4. Studying brains and AI systems as solutions to the same problems (§6, §19, §22, §24).

It overlaps a lot with computational neuroscience, cognitive science and neural engineering.

---

## How to use this map

| Symbol | Meaning |
|---|---|
| ★ | Where to start in each section |
| **L1** | Introductory |
| **L2** | Assumes some neuroscience or ML background |
| **L3** | Technical or specialized |
| 🛠 | Hands-on exercise. *(Colab)* means it runs on free Google Colab |
| *(preprint)* | Not yet peer-reviewed |

**Ground rules**
1. Reviews are usually easier to start with than original papers.
2. Avoid thinking of the brain as "region X does function Y". Brain regions and functions don't map one-to-one, and inferring a mental state from activity in one region (*reverse inference*) is a common mistake.
3. A model that predicts neural data doesn't necessarily explain it. See §4 and §22.
4. Many BCI and clinical results come from one or a few people.

---

## Contents

**Getting started**
1. [Start Here: Top 12](#1-start-here-top-12)
2. [Prerequisites](#2-prerequisites)
3. [Entry tracks](#3-entry-tracks)
4. [How to evaluate a model: a methods primer](#4-how-to-evaluate-a-model-a-methods-primer)

**Core topics**

5. [Measuring the brain: neuroimaging & neural data](#5-measuring-the-brain-neuroimaging--neural-data)
6. [NeuroAI as a field: perspectives](#6-neuroai-as-a-field-perspectives)
7. [Deep-network models of sensory cortex: vision & audition](#7-deep-network-models-of-sensory-cortex-vision--audition)
8. [Population dynamics & neural manifolds](#8-population-dynamics--neural-manifolds)
9. [RNNs as models of cognition (incl. attention & meta-learning)](#9-rnns-as-models-of-cognition)
10. [Learning: plasticity, credit assignment, RL & dopamine](#10-learning-plasticity-credit-assignment-rl--dopamine)
11. [Memory, hippocampus & cognitive maps](#11-memory-hippocampus--cognitive-maps)
12. [Language & LLM–brain alignment](#12-language--llmbrain-alignment)
13. [Motor control & embodiment](#13-motor-control--embodiment)
14. [Cognitive modeling, behavior & development](#14-cognitive-modeling-behavior--development)

**Recent directions**

15. [Foundation models for neural data](#15-foundation-models-for-neural-data)
16. [Brain–computer interfaces & decoding](#16-braincomputer-interfaces--decoding)
17. [Connectomics & whole-organism models](#17-connectomics--whole-organism-models)
18. [Behavior quantification](#18-behavior-quantification)
19. [Representational alignment & interpretability](#19-representational-alignment--interpretability)

**Classics, critiques & theory**

20. [Systems & cognitive neuroscience classics](#20-systems--cognitive-neuroscience-classics)
21. [Causal tools, perturbations & biophysical models](#21-causal-tools-perturbations--biophysical-models)
22. [Critiques & open debates](#22-critiques--open-debates)
23. [Clinical neuromodulation & computational psychiatry](#23-clinical-neuromodulation--computational-psychiatry)
24. [Theory](#24-theory)
25. [Ethics & neurorights](#25-ethics--neurorights)

**Resources**

26. [Toolbox: datasets, software, benchmarks](#26-toolbox-datasets-software-benchmarks)
27. [Textbooks & courses](#27-textbooks--courses)
28. [Keeping up](#28-keeping-up)
29. [Careers & finding labs](#29-careers--finding-labs)
30. [Other reading lists worth bookmarking](#30-other-reading-lists-worth-bookmarking)

---

## 1. Start Here: Top 12

If you only read twelve things, read these, roughly in this order.

1. **Marr (1982).** [*Vision*](https://doi.org/10.7551/mitpress/9780262514620.001.0001), **chapter 1**. Marr's three levels of analysis.
2. **Richards et al. (2019).** [*A deep learning framework for neuroscience.*](https://doi.org/10.1038/s41593-019-0520-2) Nat Neurosci 22:1761–1770.
3. **Zador et al. (2023).** [*Catalyzing next-generation Artificial Intelligence through NeuroAI.*](https://doi.org/10.1038/s41467-023-37180-x) Nat Commun 14:1597. How neuroscience can help AI.
4. **Lindsay (2021).** [*Convolutional neural networks as a model of the visual system: past, present, and future.*](https://doi.org/10.1162/jocn_a_01544) J Cogn Neurosci 33:2017–2031. Free version: [arXiv:2001.07092](https://arxiv.org/abs/2001.07092).
5. **Schrimpf et al. (2021).** [*The neural architecture of language: integrative modeling converges on predictive processing.*](https://doi.org/10.1073/pnas.2105646118) PNAS 118:e2105646118. Language models compared with the brain's language network.
6. **Vyas, Golub, Sussillo & Shenoy (2020).** [*Computation through neural population dynamics.*](https://doi.org/10.1146/annurev-neuro-092619-094115) Annu Rev Neurosci 43:249–275.
7. **Sussillo & Barak (2013).** [*Opening the black box: low-dimensional dynamics in high-dimensional recurrent neural networks.*](https://doi.org/10.1162/NECO_a_00409) Neural Comput 25:626–649.
8. **Lillicrap, Santoro, Marris, Akerman & Hinton (2020).** [*Backpropagation and the brain.*](https://doi.org/10.1038/s41583-020-0277-3) Nat Rev Neurosci 21:335–346.
9. **Behrens et al. (2018).** [*What is a cognitive map? Organizing knowledge for flexible behavior.*](https://doi.org/10.1016/j.neuron.2018.10.002) Neuron 100:490–509.
10. **Wang et al. (2025).** [*Foundation model of neural activity predicts response to new stimulus types.*](https://doi.org/10.1038/s41586-025-08829-y) Nature 640:470–477. A foundation model of mouse visual cortex.
11. **International Brain Laboratory et al. (2025).** [*A brain-wide map of neural activity during complex behaviour.*](https://doi.org/10.1038/s41586-025-09235-0) Nature 645:177–191. Brain-wide recordings from 12 labs.
12. **Bowers et al. (2023).** [*Deep problems with neural network models of human vision.*](https://doi.org/10.1017/S0140525X22002813) Behav Brain Sci 46:e385. A critique of deep nets as models of vision.

---

## 2. Prerequisites

You can pick these up as you go.

- **Math:** linear algebra (eigendecomposition, SVD/PCA), probability & statistics (Bayes, likelihoods, regression), calculus & ODEs, a bit of dynamical systems (fixed points, stability).
  - [*Mathematics for Machine Learning*](https://mml-book.github.io/) (Deisenroth, Faisal & Ong; free) · Strogatz, *Nonlinear Dynamics and Chaos*
- **Programming:** Python, NumPy, PyTorch or JAX, Jupyter/Colab.
- **Deep learning:** Prince, [*Understanding Deep Learning*](https://udlbook.github.io/udlbook/) (2023, free PDF) · Zhang, Lipton, Li & Smola, [*Dive into Deep Learning*](https://d2l.ai/) (free) · [Neuromatch Deep Learning course](https://deeplearning.neuromatch.io/).
- **Neuroscience 101:** neurons, synapses, action potentials, the visual system, major brain areas.
  - Bear, Connors & Paradiso, *Neuroscience: Exploring the Brain* · Luo, *Principles of Neurobiology* · MIT OCW 9.13 *The Human Brain* (Kanwisher).
- **Popular books:** Lindsay, *Models of the Mind* (2021) · Cobb, *The Idea of the Brain* (2020).

---

## 3. Entry tracks

### Track A: from ML/CS
1. Bear or Luo chapters on neurons, synapses and vision; Lindsay's *Models of the Mind*.
2. [Neuromatch Computational Neuroscience](https://compneuro.neuromatch.io/): the intro modules (model types, model fitting, GLMs, dimensionality reduction).
3. §4 (methods), then §6–7. 🛠 the Brain-Score exercise in §7.
4. §8–9. 🛠 the neurogym exercise in §9.
5. §14 (cognitive modeling).
6. Pick one direction: foundation models (§15), BCI (§16), or connectomes (§17).
7. Finish with the critiques (§22).

### Track B: from biology/psychology
1. Python + Prince chapters 1–7 (or *Dive into Deep Learning*).
2. [Neuromatch Deep Learning](https://deeplearning.neuromatch.io/): the intro modules.
3. §4, then the Glaser et al. 2020 decoding tutorial (§16).
4. §6–7.
5. §19: 🛠 the representational-similarity exercise.
6. Then §9 (RNNs) or §10 (learning rules).

### Track C: clinical/neurotech
§5 (methods) → §16 (BCI) → §23 (neuromodulation & computational psychiatry) → §25 (ethics) → 🛠 Brain-to-Text '25 (§16).

### Then pick a direction
- **Applications:** §16 + §23 + §25
- **Science:** §8–11 + §14 + §20 + §24
- **Scale:** §15 + §17 + §5 (large datasets)
- **Representations:** §7 + §12 + §19
- **Embodiment:** §13 + §17 + §18

---

## 4. How to evaluate a model: a methods primer

How to tell whether a model–brain comparison is convincing.

- ★ **Kriegeskorte, Simmons, Bellgowan & Baker (2009).** [*Circular analysis in systems neuroscience: the dangers of double dipping.*](https://doi.org/10.1038/nn.2303) Nat Neurosci 12:535–540. Why selecting and testing on the same data inflates results. **L1**
- **Naselaris, Kay, Nishimoto & Gallant (2011).** [*Encoding and decoding in fMRI.*](https://doi.org/10.1016/j.neuroimage.2010.07.073) NeuroImage 56:400–410. Encoding vs. decoding models. **L1**
- **Varoquaux (2018).** [*Cross-validation failure: small sample sizes lead to large error bars.*](https://doi.org/10.1016/j.neuroimage.2017.06.061) NeuroImage 180:68–77. Cross-validation with small samples is noisy. **L1**
- **Wilson & Collins (2019).** [*Ten simple rules for the computational modeling of behavioral data.*](https://doi.org/10.7554/eLife.49547) eLife 8:e49547. Practical advice on fitting and comparing models. **L1**
- **Questions to ask about any model–brain comparison:**
  - Is it tested on new stimuli, subjects or tasks, or only new trials?
  - Does it beat simple baselines (an untrained network, a linear model) and how close is it to the noise ceiling?
  - Could test data have leaked into training?
  - Would very different models fit just as well?
  - Does it also explain behavior, or only neural responses?

---

## 5. Measuring the brain: neuroimaging & neural data

Learn how fMRI, EEG/MEG, ECoG, Neuropixels and calcium imaging differ in spatial and temporal resolution, coverage and invasiveness.
**Tools:** [`nilearn`](https://nilearn.github.io/), [`MNE-Python`](https://mne.tools/), [`pynwb`](https://pynwb.readthedocs.io/), [`SpikeInterface`](https://github.com/SpikeInterface/spikeinterface), [`Suite2p`](https://github.com/MouseLand/suite2p).

- ★ **Poldrack (2006).** [*Can cognitive processes be inferred from neuroimaging data?*](https://doi.org/10.1016/j.tics.2005.12.004) Trends Cogn Sci 10:59–63. The reverse-inference problem. **L1**
- **Ogawa, Lee, Kay & Tank (1990).** [*Brain magnetic resonance imaging with contrast dependent on blood oxygenation.*](https://doi.org/10.1073/pnas.87.24.9868) PNAS 87:9868–9872. The discovery of the BOLD signal. **L1**
- **Logothetis et al. (2001).** [*Neurophysiological investigation of the basis of the fMRI signal.*](https://doi.org/10.1038/35084005) Nature 412:150–157. What fMRI actually reflects. **L2**
- **Glasser et al. (2016).** [*A multi-modal parcellation of human cerebral cortex.*](https://doi.org/10.1038/nature18933) Nature 536:171–178. 180 areas per hemisphere. **L2**
- **Jun et al. (2017).** [*Fully integrated silicon probes for high-density recording of neural activity.*](https://doi.org/10.1038/nature24636) Nature 551:232–236. Neuropixels. **L2**
- **Steinmetz et al. (2021).** [*Neuropixels 2.0: a miniaturized high-density probe for stable, long-term brain recordings.*](https://doi.org/10.1126/science.abf4588) Science 372:eabf4588. **L2**
- **Stringer et al. (2019).** [*Spontaneous behaviors drive multidimensional, brainwide activity.*](https://doi.org/10.1126/science.aav7893) Science 364:eaav7893. Behavior is everywhere in the brain. **L2**
- **Stringer et al. (2019).** [*High-dimensional geometry of population responses in visual cortex.*](https://doi.org/10.1038/s41586-019-1346-5) Nature 571:361–365. **L3**
- **Allen et al. (2022).** [*A massive 7T fMRI dataset to bridge cognitive neuroscience and artificial intelligence.*](https://doi.org/10.1038/s41593-021-00962-x) Nat Neurosci 25:116–126. The Natural Scenes Dataset (NSD). **L1**
- **de Vries, Siegle & Koch (2023).** [*Sharing neurophysiology data from the Allen Brain Observatory.*](https://doi.org/10.7554/eLife.85550) eLife 12:e85550. Practical guide to the Allen data. **L1**
- **International Brain Laboratory et al. (2025).** [*A brain-wide map of neural activity during complex behaviour.*](https://doi.org/10.1038/s41586-025-09235-0) Nature 645:177–191. Recordings from 139 mice across 279 brain areas. **L2**
- **Findling, Hubert et al. / IBL (2025).** [*Brain-wide representations of prior information in mouse decision-making.*](https://doi.org/10.1038/s41586-025-09226-1) Nature 645:192–200. **L3**
- **Chen et al. (2024).** [*Brain-wide neural activity underlying memory-guided movement.*](https://doi.org/10.1016/j.cell.2023.12.035) Cell 187:676–691. **L3**
- **Pachitariu, Sridhar, Pennington & Stringer (2024).** [*Spike sorting with Kilosort4.*](https://doi.org/10.1038/s41592-024-02232-7) Nat Methods. **L3**

🛠 *(Colab)* Load one IBL session with the [ONE API](https://docs.internationalbrainlab.org/) (or one [Allen Brain Observatory](https://allensdk.readthedocs.io/) session) and plot trial-averaged firing rates for neurons in two brain areas, split by the animal's choice.

---

## 6. NeuroAI as a field: perspectives

Overviews of the field.

- ★ **Richards et al. (2019).** [*A deep learning framework for neuroscience.*](https://doi.org/10.1038/s41593-019-0520-2) Nat Neurosci 22:1761–1770. **L1**
- **Zador et al. (2023).** [*Catalyzing next-generation Artificial Intelligence through NeuroAI.*](https://doi.org/10.1038/s41467-023-37180-x) Nat Commun 14:1597. Proposes an "embodied Turing test". **L1**
- **Kriegeskorte (2015).** [*Deep neural networks: a new framework for modeling biological vision and brain information processing.*](https://doi.org/10.1146/annurev-vision-082114-035447) Annu Rev Vis Sci 1:417–446. **L1**
- **Hassabis, Kumaran, Summerfield & Botvinick (2017).** [*Neuroscience-inspired artificial intelligence.*](https://doi.org/10.1016/j.neuron.2017.06.011) Neuron 95:245–258. **L1**
- **Yang & Wang (2020).** [*Artificial neural networks for neuroscientists: a primer.*](https://doi.org/10.1016/j.neuron.2020.09.005) Neuron 107:1048–1070. **L1**
- **Marblestone, Wayne & Kording (2016).** [*Toward an integration of deep learning and neuroscience.*](https://doi.org/10.3389/fncom.2016.00094) Front Comput Neurosci 10:94. **L2**
- **Hasson, Nastase & Goldstein (2020).** [*Direct fit to nature: an evolutionary perspective on biological and artificial neural networks.*](https://doi.org/10.1016/j.neuron.2019.12.002) Neuron 105:416–434. **L2**
- **Saxe, Nelli & Summerfield (2021).** [*If deep learning is the answer, what is the question?*](https://doi.org/10.1038/s41583-020-00395-8) Nat Rev Neurosci 22:55–67. **L2**
- **Doerig et al. (2023).** [*The neuroconnectionist research programme.*](https://doi.org/10.1038/s41583-023-00705-w) Nat Rev Neurosci 24:431–450. **L2**
- **Mineault et al. (2024).** [*NeuroAI for AI Safety.*](https://arxiv.org/abs/2411.18526) arXiv:2411.18526 *(preprint)*. A long survey of open directions. **L2–L3**
- **Wang (2026).** [*Toward a foundational brain model of intelligence.*](https://www.cns.nyu.edu/wanglab/publications/pdf/wang.conb2026.pdf) Curr Opin Neurobiol 98:103198. **L2**

---

## 7. Deep-network models of sensory cortex: vision & audition

- ★ **Yamins & DiCarlo (2016).** [*Using goal-driven deep learning models to understand sensory cortex.*](https://doi.org/10.1038/nn.4244) Nat Neurosci 19:356–365. **L1**
- **Yamins et al. (2014).** [*Performance-optimized hierarchical models predict neural responses in higher visual cortex.*](https://doi.org/10.1073/pnas.1403112111) PNAS 111:8619–8624. One of the first papers to show this works. **L2**
- **Kay, Naselaris, Prenger & Gallant (2008).** [*Identifying natural images from human brain activity.*](https://doi.org/10.1038/nature06713) Nature 452:352–355. Encoding models for decoding. **L2**
- **Kriegeskorte, Mur & Bandettini (2008).** [*Representational similarity analysis – connecting the branches of systems neuroscience.*](https://doi.org/10.3389/neuro.06.004.2008) Front Syst Neurosci 2:4. **L1**
- **Schrimpf et al. (2020).** [*Integrative benchmarking to advance neurally mechanistic models of human intelligence*](https://doi.org/10.1016/j.neuron.2020.07.040) (Brain-Score). Neuron 108:413–423. **L2**
- **Kar et al. (2019).** [*Evidence that recurrent circuits are critical to the ventral stream's execution of core object recognition behavior.*](https://doi.org/10.1038/s41593-019-0392-5) Nat Neurosci 22:974–983. **L2**
- **Bashivan, Kar & DiCarlo (2019).** [*Neural population control via deep image synthesis.*](https://doi.org/10.1126/science.aav9436) Science 364:eaav9436. **L2**
- **Walker et al. (2019).** [*Inception loops discover what excites neurons most using deep predictive models.*](https://doi.org/10.1038/s41593-019-0517-x) Nat Neurosci 22:2060–2065. **L2**
- **Ponce et al. (2019).** [*Evolving images for visual neurons using a deep generative network reveals coding principles and neuronal preferences.*](https://doi.org/10.1016/j.cell.2019.04.005) Cell 177:999–1009. **L2**
- **Feather, Leclerc, Mądry & McDermott (2023).** [*Model metamers reveal divergent invariances between biological and artificial neural networks.*](https://doi.org/10.1038/s41593-023-01442-0) Nat Neurosci 26:2017–2034. **L2**
- **Wang et al. (2025).** [*Foundation model of neural activity predicts response to new stimulus types.*](https://doi.org/10.1038/s41586-025-08829-y) Nature 640:470–477. Code: [cajal/fnn](https://github.com/cajal/fnn). **L2**
- **Gokce & Schrimpf (2024).** [*Scaling laws for task-optimized models of the primate visual ventral stream.*](https://arxiv.org/abs/2411.05712) arXiv:2411.05712. Bigger models don't keep getting more brain-like. **L2**

### Audition
- **Kell, Yamins, Shook, Norman-Haignere & McDermott (2018).** [*A task-optimized neural network replicates human auditory behavior, predicts brain responses, and reveals a cortical processing hierarchy.*](https://doi.org/10.1016/j.neuron.2018.03.044) Neuron 98:630–644. **L2**
- **Tuckute, Feather, Boebinger & McDermott (2023).** [*Many but not all deep neural network audio models capture brain responses and exhibit correspondence between model stages and brain regions.*](https://doi.org/10.1371/journal.pbio.3002366) PLoS Biol 21:e3002366. **L2**

### Multimodal & benchmark challenges
- **Gifford et al. (2025).** [*The Algonauts Project 2025 Challenge: How the Human Brain Makes Sense of Multimodal Movies.*](https://arxiv.org/abs/2501.00504) arXiv:2501.00504. **L2**
- **d'Ascoli et al. (2025).** [*TRIBE: TRImodal Brain Encoder for whole-brain fMRI response prediction.*](https://arxiv.org/abs/2507.22229) arXiv:2507.22229 *(preprint)*. The winning Algonauts 2025 entry. **L2**
- **Scotti & Tripathy (2025).** [*Insights from the Algonauts 2025 Winners.*](https://arxiv.org/abs/2508.10784) arXiv:2508.10784. All top teams used pretrained features with simple encoders. **L2**

🛠 *(Colab)* On [Brain-Score](https://www.brain-score.org/), compare a supervised ResNet-50 with a self-supervised ResNet-50 on the public V4 and IT neural benchmarks. Is the difference meaningful compared with the noise ceiling?

---

## 8. Population dynamics & neural manifolds

- ★ **Vyas, Golub, Sussillo & Shenoy (2020).** [*Computation through neural population dynamics.*](https://doi.org/10.1146/annurev-neuro-092619-094115) Annu Rev Neurosci 43:249–275. **L1**
- **Cunningham & Yu (2014).** [*Dimensionality reduction for large-scale neural recordings.*](https://doi.org/10.1038/nn.3776) Nat Neurosci 17:1500–1509. **L1**
- **Churchland et al. (2012).** [*Neural population dynamics during reaching.*](https://doi.org/10.1038/nature11129) Nature 487:51–56. **L2**
- **Gallego, Perich, Miller & Solla (2017).** [*Neural manifolds for the control of movement.*](https://doi.org/10.1016/j.neuron.2017.05.025) Neuron 94:978–984. **L2**
- **Gallego et al. (2020).** [*Long-term stability of cortical population dynamics underlying consistent behavior.*](https://doi.org/10.1038/s41593-019-0555-4) Nat Neurosci 23:260–270. **L2**
- **Chung & Abbott (2021).** [*Neural population geometry: an approach for understanding biological and artificial neural networks.*](https://doi.org/10.1016/j.conb.2021.10.010) Curr Opin Neurobiol 70:137–144. **L2**
- **Jazayeri & Ostojic (2021).** [*Interpreting neural computations by examining intrinsic and embedding dimensionality of neural activity.*](https://doi.org/10.1016/j.conb.2021.08.002) Curr Opin Neurobiol 70:113–120. **L2**
- **Pandarinath et al. (2018).** [*Inferring single-trial neural population dynamics using sequential auto-encoders*](https://doi.org/10.1038/s41592-018-0109-9) (LFADS). Nat Methods 15:805–815. **L3**
- **Schneider, Lee & Mathis (2023).** [*Learnable latent embeddings for joint behavioural and neural analysis*](https://doi.org/10.1038/s41586-023-06031-6) (CEBRA). Nature 617:360–368. **L2**
- **Durstewitz, Koppe & Thurm (2023).** [*Reconstructing computational system dynamics from neural data with recurrent neural networks.*](https://doi.org/10.1038/s41583-023-00740-7) Nat Rev Neurosci 24:693–710. **L3**

🛠 *(Colab)* Using the Steinmetz et al. 2019 Neuropixels data loader in the [Neuromatch course book](https://compneuro.neuromatch.io/), run PCA and [CEBRA](https://cebra.ai/) on one session and see which separates left and right choices better.

---

## 9. RNNs as models of cognition

Includes attention and meta-learning.

- ★ **Sussillo & Barak (2013).** [*Opening the black box: low-dimensional dynamics in high-dimensional recurrent neural networks.*](https://doi.org/10.1162/NECO_a_00409) Neural Comput 25:626–649. **L2**
- **Mante, Sussillo, Shenoy & Newsome (2013).** [*Context-dependent computation by recurrent dynamics in prefrontal cortex.*](https://doi.org/10.1038/nature12742) Nature 503:78–84. **L2**
- **Song, Yang & Wang (2016).** [*Training excitatory-inhibitory recurrent neural networks for cognitive tasks: a simple and flexible framework.*](https://doi.org/10.1371/journal.pcbi.1004792) PLoS Comput Biol 12:e1004792. **L2**
- **Yang, Joglekar, Song, Newsome & Wang (2019).** [*Task representations in neural networks trained to perform many cognitive tasks.*](https://doi.org/10.1038/s41593-018-0310-2) Nat Neurosci 22:297–306. **L2**
- **Rajan, Harvey & Tank (2016).** [*Recurrent network models of sequence generation and memory.*](https://doi.org/10.1016/j.neuron.2016.02.009) Neuron 90:128–142. **L2**
- **Wang et al. (2018).** [*Prefrontal cortex as a meta-reinforcement learning system.*](https://doi.org/10.1038/s41593-018-0147-8) Nat Neurosci 21:860–868. **L2**
- **Driscoll, Shenoy & Sussillo (2024).** [*Flexible multitask computation in recurrent networks utilizes shared dynamical motifs.*](https://doi.org/10.1038/s41593-024-01668-6) Nat Neurosci 27:1349–1363. **L3**
- **Lindsay & Miller (2018).** [*How biological attention mechanisms improve task performance in a large-scale visual system model.*](https://doi.org/10.7554/eLife.38105) eLife 7:e38105. Top-down attention as feature-similarity gain. **L2**

🛠 *(Colab)* Train a vanilla RNN on the perceptual-decision task in [`neurogym`](https://github.com/neurogym/neurogym), then find its fixed points as in Sussillo & Barak.

---

## 10. Learning: plasticity, credit assignment, RL & dopamine

- ★ **Lillicrap, Santoro, Marris, Akerman & Hinton (2020).** [*Backpropagation and the brain.*](https://doi.org/10.1038/s41583-020-0277-3) Nat Rev Neurosci 21:335–346. **L1**

### Plasticity & biologically plausible learning
- **Bi & Poo (1998).** [*Synaptic modifications in cultured hippocampal neurons: dependence on spike timing, synaptic strength, and postsynaptic cell type.*](https://doi.org/10.1523/JNEUROSCI.18-24-10464.1998) J Neurosci 18:10464–10472. STDP. **L1**
- **Lillicrap, Cownden, Tweed & Akerman (2016).** [*Random synaptic feedback weights support error backpropagation for deep learning.*](https://doi.org/10.1038/ncomms13276) Nat Commun 7:13276. Feedback alignment. **L2**
- **Whittington & Bogacz (2017).** [*An approximation of the error backpropagation algorithm in a predictive coding network with local Hebbian synaptic plasticity.*](https://doi.org/10.1162/NECO_a_00949) Neural Comput 29:1229–1262. **L2**
- **Whittington & Bogacz (2019).** [*Theories of error back-propagation in the brain.*](https://doi.org/10.1016/j.tics.2018.12.005) Trends Cogn Sci 23:235–250. **L1**
- **Scellier & Bengio (2017).** [*Equilibrium propagation: bridging the gap between energy-based models and backpropagation.*](https://doi.org/10.3389/fncom.2017.00024) Front Comput Neurosci 11:24. **L2**
- **Sacramento, Costa, Bengio & Senn (2018).** [*Dendritic cortical microcircuits approximate the backpropagation algorithm.*](https://arxiv.org/abs/1810.11393) NeurIPS. **L3**
- **Bellec et al. (2020).** [*A solution to the learning dilemma for recurrent networks of spiking neurons*](https://doi.org/10.1038/s41467-020-17236-y) (e-prop). Nat Commun 11:3625. **L3**
- **Payeur, Guerguiev, Zenke, Richards & Naud (2021).** [*Burst-dependent synaptic plasticity can coordinate learning in hierarchical circuits.*](https://doi.org/10.1038/s41593-021-00857-x) Nat Neurosci 24:1010–1019. **L3**
- **Liu, Smith, Mihalas, Shea-Brown & Sümbül (2021).** [*Cell-type-specific neuromodulation guides synaptic credit assignment in a spiking neural network.*](https://doi.org/10.1073/pnas.2111821118) PNAS 118:e2111821118. **L3**
- **Hinton (2022).** [*The Forward-Forward Algorithm: Some Preliminary Investigations.*](https://arxiv.org/abs/2212.13345) arXiv:2212.13345 *(preprint)*. **L2**

### Reward, RL & dopamine
- **Schultz, Dayan & Montague (1997).** [*A neural substrate of prediction and reward.*](https://doi.org/10.1126/science.275.5306.1593) Science 275:1593–1599. Dopamine as RL prediction error. **L1**
- **Niv (2009).** [*Reinforcement learning in the brain.*](https://doi.org/10.1016/j.jmp.2008.12.005) J Math Psychol 53:139–154. A tutorial. **L1**
- **Botvinick et al. (2019).** [*Reinforcement learning, fast and slow.*](https://doi.org/10.1016/j.tics.2019.02.006) Trends Cogn Sci 23:408–422. **L1**
- **Dabney et al. (2020).** [*A distributional code for value in dopamine-based reinforcement learning.*](https://doi.org/10.1038/s41586-019-1924-6) Nature 577:671–675. **L2**

🛠 *(Colab)* Implement TD(0) learning on a simple Pavlovian conditioning task (cue → delay → reward) and plot the prediction error over trials. It should move from the reward to the cue.

---

## 11. Memory, hippocampus & cognitive maps

- ★ **Behrens et al. (2018).** [*What is a cognitive map? Organizing knowledge for flexible behavior.*](https://doi.org/10.1016/j.neuron.2018.10.002) Neuron 100:490–509. **L1**
- **Scoville & Milner (1957).** [*Loss of recent memory after bilateral hippocampal lesions.*](https://doi.org/10.1136/jnnp.20.1.11) J Neurol Neurosurg Psychiatry 20:11–21. Patient H.M. **L1**
- **O'Keefe & Dostrovsky (1971).** [*The hippocampus as a spatial map.*](https://doi.org/10.1016/0006-8993(71)90358-1) Brain Res 34:171–175. Place cells. **L1**
- **Hafting, Fyhn, Molden, Moser & Moser (2005).** [*Microstructure of a spatial map in the entorhinal cortex.*](https://doi.org/10.1038/nature03721) Nature 436:801–806. Grid cells. **L1**
- **Hopfield (1982).** [*Neural networks and physical systems with emergent collective computational abilities.*](https://doi.org/10.1073/pnas.79.8.2554) PNAS 79:2554–2558. Memories as attractors. **L2**
- **Ramsauer et al. (2021).** [*Hopfield Networks is All You Need.*](https://arxiv.org/abs/2008.02217) ICLR. Links Hopfield nets to transformer attention. **L2**
- **Kumaran, Hassabis & McClelland (2016).** [*What learning systems do intelligent agents need? Complementary learning systems theory updated.*](https://doi.org/10.1016/j.tics.2016.05.004) Trends Cogn Sci 20:512–534. **L1**
- **Stachenfeld, Botvinick & Gershman (2017).** [*The hippocampus as a predictive map.*](https://doi.org/10.1038/nn.4650) Nat Neurosci 20:1643–1653. **L2**
- **Mattar & Daw (2018).** [*Prioritized memory access explains planning and hippocampal replay.*](https://doi.org/10.1038/s41593-018-0232-z) Nat Neurosci 21:1609–1617. **L2**
- **Whittington et al. (2020).** [*The Tolman-Eichenbaum Machine: unifying space and relational memory through generalization in the hippocampal formation.*](https://doi.org/10.1016/j.cell.2020.10.024) Cell 183:1249–1263. **L2**
- **Whittington, Warren & Behrens (2022).** [*Relating transformers to models and neural representations of the hippocampal formation.*](https://arxiv.org/abs/2112.04035) ICLR. **L2**

**Do grid cells emerge in trained networks? Read these together:**
- **Banino et al. (2018).** [*Vector-based navigation using grid-like representations in artificial agents.*](https://doi.org/10.1038/s41586-018-0102-6) Nature 557:429–433. **L2**
- **Cueva & Wei (2018).** [*Emergence of grid-like representations by training recurrent neural networks to perform spatial localization.*](https://arxiv.org/abs/1803.07770) ICLR. **L2**
- **Schaeffer, Khona & Fiete (2022).** [*No free lunch from deep learning in neuroscience: a case study through models of the entorhinal-hippocampal circuit.*](https://papers.neurips.cc/paper_files/paper/2022/file/66808849a9f5d8e2d00dbdc844de6333-Paper-Conference.pdf) NeurIPS. **L2**

---

## 12. Language & LLM–brain alignment

- ★ **Huth et al. (2016).** [*Natural speech reveals the semantic maps that tile human cerebral cortex.*](https://doi.org/10.1038/nature17637) Nature 532:453–458. **L1**
- **Schrimpf et al. (2021).** [*The neural architecture of language: integrative modeling converges on predictive processing.*](https://doi.org/10.1073/pnas.2105646118) PNAS 118:e2105646118. **L2**
- **Caucheteux & King (2022).** [*Brains and algorithms partially converge in natural language processing.*](https://doi.org/10.1038/s42003-022-03036-1) Commun Biol 5:134. **L2**
- **Goldstein et al. (2022).** [*Shared computational principles for language processing in humans and deep language models.*](https://doi.org/10.1038/s41593-022-01026-4) Nat Neurosci 25:369–380. **L2**
- **Millet et al. (2022).** [*Toward a realistic model of speech processing in the brain with self-supervised learning.*](https://arxiv.org/abs/2206.01685) NeurIPS. **L2**
- **Tuckute et al. (2024).** [*Driving and suppressing the human language network using large language models.*](https://doi.org/10.1038/s41562-023-01783-7) Nat Hum Behav 8:544–561. **L2**
- **Défossez et al. (2023).** [*Decoding speech perception from non-invasive brain recordings.*](https://doi.org/10.1038/s42256-023-00714-5) Nat Mach Intell 5:1097–1107. **L2**
- **Benchetrit, Banville & King (2024).** [*Brain decoding: toward real-time reconstruction of visual perception.*](https://arxiv.org/abs/2310.19812) ICLR (MEG). **L2**

🛠 Download the [LeBel et al. 2023 narrative-listening fMRI dataset](https://openneuro.org/datasets/ds003020) (OpenNeuro `ds003020`), extract GPT-2 embeddings for the story words, and fit a ridge-regression encoding model for one subject. Compare layers, and compare with a simple word-rate baseline.

---

## 13. Motor control & embodiment

- ★ **Merel, Botvinick & Wayne (2019).** [*Hierarchical motor control in mammals and machines.*](https://doi.org/10.1038/s41467-019-13239-6) Nat Commun 10:5489. **L1**
- **Todorov & Jordan (2002).** [*Optimal feedback control as a theory of motor coordination.*](https://doi.org/10.1038/nn963) Nat Neurosci 5:1226–1235. **L2**
- **Churchland et al. (2012)** (§8).
- **Aldarondo et al. (2024).** [*A virtual rodent predicts the structure of neural activity across behaviours.*](https://doi.org/10.1038/s41586-024-07633-4) Nature 632:594–602. A simulated rat trained with deep RL. **L2**
- **Vaxenburg et al. (2025).** [*Whole-body physics simulation of fruit fly locomotion.*](https://doi.org/10.1038/s41586-025-09029-4) Nature 643:1312–1320. **L3**
- **Zador et al. (2023)** (§6).

---

## 14. Cognitive modeling, behavior & development

Models should explain behavior, not just neural data. This section links NeuroAI to computational cognitive science.

- ★ **Wilson & Collins (2019).** [*Ten simple rules for the computational modeling of behavioral data.*](https://doi.org/10.7554/eLife.49547) eLife 8:e49547. **L1**
- **Tenenbaum, Kemp, Griffiths & Goodman (2011).** [*How to grow a mind: statistics, structure, and abstraction.*](https://doi.org/10.1126/science.1192788) Science 331:1279–1285. **L1**
- **Lake, Ullman, Tenenbaum & Gershman (2017)** (§22).
- **Peterson, Bourgin, Agrawal, Reichman & Griffiths (2021).** [*Using large-scale experiments and machine learning to discover theories of human decision-making.*](https://doi.org/10.1126/science.abe2629) Science 372:1209–1214. **L2**
- **Binz et al. (2025).** [*A foundation model to predict and capture human cognition*](https://doi.org/10.1038/s41586-025-09215-4) (Centaur). Nature 644:1002–1009. An LLM fine-tuned on data from 160 psychology experiments. **L2**
- **Gottlieb & Oudeyer (2018).** [*Towards a neuroscience of active sampling and curiosity.*](https://doi.org/10.1038/s41583-018-0078-0) Nat Rev Neurosci 19:758–770. **L2**
- **Vong, Wang, Orhan & Lake (2024).** [*Grounded language acquisition through the eyes and ears of a single child.*](https://doi.org/10.1126/science.adi1374) Science 383:504–511. A model trained on headcam video from one child learns words. **L2**
- **Zador (2019)** (§22).

🛠 *(Colab)* Using the code accompanying Wilson & Collins 2019, simulate a two-armed bandit learner, fit the model, and check that you recover the parameters.

---

## 15. Foundation models for neural data

This area moves fast. See [mazabou/awesome-neurofm](https://github.com/mazabou/awesome-neurofm) for a longer list.

- ★ **Azabou et al. (2023).** [*A Unified, Scalable Framework for Neural Population Decoding*](https://arxiv.org/abs/2310.16046) (POYO). NeurIPS 36. **L3**
- **Ye & Pandarinath (2021).** [*Representation learning for neural population activity with Neural Data Transformers.*](https://arxiv.org/abs/2108.01210) arXiv:2108.01210. **L3**
- **Ye, Collinger, Wehbe & Gaunt (2023).** [*Neural Data Transformer 2: multi-context pretraining for neural spiking activity*](https://proceedings.neurips.cc/paper_files/paper/2023/hash/fe51de4e7baf52e743b679e3bdba7905-Abstract.html) (NDT2). NeurIPS. **L3**
- **Zhang et al. (2024).** [*Towards a "universal translator" for neural dynamics at single-cell, single-spike resolution*](https://arxiv.org/abs/2407.14668) (MtM). NeurIPS. **L3**
- **Azabou et al. (2025).** [*Multi-session, multi-task neural decoding from distinct cell-types and brain regions*](https://openreview.net/forum?id=IuU0wcO0mo) (POYO+). ICLR. **L3**
- **Ryoo et al. (2025).** [*Generalizable, real-time neural decoding with hybrid state-space models*](https://arxiv.org/abs/2506.05320) (POSSM). NeurIPS. **L3**
- **Ye et al. (2025).** [*A Generalist Intracortical Motor Decoder*](https://proceedings.neurips.cc/paper_files/paper/2025/hash/a00000e6a2208172700510bcd69d48e9-Abstract-Conference.html) (NDT3). NeurIPS. **L3**
- **d'Ascoli et al. (2026).** [*A foundation model of vision, audition, and language for in-silico neuroscience*](https://arxiv.org/abs/2605.04326) (TRIBE v2). arXiv:2605.04326 *(preprint)*. Trained on fMRI from 720 people. Code: [facebookresearch/tribev2](https://github.com/facebookresearch/tribev2). **L2**
- **Bertram et al. (2026).** [*How 'Neural' is a Neural Foundation Model?*](https://arxiv.org/abs/2601.21508) arXiv:2601.21508 *(preprint)*. A critical look at Wang et al. 2025. **L2**
- See also **Wang et al. 2025** (§7) and **CEBRA** (§8).

**Benchmarks**
- **Pei et al. (2021).** [*Neural Latents Benchmark '21: Evaluating latent variable models of neural population activity.*](https://arxiv.org/abs/2109.04463) NeurIPS Datasets & Benchmarks. [Project site](https://neurallatents.github.io/). **L2**
- **Karpowicz et al. (2024).** [*Few-shot Algorithms for Consistent Neural Decoding (FALCON) Benchmark.*](https://proceedings.neurips.cc/paper_files/paper/2024/hash/8c2e6bb15be1894b8fb4e0f9bcad1739-Abstract.html) NeurIPS Datasets & Benchmarks. [Project site](https://snel-repo.github.io/falcon/). **L2**

Also see *The Transmitter*'s [article on the debate over brain foundation models](https://www.thetransmitter.org/systems-neuroscience/future-of-ai-brain-models-sparks-debate/) (2026).

🛠 Submit a simple baseline (smoothed spikes + ridge regression) to the [Neural Latents Benchmark](https://neurallatents.github.io/), then compare with a pretrained POYO model via [`torch_brain`](https://torch-brain.readthedocs.io/). *(Colab)* Or try the TRIBE v2 demo on a short video.

---

## 16. Brain–computer interfaces & decoding

- ★ **Glaser et al. (2020).** [*Machine learning for neural decoding.*](https://doi.org/10.1523/ENEURO.0506-19.2020) eNeuro 7:ENEURO.0506-19.2020. A hands-on tutorial with code. **L1**
- **Hochberg et al. (2006).** [*Neuronal ensemble control of prosthetic devices by a human with tetraplegia.*](https://doi.org/10.1038/nature04970) Nature 442:164–171. The first BrainGate implant. **L1**
- **Anumanchipalli, Chartier & Chang (2019).** [*Speech synthesis from neural decoding of spoken sentences.*](https://doi.org/10.1038/s41586-019-1119-1) Nature 568:493–498. **L2**
- **Willett et al. (2021).** [*High-performance brain-to-text communication via handwriting.*](https://doi.org/10.1038/s41586-021-03506-2) Nature 593:249–254. **L2**
- **Willett et al. (2023).** [*A high-performance speech neuroprosthesis.*](https://doi.org/10.1038/s41586-023-06377-x) Nature 620:1031–1036. Intracortical arrays. **L2**
- **Metzger et al. (2023).** [*A high-performance neuroprosthesis for speech decoding and avatar control.*](https://doi.org/10.1038/s41586-023-06443-4) Nature 620:1037–1046. ECoG; read together with Willett 2023. **L2**
- **Card et al. (2024).** [*An accurate and rapidly calibrating speech neuroprosthesis.*](https://doi.org/10.1056/NEJMoa2314132) N Engl J Med 391:609–618. Code and data are public. **L1–L2**
- **Wairagkar et al. (2025).** [*An instantaneous voice-synthesis neuroprosthesis.*](https://doi.org/10.1038/s41586-025-09127-3) Nature 644:145–152. **L2**
- **Kunz et al. (2025).** [*Inner speech in motor cortex and implications for speech neuroprostheses.*](https://doi.org/10.1016/j.cell.2025.06.015) Cell. **L2**
- **Willett et al. (2024).** [*Brain-to-Text Benchmark '24: Lessons Learned.*](https://arxiv.org/abs/2412.17227) arXiv:2412.17227. **L2**
- **Tang, LeBel, Jain & Huth (2023).** [*Semantic reconstruction of continuous language from non-invasive brain recordings.*](https://doi.org/10.1038/s41593-023-01304-9) Nat Neurosci 26:858–866. **L2**
- **Scotti et al. (2023).** [*Reconstructing the Mind's Eye: fMRI-to-Image with Contrastive Learning and Diffusion Priors*](https://arxiv.org/abs/2305.18274) (MindEye). NeurIPS. **L2**
- **Scotti et al. (2024).** [*MindEye2: Shared-subject models enable fMRI-to-image with 1 hour of data.*](https://arxiv.org/abs/2403.11207) ICML. **L2**

Note: Neuralink, Synchron and Precision Neuroscience have mostly shared results through press releases rather than peer-reviewed papers.

🛠 Work through the [Brain-to-Text '25](https://www.kaggle.com/competitions/brain-to-text-25) competition using the Card et al. 2024 baseline code ([Neuroprosthetics-Lab/nejm-brain-to-text](https://github.com/Neuroprosthetics-Lab/nejm-brain-to-text)).

---

## 17. Connectomics & whole-organism models

- ★ **Shiu et al. (2024).** [*A Drosophila computational brain model reveals sensorimotor processing.*](https://doi.org/10.1038/s41586-024-07763-9) Nature 634:210–219. A spiking model of the whole fly brain built from its connectome. Code: [philshiu/Drosophila_brain_model](https://github.com/philshiu/Drosophila_brain_model). **L2**
- **White, Southgate, Thomson & Brenner (1986).** [*The structure of the nervous system of the nematode Caenorhabditis elegans.*](https://doi.org/10.1098/rstb.1986.0056) Phil Trans R Soc B 314:1–340. The first complete connectome. **L1**
- **Dorkenwald et al. (2024).** [*Neuronal wiring diagram of an adult brain*](https://doi.org/10.1038/s41586-024-07558-y) (FlyWire). Nature 634:124–138. **L1**
- **Schlegel et al. (2024).** [*Whole-brain annotation and multi-connectome cell typing of Drosophila.*](https://doi.org/10.1038/s41586-024-07686-5) Nature 634:139–152. **L2**
- **Lappalainen et al. (2024).** [*Connectome-constrained networks predict neural activity across the fly visual system.*](https://doi.org/10.1038/s41586-024-07939-3) Nature 634:1132–1140. Code: [TuragaLab/flyvis](https://github.com/TuragaLab/flyvis). **L2**
- **Beiran & Litwin-Kumar (2025).** [*Prediction of neural activity in connectome-constrained recurrent networks.*](https://doi.org/10.1038/s41593-025-02080-4) Nat Neurosci 28:2561–2574. Code: [emebeiran/connconstr](https://github.com/emebeiran/connconstr). **L3**
- **The MICrONS Consortium (2025).** [*Functional connectomics spanning multiple areas of mouse visual cortex.*](https://doi.org/10.1038/s41586-025-08790-w) Nature 640:435–447. **L2**
- **Ding et al. (2025).** [*Functional connectomics reveals general wiring rule in mouse visual cortex.*](https://doi.org/10.1038/s41586-025-08840-3) Nature 640:459–469. **L2**
- **Berg et al. (2026).** [*Sexual dimorphism in the complete Drosophila male central nervous system connectome.*](https://doi.org/10.1016/j.cell.2026.08.015) Cell 189:5504–5526. **L2**
- **Vaxenburg et al. (2025)** (§13).

🛠 Browse [FlyWire Codex](https://codex.flywire.ai/), then run the Shiu model and silence a neuron type of your choice.

---

## 18. Behavior quantification

- ★ **Krakauer, Ghazanfar, Gomez-Marin, MacIver & Poeppel (2017).** [*Neuroscience needs behavior: correcting a reductionist bias.*](https://doi.org/10.1016/j.neuron.2016.12.041) Neuron 93:480–490. **L1**
- **Datta, Anderson, Branson, Perona & Leifer (2019).** [*Computational neuroethology: a call to action.*](https://doi.org/10.1016/j.neuron.2019.09.038) Neuron 104:11–24. **L1**
- **Mathis et al. (2018).** [*DeepLabCut: markerless pose estimation of user-defined body parts with deep learning.*](https://doi.org/10.1038/s41593-018-0209-y) Nat Neurosci 21:1281–1289. **L1**
- **Pereira et al. (2022).** [*SLEAP: a deep learning system for multi-animal pose tracking.*](https://doi.org/10.1038/s41592-022-01426-1) Nat Methods 19:486–495. **L1**
- **Wiltschko et al. (2015).** [*Mapping sub-second structure in mouse behavior*](https://doi.org/10.1016/j.neuron.2015.11.031) (MoSeq). Neuron 88:1121–1135. **L2**

🛠 *(Colab)* Track a pose in a short video with a pretrained [DeepLabCut](https://github.com/DeepLabCut/DeepLabCut) model-zoo network.

---

## 19. Representational alignment & interpretability

- ★ **Sucholutsky et al. (2023).** [*Getting aligned on representational alignment.*](https://arxiv.org/abs/2310.13018) arXiv:2310.13018. **L2**
- **Jonas & Kording (2017).** [*Could a neuroscientist understand a microprocessor?*](https://doi.org/10.1371/journal.pcbi.1005268) PLoS Comput Biol 13:e1005268. **L1**
- **Kornblith, Norouzi, Lee & Hinton (2019).** [*Similarity of neural network representations revisited*](https://arxiv.org/abs/1905.00414) (CKA). ICML. **L3**
- **Williams, Kunz, Kornblith & Linderman (2021).** [*Generalized shape metrics on neural representations.*](https://arxiv.org/abs/2110.14739) NeurIPS. **L3**
- **Huh, Cheung, Wang & Isola (2024).** [*The Platonic Representation Hypothesis.*](https://arxiv.org/abs/2405.07987) ICML. Still debated. **L2**
- **Baker, Lansdell & Kording (2022).** [*Three aspects of representation in neuroscience.*](https://doi.org/10.1016/j.tics.2022.08.014) Trends Cogn Sci 26:942–958. **L2**

**Mechanistic interpretability as "neuroscience of ANNs"**
- **Olah et al. (2020).** [*Zoom In: An Introduction to Circuits.*](https://distill.pub/2020/circuits/zoom-in/) Distill. **L2**
- **Elhage et al. (2022).** [*Toy Models of Superposition.*](https://transformer-circuits.pub/2022/toy_model/index.html) Transformer Circuits. **L2**
- **Bricken et al. (2023).** [*Towards Monosemanticity.*](https://transformer-circuits.pub/2023/monosemantic-features) Transformer Circuits. **L2**

🛠 *(Colab)* Use [`rsatoolbox`](https://github.com/rsagroup/rsatoolbox) to compare RDMs from [NSD](https://naturalscenesdataset.org/) or [THINGS](https://things-initiative.org/) fMRI with RDMs from two vision models. Then try CKA and see if the results agree.

---

## 20. Systems & cognitive neuroscience classics

- ★ **Hubel & Wiesel (1962).** [*Receptive fields, binocular interaction and functional architecture in the cat's visual cortex.*](https://doi.org/10.1113/jphysiol.1962.sp006837) J Physiol 160:106–154. **L1**
- **Hubel & Wiesel (1959).** [*Receptive fields of single neurones in the cat's striate cortex.*](https://doi.org/10.1113/jphysiol.1959.sp006308) J Physiol 148:574–591. **L1**
- **Hodgkin & Huxley (1952).** [*A quantitative description of membrane current and its application to conduction and excitation in nerve.*](https://doi.org/10.1113/jphysiol.1952.sp004764) J Physiol 117:500–544. **L2**
- **Kanwisher, McDermott & Chun (1997).** [*The fusiform face area: a module in human extrastriate cortex specialized for face perception.*](https://doi.org/10.1523/JNEUROSCI.17-11-04302.1997) J Neurosci 17:4302–4311. **L1**
- See also §11 (memory) and §12 (language).

---

## 21. Causal tools, perturbations & biophysical models

Tools for turning neurons on and off, mapping cell types, and detailed neuron models.

- ★ **Boyden, Zhang, Bamberg, Nagel & Deisseroth (2005).** [*Millisecond-timescale, genetically targeted optical control of neural activity.*](https://doi.org/10.1038/nn1525) Nat Neurosci 8:1263–1268. Optogenetics. **L1**
- **Jazayeri & Afraz (2017).** [*Navigating the neural space in search of the neural code.*](https://doi.org/10.1016/j.neuron.2017.02.019) Neuron 93:1003–1014. What perturbation experiments can tell us. **L2**
- **Armbruster et al. (2007).** [*Evolving the lock to fit the key to create a family of G protein-coupled receptors potently activated by an inert ligand.*](https://doi.org/10.1073/pnas.0700293104) PNAS 104:5163–5168. DREADDs. **L2**
- **Chung et al. (2013).** [*Structural and molecular interrogation of intact biological systems.*](https://doi.org/10.1038/nature12107) Nature 497:332–337. CLARITY. **L2**
- **BICCN (2021).** [*A multimodal cell census and atlas of the mammalian primary motor cortex.*](https://doi.org/10.1038/s41586-021-03950-0) Nature 598:86–102. **L2**
- **Deistler et al. (2025).** [*Jaxley: differentiable simulation enables large-scale training of detailed biophysical models of neural dynamics.*](https://doi.org/10.1038/s41592-025-02895-w) Nat Methods. Fits detailed neuron models with gradient descent. Code: [jaxleyverse/jaxley](https://github.com/jaxleyverse/jaxley). **L2**

---

## 22. Critiques & open debates

Worth reading early.

- ★ **Bowers et al. (2023).** [*Deep problems with neural network models of human vision.*](https://doi.org/10.1017/S0140525X22002813) Behav Brain Sci 46:e385. **L2**
- **Lake, Ullman, Tenenbaum & Gershman (2017).** [*Building machines that learn and think like people.*](https://doi.org/10.1017/S0140525X16001837) Behav Brain Sci 40:e253. **L1**
- **Zador (2019).** [*A critique of pure learning and what artificial neural networks can learn from animal brains.*](https://doi.org/10.1038/s41467-019-11786-6) Nat Commun 10:3770. **L1**
- **Schaeffer, Khona & Fiete (2022)** (§11), **Jonas & Kording (2017)** (§19), **Krakauer et al. (2017)** (§18) and **Bertram et al. (2026)** (§15).
- **Open questions:**
  - Does predicting neural responses equal understanding?
  - Is brain alignment saturating with scale?
  - Which aspects of biology (dendrites, cell types, neuromodulators, bodies) matter computationally?
  - How should we compare representations, and models of behavior?

---

## 23. Clinical neuromodulation & computational psychiatry

- ★ **Huys, Maia & Frank (2016).** [*Computational psychiatry as a bridge from neuroscience to clinical applications.*](https://doi.org/10.1038/nn.4238) Nat Neurosci 19:404–413. **L1**
- **Mayberg et al. (2005).** [*Deep brain stimulation for treatment-resistant depression.*](https://doi.org/10.1016/j.neuron.2005.02.014) Neuron 45:651–660. **L1**
- **Scangos et al. (2021).** [*Closed-loop neuromodulation in an individual with treatment-resistant depression.*](https://doi.org/10.1038/s41591-021-01480-w) Nat Med 27:1696–1700. One patient. **L2**
- **Cole et al. (2020).** [*Stanford Accelerated Intelligent Neuromodulation Therapy for treatment-resistant depression.*](https://doi.org/10.1176/appi.ajp.2019.19070720) Am J Psychiatry 177:716–726. Open-label SAINT/SNT. **L2**
- **Cole et al. (2022).** [*Stanford Neuromodulation Therapy (SNT): a double-blind randomized controlled trial.*](https://doi.org/10.1176/appi.ajp.2021.20101429) Am J Psychiatry 179:132–141. **L2**

---

## 24. Theory

- ★ **Marr (1982).** [*Vision*](https://doi.org/10.7551/mitpress/9780262514620.001.0001) (MIT Press; 2010 reissue). Start with chapter 1. **L1**
- **Barlow (1961).** [*Possible principles underlying the transformations of sensory messages.*](https://doi.org/10.7551/mitpress/9780262518420.003.0013) In *Sensory Communication*. Efficient coding. **L1**
- **Olshausen & Field (1996).** [*Emergence of simple-cell receptive field properties by learning a sparse code for natural images.*](https://doi.org/10.1038/381607a0) Nature 381:607–609. **L2**
- **Simoncelli & Olshausen (2001).** [*Natural image statistics and neural representation.*](https://doi.org/10.1146/annurev.neuro.24.1.1193) Annu Rev Neurosci 24:1193–1216. **L2**
- **Rao & Ballard (1999).** [*Predictive coding in the visual cortex: a functional interpretation of some extra-classical receptive-field effects.*](https://doi.org/10.1038/4580) Nat Neurosci 2:79–87. **L2**
- **Pillow et al. (2008).** [*Spatio-temporal correlations and visual signalling in a complete neuronal population.*](https://doi.org/10.1038/nature07140) Nature 454:995–999. GLMs for spike trains. **L3**
- **Friston (2010).** [*The free-energy principle: a unified brain theory?*](https://doi.org/10.1038/nrn2787) Nat Rev Neurosci 11:127–138. Contested; read critically. **L3**
- **Bogacz (2017).** [*A tutorial on the free-energy framework for modelling perception and learning.*](https://doi.org/10.1016/j.jmp.2015.11.003) J Math Psychol 76:198–211. An easier introduction. **L2**
- **Roy, Jaiswal & Panda (2019).** [*Towards spike-based machine intelligence with neuromorphic computing.*](https://doi.org/10.1038/s41586-019-1677-2) Nature 575:607–617. Brain-inspired, energy-efficient hardware. **L1–L2**

---

## 25. Ethics & neurorights

- ★ **Yuste et al. (2017).** [*Four ethical priorities for neurotechnologies and AI.*](https://doi.org/10.1038/551159a) Nature 551:159–163. **L1**
- **Ienca & Andorno (2017).** [*Towards new human rights in the age of neuroscience and neurotechnology.*](https://doi.org/10.1186/s40504-017-0050-1) Life Sci Soc Policy 13:5. **L1**
- On mental privacy, see Tang et al. 2023 and Kunz et al. 2025 (§16).

---

## 26. Toolbox: datasets, software, benchmarks

### Datasets
| Type | Datasets |
|---|---|
| Human fMRI/MEG/EEG | [Natural Scenes Dataset (NSD)](https://naturalscenesdataset.org/) · [Human Connectome Project](https://www.humanconnectome.org/) · [OpenNeuro](https://openneuro.org/) · [THINGS](https://things-initiative.org/) (fMRI/MEG/EEG + behavior) · [CNeuroMod](https://docs.cneuromod.ca/) · BOLD5000 · [LeBel et al. 2023 narrative fMRI](https://openneuro.org/datasets/ds003020) |
| Animal physiology | [Allen Brain Observatory](https://allensdk.readthedocs.io/) (Neuropixels, 2-photon) · [IBL brain-wide map](https://docs.internationalbrainlab.org/) (ONE API) · [DANDI Archive](https://dandiarchive.org/) (NWB) · [CRCNS](https://crcns.org/) · Steinmetz 2019 Neuropixels (via [Neuromatch](https://compneuro.neuromatch.io/)) |
| Connectomes | [FlyWire Codex](https://codex.flywire.ai/) · [MICrONS Explorer](https://www.microns-explorer.org/) / [BossDB](https://bossdb.org/) · [neuPrint](https://neuprint.janelia.org/) (hemibrain, MANC, male CNS) |
| BCI | Card et al. 2024 data / [Brain-to-Text '25](https://www.kaggle.com/competitions/brain-to-text-25) · Willett 2023 data · [Neural Latents Benchmark](https://neurallatents.github.io/) · [FALCON](https://snel-repo.github.io/falcon/) |

### Benchmarks & challenges
[Brain-Score](https://www.brain-score.org/) · Algonauts · [Neural Latents Benchmark](https://neurallatents.github.io/) · [FALCON](https://snel-repo.github.io/falcon/) · [Brain-to-Text '25](https://www.kaggle.com/competitions/brain-to-text-25)

### Software
| Purpose | Tools |
|---|---|
| Imaging / EEG / MEG | [nilearn](https://nilearn.github.io/) · [MNE-Python](https://mne.tools/) · [fMRIPrep](https://fmriprep.org/) · [nibabel](https://nipy.org/nibabel/) · [FieldTrip](https://www.fieldtriptoolbox.org/) · [Brainstorm](https://neuroimage.usc.edu/brainstorm/) |
| Electrophysiology & imaging pipelines | [pynwb](https://pynwb.readthedocs.io/) · [SpikeInterface](https://github.com/SpikeInterface/spikeinterface) · [Kilosort4](https://github.com/MouseLand/Kilosort) · [Suite2p](https://github.com/MouseLand/suite2p) · [CaImAn](https://github.com/flatironinstitute/CaImAn) · [Cellpose](https://github.com/MouseLand/cellpose) |
| Behavior | [DeepLabCut](https://github.com/DeepLabCut/DeepLabCut) · [SLEAP](https://sleap.ai/) · [keypoint-MoSeq](https://github.com/dattalab/keypoint-moseq) |
| Simulation | [Brian2](https://briansimulator.org/) · [NEURON](https://neuron.yale.edu/) · [NEST](https://www.nest-simulator.org/) · [Arbor](https://arbor-sim.org/) · [NetPyNE](http://netpyne.org/) · [jaxley](https://github.com/jaxleyverse/jaxley) (differentiable biophysics) · [snnTorch](https://github.com/jeshraghian/snntorch) · [Norse](https://github.com/norse/norse) |
| Analysis & modeling | [rsatoolbox](https://github.com/rsagroup/rsatoolbox) · [Brain-Score](https://www.brain-score.org/) · [neurogym](https://github.com/neurogym/neurogym) · [CEBRA](https://cebra.ai/) · [flyvis](https://github.com/TuragaLab/flyvis) · [torch_brain](https://torch-brain.readthedocs.io/) |
| Experiments | [PsychoPy](https://www.psychopy.org/) · [jsPsych](https://www.jspsych.org/) |

---

## 27. Textbooks & courses

### Textbooks
- **Computational neuroscience**
  - Dayan & Abbott, *Theoretical Neuroscience* (2001)
  - Gerstner, Kistler, Naud & Paninski, [*Neuronal Dynamics*](https://neuronaldynamics.epfl.ch/) (2014; free online)
  - Miller, *An Introductory Course in Computational Neuroscience* (2018; the gentlest)
  - Izhikevich, *Dynamical Systems in Neuroscience* (2007)
  - Sterling & Laughlin, *Principles of Neural Design* (2015)
  - Cohen, *Analyzing Neural Time Series Data* (2014)
- **Neuroscience**
  - Kandel et al., *Principles of Neural Science* (6e, 2021; a reference)
  - Bear, Connors & Paradiso, *Neuroscience: Exploring the Brain*
  - Luo, *Principles of Neurobiology*
- **Machine learning**
  - Prince, [*Understanding Deep Learning*](https://udlbook.github.io/udlbook/) (2023)
  - Goodfellow, Bengio & Courville, [*Deep Learning*](https://www.deeplearningbook.org/) (2016)
  - Bishop & Bishop, *Deep Learning: Foundations and Concepts* (2024)
  - Sutton & Barto, [*Reinforcement Learning*](http://incompleteideas.net/book/the-book-2nd.html) (2e, 2018; free)
  - Zhang, Lipton, Li & Smola, [*Dive into Deep Learning*](https://d2l.ai/) (free)
- **Popular & orienting**
  - Lindsay, *Models of the Mind* (2021)
  - Cobb, *The Idea of the Brain* (2020)
  - Summerfield, *Natural General Intelligence* (2022)
  - Bennett, *A Brief History of Intelligence* (2023)
  - Hawkins, *A Thousand Brains* (2021; its theory is contested)

### Courses
- **[Neuromatch Academy](https://neuromatch.io)** (free, Colab-based): [Computational Neuroscience](https://compneuro.neuromatch.io/), [Deep Learning](https://deeplearning.neuromatch.io/), and [NeuroAI](https://neuroai.neuromatch.io/). The NeuroAI course assumes CN + DL or equivalent.
- Coursera, *Computational Neuroscience* (Rao & Fairhall, UW)
- Dan Goodman, *Neuroscience for Machine Learners*
- EPFL *Neuronal Dynamics* MOOC
- MIT OCW 9.40, *Introduction to Neural Computation* · MIT OCW 9.13, *The Human Brain*
- Cosyne and CCN tutorial recordings (YouTube) · CBMM lecture archive

### Summer schools
MBL Methods in Computational Neuroscience · CSHL Computational Neuroscience: Vision · OCNC (Okinawa) · CAJAL · IBRO-Simons Computational Neuroscience Imbizo · Telluride Neuromorphic · CBMM Brains, Minds & Machines

---

## 28. Keeping up

- **Conferences:** Cosyne · CCN (Cognitive Computational Neuroscience) · SfN · Bernstein Conference · NeurIPS/ICLR/ICML workshops (NeurReps, UniReps, Re-Align)
- **Preprint feeds:** bioRxiv Neuroscience · arXiv q-bio.NC
- **Podcasts:** *Brain Inspired* (Paul Middlebrooks) · *Theoretical Neuroscience Podcast* (Gaute Einevoll)
- **Writing:** [*The Transmitter*](https://www.thetransmitter.org/) (Simons Foundation) · Patrick Mineault's NeuroAI writing (neuroai.science, xcorr) · Distill and Transformer Circuits (interpretability) · Quanta Magazine (neuroscience)
- **Communities:** World Wide Neuro seminars · Neuromatch community · comp-neuro starter packs on Bluesky

---

## 29. Careers & finding labs

- **Find labs:** PI directories in [alirezahabib/awesome-computational-neuroscience](https://github.com/alirezahabib/awesome-computational-neuroscience); read recent Cosyne and CCN abstracts to see who works on what now.
- **Programs and support:** Neuromatch (also a network); Cosyne and CCN travel grants and mentorship programs; Flatiron Institute and Allen Institute internships and post-bacs; summer schools (§27).
- **Picking a lab:** think about whether you want a theory lab, an experimental lab, or one that does both. Ask current students about data and compute.

---

## 30. Other reading lists worth bookmarking

| List | Best for |
|---|---|
| [CYHSM/awesome-neuro-ai-papers](https://github.com/CYHSM/awesome-neuro-ai-papers) | Canonical NeuroAI papers up to 2022 |
| [mazabou/awesome-neurofm](https://github.com/mazabou/awesome-neurofm) | Neural-data foundation models (up to date) |
| [Daniel-Gong/NeuroAI-Readings](https://github.com/Daniel-Gong/NeuroAI-Readings) | Topical: geometry, transformers & hippocampus |
| [asoplata/open-computational-neuroscience-resources](https://github.com/asoplata/open-computational-neuroscience-resources) | Open data & model repositories |
| [awillats/dynamics-in-neuro-reading-list](https://github.com/awillats/dynamics-in-neuro-reading-list) | System identification & dynamics |
| [Neuromatch course books](https://neuromatch.io) | Hands-on curriculum |
| [Mineault et al. 2024, *NeuroAI for AI Safety*](https://arxiv.org/abs/2411.18526) | A survey of the current NeuroAI frontier |

---

## Contributing

PRs welcome. Please keep the list selective. Format:

```
**Authors (Year).** [*Title.*](https://doi.org/...) Venue vol:pages. Optional short note. **L1/L2/L3**
```

Include a link for every paper (DOI if possible). One paper per bullet. Mark preprints.

## License

MIT