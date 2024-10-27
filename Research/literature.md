# Collection of literature

* [Audio Signal Processing for Machine Learning Youtube Course](https://www.youtube.com/watch?v=iCwMQJnKk2c&list=PL-wATfeyAMNqIee7cH3q1bh4QJFAaeNv0&ab_channel=ValerioVelardo-TheSoundofAI)
    * Introduction to audio signal features and preprocesssing
    * Covers everything from basics of digital sound signals to Mel spectrograms and MFCCs

* [High-Quality and Reproducible Automatic Drum Transcription From Crowdsourced Data](https://www.mdpi.com/2624-6120/4/4/42)
    * Study of how crowdsourced data can positively impact training of ADT models
    * Introduces some basic DL architectures and techniques:
        * Frame-synchronous CNN encoder
        * Tatum-synchronous CNN encoder
        * RNN decoder
        * Self-attention decoder
    * Gives a small overview over ADT related works since 2018
    * Uses some popular ADT datasets

* [A Review of Automatic Drum Transcription](https://www.open-access.bcu.ac.uk/6180/1/Wu-et-al.-2018-A-review-of-automatic-drum-transcription.pdf)
    * Early review of ADT, up until 2017
    * Presents a detailed history of techniques and algorithms 

* [Analyzing and reducing the synthetic-to-real transfer gap in Music Information Retrieval: the task of automatic drum transcription](https://arxiv.org/pdf/2407.19823)
    * Analyzing why syntheticly generated data for ADT performs worse
    * Proves that applying realistic distributions and properties to synthetic data makes models perform better on ADT/DTM tasks
    * Shows that real data still is better
    * Introduces a synthetic but realisticly distributed dataset

* [IN-DEPTH PERFORMANCE ANALYSIS OF THE ADTOF-BASED ALGORITHM FOR AUTOMATIC DRUM TRANSCRIPTION](https://diva-portal.org/smash/get/diva2:1887345/FULLTEXT01.pdf)
    * Analyzes how precise data within the crowdsourced datasets ADTOF-YT and ADTOF-RGW are
    * Introduces new performance measure, Octave F-measure, and the psuedo confusion matrix
    * It highlights some usual inconsistencies in the datasets
    * It claims models often have high accuracies, but are capped by quality of annotations

* [ADTOF: A LARGE DATASET OF NON-SYNTHETIC MUSIC FOR AUTOMATIC DRUM TRANSCRIPTION](https://archives.ismir.net/ismir2021/paper/000102.pdf)
    * Is an introduction of the ADTOF dataset
    * Covers the general problem description, model training, performance estimation, etc. using well known conventions
    * Good analysis over the results
    * It also claims that big crowdsourced datasets using real music gives a better result

* [AUTOMATIC DRUM TRANSCRIPTION FOR POLYPHONIC RECORDINGS USING SOFT ATTENTION MECHANISMS AND CONVOLUTIONAL NEURAL NETWORKS](https://archives.ismir.net/ismir2017/paper/000146.pdf)
    * Early view of convolutional and soft attention mechanisms in DTM
    * Shows that convolutions and attention are beneficial additions
    * Introduces a alternative peak picking algorithm which does not perform better
    * Trains a small network over small datasets

* [MEDLEYDB 2.0 : NEW DATA AND A SYSTEM FOR SUSTAINABLE DATA COLLECTION](https://rachelbittner.weebly.com/uploads/3/2/1/8/32182799/bittner_ismirlbd-mdb_2016.pdf)
    * Introduces the MDB2.0 dataset
    * Requires citation for use