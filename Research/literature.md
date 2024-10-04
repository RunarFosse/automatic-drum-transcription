# Collection of literature

* [Audio Signal Processing for Machine Learning Youtube Course](https://www.youtube.com/watch?v=iCwMQJnKk2c&list=PL-wATfeyAMNqIee7cH3q1bh4QJFAaeNv0&ab_channel=ValerioVelardo-TheSoundofAI)
    * Introduction to audio signal features and preprocesssing
    * Covers everything from basics of digital sound signals to Mel spectrograms and MFCCs

* [High-Quality and Reproducible Automatic Drum Transcription
From Crowdsourced Data](https://www.mdpi.com/2624-6120/4/4/42)
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