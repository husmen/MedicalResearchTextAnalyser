# MedicalResearchTextAnalyser

A system for linguistic inquiry and text analysis of medical research using UMLS.

## Getting Started

**N.B.** Tested on Linux (OpenSUSE Tumbleweed, Kernel 6.5.9-1)

### UMLS Setup

1. Acquire a license and download the [full UMLS resources](https://www.nlm.nih.gov/research/umls/licensedcontent/umlsknowledgesources.html).
1. Install MetamorphoSys by following these [instructions](https://www.nlm.nih.gov/research/umls/implementation_resources/metamorphosys/index.html).
1. Follow the [QuickUMLS initialization process](https://github.com/Georgetown-IR-Lab/QuickUMLS).

### Environment Setup

```bash
python -m venv .venv
# linux
source .venv/bin/activate
# windows
# .venv\Scripts\activate
pip install -r requirements.txt
jupyter lab MedicalResearchTextAnalyser.ipynb 
```

## Features

### Dataset Collection from [Scopus](https://www.scopus.com/)

- Articles search query:

    ```SQL
    TITLE ( ( radiology OR radiologist ) ) 
    AND ( TITLE ( ( ( automatic OR automated ) AND report ) OR ( artificial AND intelligence AND report ) OR ( deep AND learning AND report ) OR ( natural AND language AND processing ) OR ( large AND language AND model ) ) OR TITLE-ABS-KEY (report AND ( ( automatic OR automated ) OR ( artificial AND intelligence AND report ) OR ( deep AND learning AND report ) OR ( natural AND language AND processing ) OR ( large AND language AND model ) OR ( information AND retrieval ) OR ( computational AND linguistics ) )) ) 
    AND ( EXCLUDE ( DOCTYPE , "re" ) ) 
    AND ( LIMIT-TO ( LANGUAGE , "English" ) )
    ```

- Reviews search query:

    ```SQL
    TITLE ( ( radiology OR radiologist ) ) 
    AND ( TITLE ( ( ( automatic OR automated ) AND report ) OR ( artificial AND intelligence AND report ) OR ( deep AND learning AND report ) OR ( natural AND language AND processing ) OR ( large AND language AND model ) ) OR TITLE-ABS-KEY (report AND ( ( automatic OR automated ) OR ( artificial AND intelligence AND report ) OR ( deep AND learning AND report ) OR ( natural AND language AND processing ) OR ( large AND language AND model ) OR ( information AND retrieval ) OR ( computational AND linguistics ) )) ) 
    AND ( LIMIT-TO ( DOCTYPE , "re" ) ) 
    AND ( LIMIT-TO ( LANGUAGE , "English" ) )
    ```

### NLP Pipeline

- [X] Data Preprocessing
  - Tokenization
  - Stopword Removal
  - Stemming

- [ ] Topic Modelling and Correlation Analysis
  - [X] Empath
  - [X] TF-IDF
  - [ ] UMLS

### Visualization

- [X] Word Clouds
- [X] Topic historgams
- [X] TF-IDF Charts
- [X] Similarity Matrix Heatmaps
- [ ] GUI

## Future Work

- QuickUMLS Alternatives : [cTAKES](https://ctakes.apache.org/) and [MetaMap](https://metamap.nlm.nih.gov/) are alternatives to QuickUMLS. However they are not as easy to setup and involve a Java ecosystem. Some Docker images are available for cTakes: [ctakes-covid-container](https://github.com/Machine-Learning-for-Medical-Language/ctakes-covid-container), [ctakes-server](https://github.com/choyiny/ctakes-server), [rootstrap/ctakes](https://github.com/rootstrap/ctakes). Another alternative in the Python ecosystem is [MedCAT](https://github.com/CogStack/MedCAT).

- Explore other modelling techniques and tools:
  - [Doc2Vec](https://github.com/piskvorky/gensim)
  - [MedScapy](https://github.com/medspacy/medspacy)
  - [ScispaCy](https://github.com/allenai/scispacy)
  - [Med7](https://github.com/kormilitzin/med7)
  - [RadText](https://github.com/bionlplab/radtext)

## References

- [1] Olivier Bodenreider. “The Unified Medical Language System (UMLS): integrating biomedica. terminology”. In: Nucleic Acids Research 32.suppl1 (Jan. 2004), pp. D267–D270. ISSN: 0305-1048. DOI: 10.1093/nar/gkh061. URL: https://doi.org/10.1093/nar/gkh061.
- [2] H. Eyre et al. “Launching into clinical space with medspaCy: a new clinical text processing toolkit in Python”. In: AMIA Annu Symp Proc 2021 (2021), pp. 438–447.2
- [3] Ethan Fast, Binbin Chen, and Michael S. Bernstein. “Empath: Understanding Topic Signals in Large-Scale Text”. In: Proceedings of the 2016 CHI Conference on Human Factors in Computing Systems. CHI ’16. San Jose, California, USA: Association for Computing Machin-ery, 2016, pp. 4647–4657. ISBN: 9781450333627. DOI: 10.1145/2858036.2858535. URL: https://doi.org/10.1145/2858036.2858535.
- [4] Andrey Kormilitzin et al. “Med7: a transferable clinical natural language processing model for electronic health records”. In: arXiv preprint arXiv:2003.01271 (2020).
- [5] Zeljko Kraljevic et al. “Multi-domain clinical natural language processing with MedCAT: The Medical Concept Annotation Toolkit”. In: Artif. Intell. Med. 117 (July 2021), p. 102083. ISSN: 0933-3657. DOI: 10.1016/j.artmed.2021.102083
- [6] Ali Mozayan et al. “Practical Guide to Natural Language Processing for Radiology”. In: Ra-dioGraphics 41.5 (2021). PMID: 34469212, pp. 1446–1453. DOI: 10.1148/rg.2021200113. URL: https://doi.org/10.1148/rg.2021200113.
- [7] Mark Neumann et al. “ScispaCy: Fast and Robust Models for Biomedical Natural Language Processing”. In: Proceedings of the 18th BioNLP Workshop and Shared Task. Florence, Italy: Association for Computational Linguistics, Aug. 2019, pp. 319–327. DOI: 10.18653/v1/W19-5034. URL: https://www.aclweb.org/anthology/W19-5034.
- [8] Yifan Peng et al. NegBio: a high-performance tool for negation and uncertainty detection in radiology reports. 2017.
- [9] Radim ˇReh ̊uˇrek and Petr Sojka. “Software Framework for Topic Modelling with Large Cor-pora”. English. In: Proceedings of the LREC 2010 Workshop on New Challenges for NLP Frameworks. http://is.muni.cz/publication/884893/en. Valletta, Malta: ELRA, May 2010, pp. 45–50.
- [10] Guergana K Savova et al. “Mayo clinical Text Analysis and Knowledge Extraction System (cTAKES): architecture, component evaluation and applications”. In: Journal of the American Medical Informatics Association 17.5 (Sept. 2010), pp. 507–513. ISSN: 1067-5027. DOI: 10.1136/jamia.2009.001560. URL: https://doi.org/10.1136/jamia.2009.001560.
- [11] Luca Soldaini. “QuickUMLS: a fast, unsupervised approach for medical concept extraction”. In: 2016. URL: https://api.semanticscholar.org/CorpusID:2990304.
- [12] Song Wang et al. Radiology Text Analysis System (RadText): Architecture and Evaluation. 2022.

