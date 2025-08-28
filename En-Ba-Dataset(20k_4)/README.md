---
license: mit
task_categories:
- text-classification
language:
- bn
pretty_name: BnSentMix
size_categories:
- 10K<n<100K
---

# BnSentMix: A Diverse Bengali-English Code-Mixed Dataset for Sentiment Analysis

[![codeRepo](https://img.shields.io/badge/Code-Nishita2000/BnSentMix-blue?logo=GitHub)](https://github.com/Nishita2000/BnSentMix)

## Dataset Overview

Column Title | Description
------------ | -------------
`Data Sources` | Facebook, YouTube, E-commerce Sites
`#Samples` | 20000
`Sentiment Labels` | 1:Positive, 2:Negative, 3:Neutral, 4:Mixed
`Filtering Method` | Automated using `mBERT`
`#Annotators` | 64
`Annotation/Sample` | 2 or 3 (if tie)

## Dataset Statistics

| Statistic               | Value  |
|-------------------------|--------|
| Mean Character Length    | 62.77  |
| Max Character Length     | 1985   |
| Min Character Length     | 14     |
| Mean Word Count          | 11.65  |
| Max Word Count           | 368    |
| Min Word Count           | 4      |
| Unique Word Count        | 37734  |
| Unique Sentence Count    | 21873  |

## Citation

If you find this work useful, please cite our paper:

```bib
@inproceedings{alam-etal-2025-bnsentmix,
    title = "{B}n{S}ent{M}ix: A Diverse {B}engali-{E}nglish Code-Mixed Dataset for Sentiment Analysis",
    author = "Alam, Sadia  and Ishmam, Md Farhan  and Alvee, Navid Hasin  and Siddique, Md Shahnewaz  and
      Hossain, Md Azam  and Kamal, Abu Raihan Mostofa", editor = "Hettiarachchi, Hansi  and
      Ranasinghe, Tharindu  and Rayson, Paul  and Mitkov, Ruslan  and Gaber, Mohamed  and
      Premasiri, Damith  and Tan, Fiona Anting  and Uyangodage, Lasitha",
    booktitle = "Proceedings of the First Workshop on Language Models for Low-Resource Languages",
    month = jan,
    year = "2025",
    address = "Abu Dhabi, United Arab Emirates",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2025.loreslm-1.4/",
    pages = "68--77"
}
```