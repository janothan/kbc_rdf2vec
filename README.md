# KBC RDF2Vec
A simple Python project to generate a knowledge base completion file for evaluation given a gensim model.
The file can then be evaluated using [KBC Evaluation](https://github.com/janothan/kbc_evaluation/).

## Evaluation File Format

```
<valid triple>
    Heads: <concepts space separated>
    Tails: <concepts space separated>
```


### Development Remarks
- Docstring format: <a href="https://numpy.org/doc/stable/docs/howto_document.html">NumPy/SciPy</a>