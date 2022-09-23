# Conversion Scripts for Coreference


## The `jsonlines2conll.py` script

Script to convert a jsonlines file to a CoNLL file.  Use the `-h` and `--help` switches to get detailed help on the options.

Example command (output uses spaces):

```bash
python3 jsonlines2conll.py -g testing/singe.jsonlines -o ouput.conll
```

```
#begin document (ge/articleswiki_singe.xml); part 000
Singe   (0)

         Les         (0
      singes         0)
        sont          -
         des         (0
  mammifères          -
          de          -
          l'         (1
       ordre          -
         des          -
          de          -
         les         (2
    primates      1)|2)
...
#end document
```

Example command (merging coreference information with an existing conll file, for example to add predicted coreference):

```bash
python3 jsonlines2conll.py -g testing/singe.jsonlines -o ouput.conll -c testing/singe.conll
```

```
#begin document (ge/articleswiki_singe.xml); part 000
1   Singe   Singe   NOUN   ...

   1            Les             le     DET   ...
   2         singes          singe    NOUN   ...
   3           sont           être     AUX   ...
   4            des             un     DET   ...
   5     mammifères      mammifère    NOUN   ...
   6             de             de     ADP   ...
   7             l'             le     DET   ...
   8          ordre          ordre    NOUN   ...
9-10            des              _       _   ...
   9             de             de     ADP   ...
  10            les             le     DET   ...
  11       primates        primate    NOUN   ...
...
#end document
```

Example command (merging + output uses tabulation):

```bash
python3 jsonlines2conll.py -g testing/singe.jsonlines -o ouput.conll -c testing/singe.conll -T
```


## Main formats used in automatic coreference resolution

The **CoNLL format** is a tabular format: each token is on a separate line and annotation for the token are on separate column.  Document boundaries are indicated by specific marks, and sentence separation by a white line.

Here is an example:

```
#begin document <name of the document>
1            Les             le     DET
2         singes          singe    NOUN
3           sont           être     AUX
4            des             un     DET
5     mammifères      mammifère    NOUN
...

1           Bien           bien     ADV
2            que            que   SCONJ
3           leur            son     DET
4   ressemblance   ressemblance    NOUN
5           avec           avec     ADP
6             l'             le     DET
7          Homme          homme    NOUN
...
#end document
```

Column separator (spaces or tabulation), number and content vary according to specification (CoNLL-2012, CoNLL-U, CoNLL-X, etc.).


The **jsonlines format** stores data for several texts (a corpus).  Each line is a valid json document, as follows:

```
{
  "clusters": [],
  "doc_key": "nw:docname",
  "sentences": [["This", "is", "the", "first", "sentence", "."],
                ["This", "is", "the", "second", "."]],
  "speakers":  [["spk1", "spk1", "spk1", "spk1", "spk1", "spk1"],
                ["spk2", "spk2", "spk2", "spk2", "spk2"]]
  "pos":       [["DET", "V", "DET", "ADJ", "NOUN", "PUNCT"],
                ["DET", "V", "DET", "ADJ", "PUNCT"]],
  ...
}
```

It is used for some coreference resolution systems, such as:

- https://github.com/kentonl/e2e-coref (English)
- https://github.com/kkjawz/coref-ee (English)
- https://github.com/boberle/cofr (French)


## License

All the scripts are distributed under the terms of the Mozilla Public Licence 2.0.

