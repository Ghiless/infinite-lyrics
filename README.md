# Lyrics generation using PyTorch

```
$ git clone https://github.com/Ghiless/infinite-lyrics.git
$ cd infinite-lyrics
$ python generate.py 100
```

### Usage
```
Usage: generate.py [-h] [-c] [-w] [-l] [-p PRIME_STRING] n

positional arguments:
  n                     number of words (or characters or lines) to generate

optional arguments:
  -h, --help            show this help message and exit
  -c, --characters
  -w, --words (by default)
  -l, --lines
  -p PRIME_STRING, --prime_string PRIME_STRING
                        string to prime generation with ('I love' by default)
```
                        
### Examples
Generate a song of 100 words
```
$ python generate.py 100
```
Or
```
$ python generate.py -w 100
```

Generate a song of 250 characters
```
$ python generate.py -c 250
```

Generate a song of 15 lines (verses)
```
$ python generate.py -l 15
```

Generate a song of 15 lines starting whith 'Hello'
```
$ python generate.py -l 15 -p Hello
```
