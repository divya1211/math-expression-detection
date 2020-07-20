# Math Expression Detection

Detect mathematical expressions in worksheets and draw bounding boxes.


## Examples

[TODO]


## How is it done?

- Scraped data from Bing for the keyword "math worksheets" using [google-images-download](https://github.com/hardikvasa/google-images-download).
- Annotated ~50 worksheets, assigning 0 to non-math expressions and 1 to math expressions.
- Using the [CRAFT: Character-Region Awareness For Text detection](https://github.com/clovaai/CRAFT-pytorch) detect general purpose OCR. 
- A trained binary classifier using BERT removes non-mathematical expressions using the annotated data.
- Non-maximal supression to combine multiple intersecting bounding-boxes together. 
- Plot the bounding boxes over the images.

## Code

- `boundingbox.py`: Takes in image folder. Computes bounding box. Plots them. 
- `train_classifier.py`: Takes in the annotated data exmaples. Trains a binary classifier on top of BERT. 
- `classifier.py`: Loads up trained BERT classifier. Runs inference. 
- `data.py`: Custom PyTorch Dataset class for Math Expressions. 
- `non_maximal_supression.py`: Performs non maximal supression. [Credit](https://github.com/bruceyang2012/nms_python)


## What didn't work?

- I tried using [ScanSSD](https://arxiv.org/abs/2003.08005) pre-trained on datasetname. However, the results were not accurate. I believe this is because ScanSSD is trained on Math latex expressions, whereas we wanted it to perform on Math worksheets. Thereby the decision to create annotated examples. 
- Used perplexity from GPT-2 to remove non-math expression. I assumed that math expression perplexity would be higher than non-math expressions. However, no significant difference observed between them. 


