# Enhancing Document Legibility through Intelligent Content Interpretation and Reconstruction
### Group Members:
- Nuoyan Wang (nuoyanw2)
- Alyssa Huang (azh4)
- William Shen (wshen15)
- Ananya Kommalapati (akomma3)
- Allen Peng (allenp2)

## Usage
### Installing Requirements
```
cd path/to/ece549-final-project
pip install -r requirements.txt
```

### To run
```
cd path/to/ece549-final-project
python3 main.py
```

## Project description and goals
In today’s evolving educational landscape, technological advancements are consistently being used to enhance classroom procedures. One common usage of this is the common practice of scanning assignments and exam papers, so teachers and graders can more conveniently access and keep records of them. However, numerous scans can have problems due to various factors like lighting issues, handwriting legibility, etc. Because of this, we propose a computer vision system that can interpret document legibility from scanned documents and fill in any illegible spots with handwriting that mimics the input. We will explore various techniques, both classical and learned and compare the performance. For classical methods, we will look at edge/corner detection as well as image processing to try to correct the document and make it more intelligible. To compare classical feature detection algorithms to hybrid or predominantly deep learning algorithms, we also plan to implement deep learning based feature extraction/detection. While there are existing deep learning architectures we can immediately use, we hope to implement at least a semi-original deep learning classifier.

In order to implement this system, we plan to perform the following transformations/ detections on the input image in order:
- Perspective Change: Perform an affine transformation to ensure the correct perspective on the paper
- Grayscale: Grayscale the image and increase the contrast to make writing marks darker
- Noise Reduction: Filter the image using a Gaussian filter or otherwise to reduce artifacts that may cause false positives
- Morphological Image Operations: Use dilation or erosion to reconnect parts of letters or separate touching letters
- Projection Profile Analysis: Detect rows of text
- Connected Component Analysis: Find “blobs” of connected pixels to identify separate written characters
- Character Detection: Deep-learning based model to recognize characters and extract text from the image
- NMS: Use non-maximum suppression to ensure recognized characters don’t overlap
- Text correction: Using NLP models like language models or spell checkers to confirm detected text/generate missing letters that weren’t able to be detected 
- Text generation: Using a trained RNNs (like the one proposed by Alex Graves for handwriting synthesis), fill in missing letters


Our desired minimum final outcome is to:
- Interpret the document’s content
- Correctly identify/detect the alphanumeric characters present
- Compute a “legibility score” in comparison to the document's actual content.
- Output the same document completed to be more legible

If progress is smooth, additional maximum goals we intend to complete:
- An interesting add-on is to output corrected text in a style “similar” to the original handwriting. 

## Pipeline

Dataset Acquisition: Acquiring sample handwritten alphanumerics from real students in UIUC, with a wide range of neatness or rushedness. 

Dataset Processing: Making scanned images more illegible through artificial brightness changes, adding erase marks, creating folds, scanning with phone camera with distortion, etc.

Classical Methods Text Recognition: As described above in our project statement, this constitutes the main portion of our project. We will attempt numerous methods to accurately detect poorly written or scanned text.

CNN Text Recognition: Utilizing deep learning methods with CNN in PyTorch, results will be compared with our classical approaches. 

Output Text Generation: Our minimum goal is to output a corrected/clean version of the input text. If time permits, we may try to output in a style similar to the input text.

Member Interaction:
Specific features and tasks will have their own branch to ensure a working baseline project. Regarding group member interactions, communication will be mainly conducted within our shared Discord chat, with weekly meetings conducted either in person, or at least live online.

Components:
- Interpreting the document’s content
- Training a model to recognize handwritten alphanumeric characters - Should give us a text of what was on the document
- Possible noise - lighting issues, erase marks, water, poor handwriting
- Outputting the document, but more legible
- Could either be document as text or document matching the input handwriting
- Add ons
- Fix spelling
- Complete sentences

## Resources:

We plan to implement traditional computer vision/machine learning algorithms using Python and a deep learning based classifier using PyTorch for rapid prototyping. We will potentially use pre-existing deep learning text classifiers as an inspiration for our deep learning approach, or augment those existing text classifiers. We have access to adequate computational resources to complete our project. The most resource intensive task should be training a deep learning model on data of handwritten text, but the training time should take a few hours at most. Below are pointers to relevant resources.

Kaggle dataset and dataset from students (we will get students to complete a worksheet and perform lighting changes, pour water on the data, etc to result in “poor scans”)
- https://arxiv.org/pdf/2303.15269 - Text generation that is similar to input handwriting
- https://arxiv.org/pdf/2103.06450 - Interpreting and recognizing handwriting with learned methods
- https://www.ncbi.nlm.nih.gov/pmc/articles/PMC10950881/ - Hybrid approach of AI and classical computer vision techniques for handwritten text recognition
- https://ijeit.com/vol%201/Issue%205/IJEIT1412201205_36.pdf - Mentions a few classical methods such as border transition (similar to edge detection) that could be used to match characters
- https://arxiv.org/pdf/1910.13796 - Explains some of the differences between traditional computer vision techniques and more modern approaches involving hybrid classical/deep learning algorithms
- https://cs231n.stanford.edu/reports/2017/pdfs/810.pdf - Two deep learning approaches to classifying and recognizing handwritten text(CNN and LSTM)
- https://github.com/githubharald/SimpleHTR?tab=readme-ov-file - Basic deep learning handwritten text classifier implemented using TensorFlow
- https://www.sciencedirect.com/science/article/pii/S0167865506002509 - Projection Profile Analysis with skewed text
- Generating Sequences With Recurrent Neural Networks - Graves Handwriting Synthesis
- https://arxiv.org/pdf/1503.03957 - Paper on efficient implementation of fuzzy logic based information retrieval system, potentially useful for query matching

We also plan to collect handwritten data from our peers for testing and training purposes.


## Notes
To update requirements:
```
pip install pipreqs
pipreqs /path/to/project
```