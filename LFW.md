# Labeled Faces in the Wild (LFW)

##### University of Massachusetts - Amherst

The entire Labeled Faces in the Wild database can be downloaded as a gzipped tar file.  After uncompressing, the contents of the database will be placed in a new directory "lfw".

Each image is available as "lfw/name/name_xxxx.jpg", where "xxxx" is the image number padded to four characters with leading zeroes.  For example, the 10th George_W_Bush image can be found as "lfs/George_W_Bush/George_W_Bush_0010.jpg".

There are a total of 13233 images and 5749 people in the database. Each image is a 250x250 jpg, detected and centered using the openCV implementation of Viola-Jones face detector.  The cropping region returned by the detector was then automatically enlarged by a factor of 2.2 in each dimension to capture more of the head and then scaled to a uniform size.

### Training paradigms

We give two possibilities for forming the training sets.

#### Image Restricted Configuration

In the first formulation, the training information is restricted to the image pairs given in the pairs.txt file.  No information about the actual names of the people in the image pairs should be used.  This is meant to address the issue of transitivity.

In other words, if one matched pair consists of the 10th and 12th images of George_W_Bush, and another pair consists of the 42nd and 50th images of George_W_Bush, then under this formulation it would not be allowable to use the fact that both pairs consist of images of George_W_Bush in order to form new pairs such as the 10th and 42nd images of George_W_Bush.

To ensure this holds, one should only use the name information to identify the image, but not provide the name information to the actual algorithm.  For this reason, we refer to this formulation as the Image Restricted Configuration.  Under this formulation, only the pairs.txt file is needed.

#### Unrestricted Configuration

In the second formulation, the training information is provided as simply the names of the people in each set and the associated images. From this information, one can, for example, formulate as many match and mismatch pairs as one desires, from people within each set.

For instance, if George_W_Bush and John_Kerry both appear in one set, then any pair of George_W_Bush images can be used as a match pair, and any image of George_W_Bush can be matched with any image of John_Kerry to form a mismatch pair.

We refer to this formulation as the Unrestricted Configuration, and provide the people.txt that gives the names of people in each set.

### Test procedure

Under both configurations, the test procedure is the same.  That is, the training sets are formed from 9 of the 10 sets, with the held-out set as the test set.  The algorithm must then classify each pair from the held-out set, given in pairs.txt, based on the image information from that pair alone.  In other words, the algorithm's classification must be a function of the single pair of images, and not attempt to leverage the other test pairs.

Note that, the pairs.txt is needed (for the purposes of computing the test performance), even under the Unrestricted Configuration.  Also, under the Unrestricted Configuration, one can form mismatch pairs from images across different sets in the training data.


### Training, validation, and testing

We organize our data into two "Views".  View 1 is for algorithm development and general experimentation, prior to formal evaluation, i.e. model selection or validation.  View 2 is for performance reporting, and should be used only for the final evaluation of a method, to minimize "fitting to the test data".

#### View 1: development training/testing sets

We give the development sets in both configurations (image restricted and unrestricted).  The first configuration consists of pairsDevTraining.txt and pairsDevTest.txt.  The format for these two files is: the first line gives the number of matched pairs N (equal to the number of mismatched pairs) in the set, followed by N lines of matched pairs and N lines of mismatched pairs in the same format as the files for the performance reporting sets.

The second configuration consists of peopleDevTraining.txt and peopleDevTest.txt.  The format for these two files is: the first line gives the number of people N in the set, followed by N lines of names and number of images per person in the same format as the files for the performance reporting sets.

#### View 2: performance testing configurations

We randomly split the database into 10 sets (uniformly random at the person level).  We then randomly chosen 300 matched pairs and 300 mismatched pairs within each set.  This information is provided in pairs.txt.  Using this split, performance on the database can be given using 10-fold cross validation.

#### pairs.txt format

The pairs.txt file is formatted as follows: The top line gives the number of sets followed by the number of matched pairs per set (equal to the number of mismatched pairs per set).  The next 300 lines give the matched pairs in the following format:

name   n1   n2

which means the matched pair consists of the n1 and n2 images for the person with the given name.  For instance,

George_W_Bush   10   24

would mean that the pair consists of images George_W_Bush_0010.jpg and George_W_Bush_0024.jpg.

The following 300 lines give the mismatched pairs in the following format:

name1   n1   name2   n2

which means the mismatched  pair consists of  the n1 image  of person name1 and the n2 image of person name2.  For instance,

George_W_Bush   12   John_Kerry   8

would mean that the pair consists of images George_W_Bush_0012.jpg and John_Kery_0008.jpg.

This procedure is then repeated 9 more times to give the pairs for the next 9 sets.

#### people.txt format

The people.txt file is formatted as follows: The top line gives the number of sets.  The following line gives the number of people in the first set.  Let that number be N.  The next N lines give the names and number of images of the people in the first set, one per line. For instance, if George_W_Bush was in the first line, one line would be:

George_W_Bush   530

The next subsequent line gives the number of people in the second set, followed by the names and number of images of the people in the second set.  This procedure is repeated for all 10 sets.

### Additional details

For additional details on how the database was constructed, as well as how the configurations were chosen for performance reporting, please refer to our technical report:

Gary B. Huang, Manu Ramesh, Tamara Berg, and Erik Learned-Miller. Labeled Faces in the Wild: A Database for Studying Face Recognition in Unconstrained Environments. University of Massachusetts, Amherst, Technical Report 07-49, October, 2007.

For updated details on categories of LFW results, including information concerning unsupervised methods and methods using external training data, please refer to our follow-up technical report:

Gary B. Huang and Erik Learned-Miller. Labeled Faces in the Wild: Updates and New Reporting Procedures. UMass Amherst Technical Report UM-CS-2014-003, 2014.
