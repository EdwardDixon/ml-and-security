# ml-and-security
Code accompanying our article (co-authors [Alex Ott](https://github.com/alexott)) and [Damilare Fagbemi](https://edgeofus.com/)
on the implications of student/teacher model training from a security perspective.


## Teaching a small model from a big one
Student-teacher training or model distillation is a [powerful machine learning technique](https://www.quora.com/What-is-a-teacher-student-model-in-a-Convolutional-neural-network/answer/Edward-Dixon) with several interesting 
use-cases (model compression, model parallelisation) - but our article focuses on the security use-case (someone might
copy your machine learning-based web service with surprising ease).

* We created our own nose detector by first feeding pictures of faces to AWS Rekognition & recording the nose locations
that we got back
* The face images and keypoint coordinates then become the train & test set for a small (student) model of our own

### How do I get set up? ###

* You may need to install some python libraries we use - `pip install -r requirements.txt`

### How do I run the experiments?

See [training_exps.py](training_exps.py) to train CNNs using the student-teacher paradigm, while varying the precision of 
the target values to see how it affects how well they learn (hint: usually, not much!).  You can train a student from scratch,
or, better, use transfer learning so your student doesn't have to start from nothing.   Experimental results will be written 
to an "experiments" subfolder - which we'll create if it doesn't exist.

To test raw performance, use [speed_test_predictions.py](speed_test_predictions.py) - this is how we figure out how 
much money we can save by creating a functional equivalent of someone's cloud-hosted model.  Use [plot_nose](plot_nose.py)
to plot model results against an input image (we used this plotting code for the animation in the article)
