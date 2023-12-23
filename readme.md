# Food CNN
This is a respository containing AI models to classify images in the Food-101 dataset. I focused primarily on the ResNet architectures for training. My approach involved experimenting with ResNet18, ResNet34, ResNet50, and ResNet101 while exploring various image sizes. This endeavor aimed to optimize the model's performance in handling the dataset's complexities, such as limited training data per class, visual similarities between certain classes (e.g., steak vs. filet mignon), images with poor lighting or framing, and instances of multiple correct classes within single images.

The best model scored 80% accuracy on the validation set.

### Takeaways
As expected the AI preformed better when trained on larger higher quality images. However the training time increased exponentially.

Resnet preformed poorly when using the 101 layer version due to the small training set of only 75750 images.

ResNet18 preformed quite well reaching 70%, but it lacked complexity to capture all nuances.

ResNet34 and ResNet50 preformed the best.