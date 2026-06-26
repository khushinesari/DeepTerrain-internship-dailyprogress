# Phase 1
## Chapter 1
- A computer never truly "sees" an image—it processes numerical pixel values.
- An image is a grid (matrix) of pixels.
- A grayscale pixel stores one value (brightness).
- An RGB pixel stores three values: Red, Green, and Blue.
- A video is simply a sequence of images shown over time.
- Computer Vision is the field that teaches computers to interpret images.
- YOLO is one specific Computer Vision model that detects and localizes objects.
## Chapter 2 – Understanding Digital Images
- An image is: Pixels
   - Each pixel stores:One value (grayscale), or Three values (RGB)
- Images have:
  - Width
  - Height
  - Resolution
  - Aspect ratio
  - Channels
  - Bit depth
- Image files like JPEG and PNG are simply different ways of storing those pixel values on disk.
- When YOLO reads an image, it converts it into numerical arrays that can be processed by the neural network.
## Chapter 3 – Image Processing Basics
- Image Processing improves images; Computer Vision extracts information from them.
- Noise is unwanted variation in pixel values, and filters help reduce it.
- Blurring smooths images but can remove detail.
- Gaussian Blur preserves natural appearance better than simple averaging.
- Median filtering is excellent for removing salt-and-pepper noise.
- Sharpening enhances edges.
- Edge detection focuses on object boundaries.
- Thresholding converts grayscale images into binary images.
- Histograms describe the distribution of pixel intensities.
- Morphological operations (erosion, dilation, opening, closing) modify object shapes in binary images.
- Many filters that engineers once designed by hand are now learned automatically by convolutional neural networks.
##  Chapter 4 — Machine Learning Fundamentals
- Artificial Intelligence is the broad field of making machines perform intelligent tasks.
- Machine Learning lets computers learn patterns from data instead of relying on hand-written rules.
- A dataset consists of features (inputs) and labels (correct answers).
- YOLO is trained using supervised learning, where every training image has annotations.
- Training is the process of learning from labeled data.
- Testing/validation measures how well the model performs on unseen data.
- Inference is using a trained model to make predictions.
- Generalization is the ability to perform well on new images.
- Underfitting means the model hasn't learned enough.
- Overfitting means the model has memorized the training data instead of learning general patterns.
##  Chapter 5 – Deep Learning Fundamentals (Artificial Neural Networks)
- Deep Learning automatically learns useful features from data.
- An artificial neuron combines inputs using weights and a bias, then applies an activation function.
- Weights represent the importance of different inputs.
- Bias allows the neuron to shift its decision boundary.
- Activation functions introduce non-linearity, enabling the network to learn complex patterns.
- A layer contains many neurons, each learning different features.
- A deep neural network is formed by stacking many layers.
- Forward propagation makes predictions.
- A loss function measures prediction errors.
- Backpropagation updates the weights based on those errors.
- Gradient descent repeatedly reduces the loss by taking small optimization steps.
## Chapter 6 – Convolutional Neural Networks (CNNs)
- Ordinary fully connected networks are inefficient for images because they ignore spatial relationships and require enormous numbers of parameters.
- CNNs solve this by processing small local regions with filters (kernels).
- A convolution slides a filter across the image to detect patterns.
- Each filter produces a feature map highlighting where a particular pattern appears.
- Filters are learned automatically during training.
- Stride controls how far the filter moves each step.
- Padding preserves border information and controls output size.
- Pooling reduces spatial dimensions while retaining important features.
- As layers get deeper, the network learns increasingly abstract concepts—from edges to full objects.
- This hierarchical feature learning is what enables YOLO to detect complex objects accurately.
## Chapter 7 – From Pixels to Objects: How CNNs Actually Understand Images
- CNNs build a feature hierarchy, starting with simple patterns and ending with complete objects.
- Early layers learn low-level features such as edges and corners.
- Middle layers learn parts of objects like eyes, wheels, and doors.
- Deep layers combine those parts into complete objects.
- Because filters slide across the image, CNNs naturally gain translation invariance.
- Small objects are harder to detect because they contain fewer pixels and can disappear as feature maps become smaller.
- Modern detectors use multi-scale features so they can detect both small and large objects.
- Pretrained models work because they have already learned a rich library of reusable visual features.
- Before YOLO predicts any bounding boxes, it spends most of its computation extracting meaningful features from the image.
# Phase 2 
## Chapter 1 – What is Object Detection?
- Image Classification predicts only what is present.
- Localization predicts what and where for a single object.
- Object Detection predicts what, where, and how many for multiple objects.
- A bounding box is a rectangle used to locate an object.
- Object detection is challenging because of variations in size, lighting, occlusion, viewpoint, and crowded scenes.
- YOLO became revolutionary because it predicts all objects in a single forward pass through the network instead of processing many image regions separately.
## Chapter 2 – Bounding Boxes: The Language of Object Detection
- A bounding box is the smallest rectangle enclosing an object.
- Every object in an image gets its own bounding box.
- Images use a coordinate system with the origin at the top-left.
- Bounding boxes can be represented in several ways, but YOLO uses (center_x, center_y, width, height).
- YOLO stores normalized coordinates so labels remain consistent across different image sizes.
- Every object becomes one line in a .txt label file:
  - class_id center_x center_y width height
- Annotation tools automatically generate these label files from the rectangles you draw.
## Chapter 3 – Confidence Score and Class Prediction
