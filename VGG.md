## VGG Model

## VGG16 Implementation

![](C:/Users/liyuan/Documents/Personal/DeepLearning/img/vgg16-struct.png)

- Sequential Mode: Input->Network->Output

- 画像分類（Classification）問題：Input（224x224 RGB 画像）->Network->Output（種類：Cat・Dog・Brid…）

- Implementation Details 実装の詳細

  - Code: DeepLearning/code/vgg/vgg16.ipynb

  - Convolution

    Kernel Size: 3x3![](C:/Users/liyuan/Documents/Personal/DeepLearning/img/conv.gif)

    

  - Max pooling

    - filter 2x2
    - stride 2x2
    - **Max** value in filter

    ![](C:/Users/liyuan/Documents/Personal/DeepLearning/img/maxpool_animation.gif)

  - Code<->Graph

    ![](C:/Users/liyuan/Documents/Personal/DeepLearning/img/vgg16-code-graph.png)

  - Training:

    - Keras has well-trained vgg mode

    - If custom training data prepared.自分のDatasetがある場合は

    - Dog/Cat Dataset as an example to explain training process

      - Download Dog/Cat Dataset: [Kaggle Cats and Dogs Dataset](https://www.microsoft.com/en-us/download/details.aspx?id=54765)

      - Pick some(most of) images as Training Data;

      - Pick rest of images as Testing Data;

      - Folder struct is as follow:

        ![](C:/Users/liyuan/Documents/Personal/DeepLearning/img/dcfolder.png)

      - Loading data

        ```python
        size = (224,224)
        traind = ImageDataGenerator().flow_from_directory(directory="train", target_size=size)
        testd = ImageDataGenerator().flow_from_directory(directory="test", target_size=size)
        ```

      - Training

        ```python
        hist = model.fit_generator(
            steps_per_epoch=100,
            generator=traind,
            validation_data=testd,
            validation_steps=10,
            epochs=100)
        ```

      - To display Training graph

        ```python
        plt.plot(hist.history["accuracy"])
        plt.plot(hist.history['val_accuracy'])
        plt.plot(hist.history['loss'])
        plt.plot(hist.history['val_loss'])
        plt.title("model accuracy")
        plt.ylabel("Accuracy")
        plt.xlabel("Epoch")
        plt.legend(["Accuracy","Validation Accuracy","loss","Validation Loss"])
        plt.show()
        ```

        ![](C:/Users/liyuan/Documents/Personal/DeepLearning/img/train-hist.png)

      - Save/Load Model

        ```python
        model.save('vgg16_trained.h5')
        
        from keras.models import load_model
        saved_model = load_model("vgg16_trained.h5")
        ```

      - Prediction

        ```python
        from keras.preprocessing import image
        import numpy as np 
        img = image.load_img("test5.jpeg",target_size=(224,224))
        img = np.asarray(img)
        plt.imshow(img)
        img = np.expand_dims(img, axis=0)
        output = model.predict(img)
        if output[0][0] > output[0][1]:
            print("cat")
        else:
            print('dog')
        
        print('{:.2}'.format(output[0][0]))
        print('{:.2}'.format(output[0][1]))
        ```

        ![](C:/Users/liyuan/Documents/Personal/DeepLearning/img/print-predict.png)

        

- Reference:

  - https://www.cc.gatech.edu/~san37/post/dlhc-cnn/
  - https://developers.google.com/machine-learning/practica/image-classification/convolutional-neural-networks?hl=id
  - https://www.pyimagesearch.com/2018/12/31/keras-conv2d-and-convolutional-layers/