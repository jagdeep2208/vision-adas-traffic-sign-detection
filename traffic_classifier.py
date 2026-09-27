import numpy as np
import cv2
import os
from sklearn.model_selection import train_test_split

# 1. Path aur Variables set karein
path = "dataset/Train" # Agar folder ka naam Train hai
classes = 43
images = []
classNo = []

print("Loading dataset... Please wait.")

# 2. Saare folders se images load karein
myList = os.listdir(path)
for x in range(0, classes):
    myPicList = os.listdir(path + "/" + str(x))
    for y in myPicList:
        curImg = cv2.imread(path + "/" + str(x) + "/" + y)
        curImg = cv2.resize(curImg, (32, 32)) # Image size 32x32 fix karna
        curImg = curImg / 255.0
        images.append(curImg)
        classNo.append(x)
    print(f"Loaded Class: {x}", end="\r")

# 3. List ko Numpy Array mein badlein
images = np.array(images)
classNo = np.array(classNo)

print("\nData loading complete!")
print("Total Images:", images.shape[0])
print("Image Shape:", images[1].shape)
print("Labels Shape:", classNo.shape)

# 4. Data Splitting (Training aur Testing ke liye)
X_train, X_test, y_train, y_test = train_test_split(images, classNo, test_size=0.2)
print("Train Data Shape:", X_train.shape)
print("Test Data Shape:", X_test.shape)
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dropout, Flatten, Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical

# 1. Labels ko One-Hot Encoding mein badlein (e.g. 3 -> [0,0,0,1,0...])
y_train = to_categorical(y_train, 43)
y_test = to_categorical(y_test, 43)

# 2. CNN Model Design karna
model = Sequential()
# Pehli Layer: Images se features nikalne ke liye
model.add(Conv2D(filters=32, kernel_size=(5,5), activation='relu', input_shape=(32,32,3)))
model.add(Conv2D(filters=32, kernel_size=(5,5), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(rate=0.25))

# Dusri Layer: Deep patterns samajhne ke liye
model.add(Conv2D(filters=64, kernel_size=(3, 3), activation='relu'))
model.add(Conv2D(filters=64, kernel_size=(3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(rate=0.25))

# Teesri Layer: Data ko flat karke final decision lena
model.add(Flatten())
model.add(Dense(256, activation='relu'))
model.add(Dropout(rate=0.5))
model.add(Dense(43, activation='softmax')) # 43 classes ke liye output

# 3. Model Compile karna
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# 4. Training Shuru karna
print("--- Training Started ---")
epochs = 15 # Aap ise 10-15 rakh sakte hain
history = model.fit(X_train, y_train, batch_size=32, epochs=epochs, validation_data=(X_test, y_test))

# 5. Model Save karna (Bohat zaruri)
if not os.path.exists('models'):
    os.makedirs('models')
model.save("models/traffic_classifier.h5")
print("Model saved successfully as models/traffic_classifier.h5")