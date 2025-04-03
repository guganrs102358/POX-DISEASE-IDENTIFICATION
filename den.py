import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import DenseNet121
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from sklearn.metrics import classification_report, confusion_matrix, cohen_kappa_score, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns

# Set dataset paths
train_dir = 'D:\DL PROJECT\python\dataset\train'
valid_dir = 'D:\DL PROJECT\python\dataset\valid'
test_dir = 'D:\DL PROJECT\Python\dataset\test'

# Image dimensions and batch size
img_size = (224, 224)
batch_size = 32

# Data Augmentation and Image Generators
train_datagen = ImageDataGenerator(rescale=1./255)
valid_datagen = ImageDataGenerator(rescale=1./255)
test_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
    r'D:\DL PROJECT\Python\dataset\train', target_size=img_size, batch_size=batch_size, class_mode='categorical'
)
valid_generator = valid_datagen.flow_from_directory(
    r'D:\DL PROJECT\Python\dataset\valid', target_size=img_size, batch_size=batch_size, class_mode='categorical'
)
test_generator = test_datagen.flow_from_directory(
    r'D:\DL PROJECT\Python\dataset\test', target_size=img_size, batch_size=batch_size, class_mode='categorical', shuffle=False
)

# Load DenseNet121 Model
base_model = DenseNet121(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
x = GlobalAveragePooling2D()(base_model.output)
x = Dense(128, activation='relu')(x)
x = Dense(64, activation='relu')(x)
output_layer = Dense(4, activation='softmax')(x)  # 4 classes

model = Model(inputs=base_model.input, outputs=output_layer)

# Freeze the base model layers
for layer in base_model.layers:
    layer.trainable = False

# Compile Model
model.compile(optimizer=Adam(learning_rate=0.0001), loss='categorical_crossentropy', metrics=['accuracy'])

# Train Model
history = model.fit(train_generator, validation_data=valid_generator, epochs=20)

# Evaluate Model
test_generator.reset()
y_true = test_generator.classes
y_pred_probs = model.predict(test_generator)
y_pred = np.argmax(y_pred_probs, axis=1)

# Classification Report
report = classification_report(y_true, y_pred, target_names=test_generator.class_indices.keys(), output_dict=True)
accuracy = accuracy_score(y_true, y_pred)
kappa = cohen_kappa_score(y_true, y_pred)
precision = report['weighted avg']['precision']
recall = report['weighted avg']['recall']
f1_score = report['weighted avg']['f1-score']
specificity = np.mean([report[label]['recall'] for label in test_generator.class_indices.keys()])  # Approximate
error_rate = 1 - accuracy

# Print Metrics
print(f"Accuracy: {accuracy:.4f}")
print(f"Cohen's Kappa: {kappa:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1-Score: {f1_score:.4f}")
print(f"Specificity: {specificity:.4f}")
print(f"Error Rate: {error_rate:.4f}")

# Confusion Matrix
cm = confusion_matrix(y_true, y_pred)

# Plot Training Accuracy & Loss
plt.figure(figsize=(12, 4))

plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.legend()
plt.title('Accuracy')

plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.legend()
plt.title('Loss')
plt.show()

# Plot Confusion Matrix
plt.figure(figsize=(6, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=test_generator.class_indices.keys(), yticklabels=test_generator.class_indices.keys())
plt.title('Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('True')
plt.show()
