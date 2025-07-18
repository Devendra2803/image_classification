import os, zipfile, io, tempfile, logging, gzip
import numpy as np
from PIL import Image
import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.metrics import precision_score, recall_score
from app.database import SessionLocal
from app.models import TrainingLog

logger = logging.getLogger(__name__)

class MobileNetTrainer:
    def __init__(self, project_name, num_classes, class_names):
        self.project = project_name
        self.num_classes = num_classes
        self.class_names = class_names
        self.model = None
        self.model_blob = None
        self.img_size = (224, 224)

    def _extract_zip(self, zip_bytes):
        temp_dir = tempfile.mkdtemp()
        zip_path = os.path.join(temp_dir, "data.zip")
        with open(zip_path, "wb") as f:
            f.write(zip_bytes)
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(temp_dir)
        return temp_dir

    def _check_image_count(self, folder):
        return sum([len(files) for _, _, files in os.walk(folder)])

    def _split_data(self, data_dir):
        datagen = ImageDataGenerator(rescale=1.0 / 255, validation_split=0.2)
        train_gen = datagen.flow_from_directory(
            data_dir, target_size=self.img_size, batch_size=32,
            class_mode="categorical", subset="training"
        )
        val_gen = datagen.flow_from_directory(
            data_dir, target_size=self.img_size, batch_size=32,
            class_mode="categorical", subset="validation"
        )
        return train_gen, val_gen

    def _build_model(self):
        input_shape = (*self.img_size, 3)
        base_model = MobileNetV2(weights="imagenet", include_top=False, input_shape=input_shape)
        x = GlobalAveragePooling2D()(base_model.output)
        preds = Dense(self.num_classes, activation="softmax")(x)
        model = Model(inputs=base_model.input, outputs=preds)
        model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
        return model

    def process_and_train(self, zip_bytes):
        data_dir = self._extract_zip(zip_bytes)
        total_images = self._check_image_count(data_dir)
        logger.info(f"[{self.project}] Total image count: {total_images}")

        if total_images < 60:
            return {"error": "Too few images (minimum 300 required)", "count": total_images}
        if total_images > 3000:
            return {"error": "Too many images (maximum 3000 allowed)", "count": total_images}

        train_gen, val_gen = self._split_data(data_dir)
        self.model = self._build_model()
        callback = EarlyStopping(patience=3, restore_best_weights=True)

        history = self.model.fit(train_gen, validation_data=val_gen, epochs=20, callbacks=[callback])

        val_gen.reset()
        y_true = val_gen.classes
        y_pred = self.model.predict(val_gen)
        y_pred_classes = np.argmax(y_pred, axis=1)

        precision = precision_score(y_true, y_pred_classes, average="macro", zero_division=0)
        recall = recall_score(y_true, y_pred_classes, average="macro", zero_division=0)
        train_acc = history.history["accuracy"][-1]
        val_acc = history.history["val_accuracy"][-1]

        with tempfile.NamedTemporaryFile(suffix=".h5", delete=False) as tmp:
            self.model.save(tmp.name)
            tmp.flush()
            with open(tmp.name, "rb") as f:
                raw_model_bytes = f.read()
            self.model_blob = gzip.compress(raw_model_bytes)
        os.unlink(tmp.name)

        logger.info(f"[{self.project}] Model trained and saved to memory (compressed).")

        return {
            "message": "Model trained successfully",
            "train_accuracy": train_acc,
            "val_accuracy": val_acc,
            "precision": precision,
            "recall": recall
        }

    def predict_from_bytes(self, image_bytes):
        img = Image.open(io.BytesIO(image_bytes)).convert("RGB").resize(self.img_size)
        img_array = np.expand_dims(np.array(img) / 255.0, axis=0)
        predictions = self.model.predict(img_array)[0]
        top_class = np.argmax(predictions)
        return {
            "predicted_class": self.class_names[top_class],
            "confidence_score": float(predictions[top_class])
        }
