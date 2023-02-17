import keras
import numpy as np
import tensorflow as tf
from keras import layers
from keras import models
from os import listdir
import matplotlib.pyplot as plt
import seaborn as sns


class RayTensor:
    def __init__(self):
        self.epochs = 100
        self.batch_size = 64
        self.xray_img_height = 200
        self.xray_img_width = 200
        self.ct_img_height = 100
        self.ct_img_width = 100
        self.xray_train_path = "train_ds/train"
        self.xray_test_path = "train_ds/test"
        self.ct_path = "ct_ds/train"
        self.ct_metrics_path = "ct_ds/test"
        self.validation_split = 0.2
        self.xray_class_names = ["COVID", "NORMAL", "PNEUMONIA", "TURBERCULOSIS"]
        self.ct_class_names = ["1NonCOVID", "2COVID", "3CAP"]

    def xray_model_create(self):
        train_ds = tf.keras.utils.image_dataset_from_directory(
            self.xray_train_path,
            batch_size=self.batch_size,
            seed=123,
            image_size=(self.xray_img_height, self.xray_img_width),
        )
        val_ds = tf.keras.utils.image_dataset_from_directory(
            self.xray_test_path,
            batch_size=self.batch_size,
            seed=123,
            image_size=(self.xray_img_height, self.xray_img_width),
        )

        class_names = train_ds.class_names

        model = models.Sequential(
            [
                layers.RandomFlip(
                    "horizontal",
                    input_shape=(self.xray_img_height, self.xray_img_width, 3),
                ),
                layers.RandomRotation(0.1),
                layers.RandomZoom(0.1),
                layers.Rescaling(1.0 / 255),
                layers.Conv2D(64, (3, 3), padding="same", activation="relu"),
                layers.MaxPooling2D((2, 2)),
                layers.Conv2D(128, (3, 3), padding="same", activation="relu"),
                layers.MaxPooling2D((2, 2)),
                layers.Conv2D(128, (3, 3), padding="same", activation="relu"),
                layers.MaxPooling2D((2, 2)),
                layers.Conv2D(128, (3, 3), padding="same", activation="relu"),
                layers.MaxPooling2D((2, 2)),
                layers.Dropout(0.25),
                layers.Flatten(),
                layers.Dense(512, activation="relu"),
                layers.Dense(len(class_names), activation="softmax"),
            ]
        )

        model.compile(
            optimizer="adam",
            loss="sparse_categorical_crossentropy",
            metrics=["accuracy"],
        )

        model.fit(train_ds, validation_data=val_ds, epochs=self.epochs)

        model.save("xray_model.h5")

    def xray_predict(self, path_to_image):
        model = keras.models.load_model("models/xray_model.h5")
        img = tf.keras.utils.load_img(
            path_to_image, target_size=(self.xray_img_height, self.xray_img_width)
        )

        img_array = tf.keras.utils.img_to_array(img)
        img_array = tf.expand_dims(img_array, 0)
        predictions = model.predict(img_array)
        score = tf.nn.softmax(predictions[0])

        if self.xray_class_names[np.argmax(score)] == "PNEUMONIA":
            self.xray_class_names[np.argmax(score)] = "пневмония"
        elif self.xray_class_names[np.argmax(score)] == "NORMAL":
            self.xray_class_names[np.argmax(score)] = "всё хорошо"
        elif self.xray_class_names[np.argmax(score)] == "COVID19":
            self.xray_class_names[np.argmax(score)] = "Covid-19"
        elif self.xray_class_names[np.argmax(score)] == "TURBERCULOSIS":
            self.xray_class_names[np.argmax(score)] = "туберкулёз"

        return [
            self.xray_class_names,
            np.array(score * 100),
            100 * np.max(score),
            self.xray_class_names[np.argmax(score)],
        ]

    @staticmethod
    def xray_metrics():

        covid_cm = []
        pneumonia_cm = []
        turberculosis_cm = []
        normal_cm = []

        accuracy_fc = []

        for i in listdir(f"train_ds/val/"):
            if i == "COVID":
                covid, pneumonia, turberculosis, normal = 0, 0, 0, 0
                print(i, "", len(listdir(f"train_ds/val/{i}")))
                for j in listdir(f"train_ds/val/{i}"):
                    if app.xray_predict(f"train_ds/val/{i}/{j}")[3] == "COVID":
                        covid += 1
                    elif app.xray_predict(f"train_ds/val/{i}/{j}")[3] == "пневмония":
                        pneumonia += 1
                    elif app.xray_predict(f"train_ds/val/{i}/{j}")[3] == "туберкулёз":
                        turberculosis += 1
                    elif app.xray_predict(f"train_ds/val/{i}/{j}")[3] == "всё хорошо":
                        normal += 1
                    print(
                        f"COVID: {covid}, PNEUMONIA: {pneumonia}, TURBERCULOSIS: {turberculosis}, NORMAL: {normal}"
                    )
                print(
                    "COVID - итоговая точность: ",
                    round((covid / len(listdir("train_ds/val/COVID"))), 3) * 100,
                    "%",
                )
                accuracy_fc.append(
                    {
                        "Covid": round((covid / len(listdir("train_ds/val/COVID"))), 3)
                        * 100
                    }
                )
                covid_cm = [covid, pneumonia, turberculosis, normal]
            elif i == "PNEUMONIA":
                covid, pneumonia, turberculosis, normal = 0, 0, 0, 0
                print(i, "", len(listdir(f"train_ds/val/{i}")))
                for j in listdir(f"train_ds/val/{i}"):
                    if app.xray_predict(f"train_ds/val/{i}/{j}")[3] == "COVID":
                        covid += 1
                    elif app.xray_predict(f"train_ds/val/{i}/{j}")[3] == "пневмония":
                        pneumonia += 1
                    elif app.xray_predict(f"train_ds/val/{i}/{j}")[3] == "туберкулёз":
                        turberculosis += 1
                    elif app.xray_predict(f"train_ds/val/{i}/{j}")[3] == "всё хорошо":
                        normal += 1
                    print(
                        f"COVID: {covid}, PNEUMONIA: {pneumonia}, TURBERCULOSIS: {turberculosis}, NORMAL: {normal}"
                    )
                print(
                    "PNEUMONIA - итоговая точность: ",
                    round((pneumonia / len(listdir("train_ds/val/PNEUMONIA"))), 3)
                    * 100,
                    "%",
                )
                accuracy_fc.append(
                    {
                        "Pneumonia": round(
                            (pneumonia / len(listdir("train_ds/val/PNEUMONIA"))), 3
                        )
                        * 100
                    }
                )
                pneumonia_cm = [covid, pneumonia, turberculosis, normal]
            elif i == "TURBERCULOSIS":
                covid, pneumonia, turberculosis, normal = 0, 0, 0, 0
                print(i, "", len(listdir(f"train_ds/val/{i}")))
                for j in listdir(f"train_ds/val/{i}"):
                    if app.xray_predict(f"train_ds/val/{i}/{j}")[3] == "COVID":
                        covid += 1
                    elif app.xray_predict(f"train_ds/val/{i}/{j}")[3] == "пневмония":
                        pneumonia += 1
                    elif app.xray_predict(f"train_ds/val/{i}/{j}")[3] == "туберкулёз":
                        turberculosis += 1
                    elif app.xray_predict(f"train_ds/val/{i}/{j}")[3] == "всё хорошо":
                        normal += 1
                    print(
                        f"COVID: {covid}, PNEUMONIA: {pneumonia}, TURBERCULOSIS: {turberculosis}, NORMAL: {normal}"
                    )
                print(
                    "TURBERCULOSIS - итоговая точность: ",
                    round(
                        (turberculosis / len(listdir("train_ds/val/TURBERCULOSIS"))), 3
                    )
                    * 100,
                    "%",
                )
                accuracy_fc.append(
                    {
                        "Turberculosis": round(
                            (
                                turberculosis
                                / len(listdir("train_ds/val/TURBERCULOSIS"))
                            ),
                            3,
                        )
                        * 100
                    }
                )
                turberculosis_cm = [covid, pneumonia, turberculosis, normal]
            elif i == "NORMAL":
                covid, pneumonia, turberculosis, normal = 0, 0, 0, 0
                print(i, "", len(listdir(f"train_ds/val/{i}")))
                for j in listdir(f"train_ds/val/{i}"):
                    if app.xray_predict(f"train_ds/val/{i}/{j}")[3] == "COVID":
                        covid += 1
                    elif app.xray_predict(f"train_ds/val/{i}/{j}")[3] == "пневмония":
                        pneumonia += 1
                    elif app.xray_predict(f"train_ds/val/{i}/{j}")[3] == "туберкулёз":
                        turberculosis += 1
                    elif app.xray_predict(f"train_ds/val/{i}/{j}")[3] == "всё хорошо":
                        normal += 1
                    print(
                        f"COVID: {covid}, PNEUMONIA: {pneumonia}, TURBERCULOSIS: {turberculosis}, NORMAL: {normal}"
                    )
                print(
                    "NORMAL - итоговая точность: ",
                    round((normal / len(listdir("train_ds/val/NORMAL"))), 3) * 100,
                    "%",
                )
                accuracy_fc.append(
                    {
                        "Normal": round(
                            (normal / len(listdir("train_ds/val/NORMAL"))), 3
                        )
                        * 100
                    }
                )
                normal_cm = [covid, pneumonia, turberculosis, normal]
        covid_acc = accuracy_fc[0].get("Covid")
        pneumonia_acc = accuracy_fc[2].get("Pneumonia")
        turberculosis_acc = accuracy_fc[3].get("Turberculosis")
        normal_acc = accuracy_fc[1].get("Normal")
        sns.heatmap(
            [covid_cm, pneumonia_cm, turberculosis_cm, normal_cm],
            annot=True,
            fmt="g",
            cmap="Dark2_r",
            linecolor="b",
            xticklabels=["covid", "pneumonia", "turberculosis", "normal"],
            yticklabels=["covid", "pneumonia", "turberculosis", "normal"],
        ).set(
            xlabel=f"Accuracy:"
            f"\n Covid: {covid_acc}%"
            f"\nPneumonia: {pneumonia_acc}%"
            f"\nTurberculosis: {turberculosis_acc}%"
            f"\nNormal: {normal_acc}%"
            f"\nTotal: {round(sum([pneumonia_acc, normal_acc, turberculosis_acc, covid_acc]) / 4, 2)}%"
        )

        plt.show()

    def ct_model_create(self):
        train_ds = tf.keras.utils.image_dataset_from_directory(
            self.ct_path,
            validation_split=self.validation_split,
            subset="training",
            batch_size=self.batch_size,
            seed=123,
            image_size=(self.ct_img_height, self.ct_img_width),
        )
        val_ds = tf.keras.utils.image_dataset_from_directory(
            self.ct_path,
            validation_split=self.validation_split,
            subset="validation",
            batch_size=self.batch_size,
            seed=123,
            image_size=(self.ct_img_height, self.ct_img_width),
        )

        class_names = train_ds.class_names

        model = models.Sequential(
            [
                layers.RandomFlip(
                    "horizontal", input_shape=(self.ct_img_height, self.ct_img_width, 3)
                ),
                layers.RandomRotation(0.1),
                layers.RandomZoom(0.1),
                layers.Rescaling(1.0 / 255),
                layers.Conv2D(64, (3, 3), padding="same", activation="relu"),
                layers.MaxPooling2D((2, 2)),
                layers.Conv2D(128, (3, 3), padding="same", activation="relu"),
                layers.MaxPooling2D((2, 2)),
                layers.Conv2D(128, (3, 3), padding="same", activation="relu"),
                layers.MaxPooling2D((2, 2)),
                layers.Conv2D(128, (3, 3), padding="same", activation="relu"),
                layers.MaxPooling2D((2, 2)),
                layers.Dropout(0.25),
                layers.Flatten(),
                layers.Dense(512, activation="relu"),
                layers.Dense(len(class_names), activation="softmax"),
            ]
        )

        model.compile(
            optimizer="adam",
            loss="sparse_categorical_crossentropy",
            metrics=["accuracy"],
        )

        model.fit(train_ds, validation_data=val_ds, epochs=self.epochs)

        model.save("ct_model_beta.h5")

    def ct_predict(self, path_to_image):
        model = keras.models.load_model("models/ct_model.h5")
        img = tf.keras.utils.load_img(
            path_to_image, target_size=(self.ct_img_height, self.ct_img_width)
        )

        img_array = tf.keras.utils.img_to_array(img)
        img_array = tf.expand_dims(img_array, 0)
        predictions = model.predict(img_array)
        score = tf.nn.softmax(predictions[0])

        if self.ct_class_names[np.argmax(score)] == "1NonCOVID":
            self.ct_class_names[np.argmax(score)] = "всё хорошо"
        elif self.ct_class_names[np.argmax(score)] == "2COVID":
            self.ct_class_names[np.argmax(score)] = "Covid-19"
        elif self.ct_class_names[np.argmax(score)] == "3CAP":
            self.ct_class_names[np.argmax(score)] = "пневмония"

        return [
            self.ct_class_names,
            np.array(score * 100),
            100 * np.max(score),
            self.ct_class_names[np.argmax(score)],
        ]

    @staticmethod
    def ct_metrics():
        covid_cm = []
        pneumonia_cm = []
        normal_cm = []

        accuracy_fc = []

        for i in listdir(f"ct_ds/test/"):
            if i == "2COVID":
                covid, pneumonia, normal = 0, 0, 0
                print(i, "", len(listdir(f"ct_ds/test/{i}")))
                for j in listdir(f"ct_ds/test/{i}"):
                    if app.ct_predict(f"ct_ds/test/{i}/{j}")[3] == "Covid-19":
                        covid += 1
                    elif app.ct_predict(f"ct_ds/test/{i}/{j}")[3] == "пневмония":
                        pneumonia += 1
                    elif app.ct_predict(f"ct_ds/test/{i}/{j}")[3] == "всё хорошо":
                        normal += 1
                    print(f"COVID: {covid}, PNEUMONIA: {pneumonia}, NORMAL: {normal}")
                print(
                    "COVID - итоговая точность: ",
                    round((covid / len(listdir("ct_ds/test/2COVID"))), 3) * 100,
                    "%",
                )
                accuracy_fc.append(
                    {
                        "Covid": round((covid / len(listdir("ct_ds/test/2COVID"))), 3)
                        * 100
                    }
                )
                covid_cm = [covid, pneumonia, normal]

            elif i == "1NonCOVID":
                covid, pneumonia, normal = 0, 0, 0
                print(i, "", len(listdir(f"ct_ds/test/{i}")))
                for j in listdir(f"ct_ds/test/{i}"):
                    if app.ct_predict(f"ct_ds/test/{i}/{j}")[3] == "Covid-19":
                        covid += 1
                    elif app.ct_predict(f"ct_ds/test/{i}/{j}")[3] == "пневмония":
                        pneumonia += 1
                    elif app.ct_predict(f"ct_ds/test/{i}/{j}")[3] == "всё хорошо":
                        normal += 1
                    print(f"COVID: {covid}, PNEUMONIA: {pneumonia}, NORMAL: {normal}")
                print(
                    "NORMAL - итоговая точность: ",
                    round((normal / len(listdir("ct_ds/test/1NonCOVID"))), 3) * 100,
                    "%",
                )
                accuracy_fc.append(
                    {
                        "Normal": round(
                            (normal / len(listdir("ct_ds/test/1NonCOVID"))), 3
                        )
                        * 100
                    }
                )
                normal_cm = [covid, pneumonia, normal]

            elif i == "3CAP":
                covid, pneumonia, normal = 0, 0, 0
                print(i, "", len(listdir(f"ct_ds/test/{i}")))
                for j in listdir(f"ct_ds/test/{i}"):
                    if app.ct_predict(f"ct_ds/test/{i}/{j}")[3] == "Covid-19":
                        covid += 1
                    elif app.ct_predict(f"ct_ds/test/{i}/{j}")[3] == "пневмония":
                        pneumonia += 1
                    elif app.ct_predict(f"ct_ds/test/{i}/{j}")[3] == "всё хорошо":
                        normal += 1
                    print(f"COVID: {covid}, PNEUMONIA: {pneumonia}, NORMAL: {normal}")
                print(
                    "PNEUMONIA - итоговая точность: ",
                    round((pneumonia / len(listdir("ct_ds/test/3CAP"))), 3) * 100,
                    "%",
                )
                accuracy_fc.append(
                    {
                        "Pneumonia": round(
                            (pneumonia / len(listdir("ct_ds/test/3CAP"))), 3
                        )
                        * 100
                    }
                )
                pneumonia_cm = [covid, pneumonia, normal]

        covid_acc = accuracy_fc[1].get("Covid")
        pneumonia_acc = accuracy_fc[2].get("Pneumonia")
        normal_acc = accuracy_fc[0].get("Normal")

        sns.heatmap(
            [covid_cm, pneumonia_cm, normal_cm],
            annot=True,
            fmt="g",
            cmap="Dark2_r",
            linecolor="b",
            xticklabels=["covid", "pneumonia", "normal"],
            yticklabels=["covid", "pneumonia", "normal"],
        ).set(
            xlabel=f"Accuracy:"
            f"\n Covid: {covid_acc}%\n"
            f"Pneumonia: {pneumonia_acc}%\n"
            f"Normal: {normal_acc}%\n"
            f"Total: {round(sum([pneumonia_acc, normal_acc, covid_acc]) / 3, 2)}% "
        )


app = RayTensor()
